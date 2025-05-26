import gym
import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
import streamlit as st
from stable_baselines3 import DQN, DDPG, A2C, SAC, PPO
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import torch
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import time  # Added for timing

# Define action spaces
DISCRETE_ADJUSTMENTS = [0, -1, -2, -3, -4]
CONTINUOUS_ACTION_BOUNDS = [-4, 0]

# --- Custom Replay Buffer for Prioritized Experience Replay ---
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1, alpha=0.6, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, **kwargs)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size, env=None, beta=0.4):
        priorities = self.priorities[:self.buffer_size]
        if self.buffer_size == 0:
            priorities = np.ones_like(priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.buffer_size, batch_size, p=probs)
        samples = super().sample(batch_size, env=env)
        weights = (self.buffer_size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

# --- Custom DQN for Prioritized Replay ---
class CustomDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            samples, indices, weights = self.replay_buffer.sample(batch_size, beta=0.4)
            replay_data = samples
            weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = (weights * (current_q_values - target_q_values.unsqueeze(1)) ** 2).mean()
            td_errors = (current_q_values - target_q_values.unsqueeze(1)).abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors.flatten())
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

# --- Data Loading, Preprocessing, RL Environment, Training, Prediction ---
@st.cache_resource(show_spinner=False)
def load_data_train_and_predict():
    # Start overall timer
    start_time = time.time()
    status_message = st.empty()
    status_message.text("Loading and preprocessing data...")

    # Load Dataset
    try:
        df = pd.read_csv("Corrected_Time_Table.csv", low_memory=False)
        if df.empty:
            status_message.warning("Dataset is empty.")
            return pd.DataFrame(), {}, pd.DataFrame(), {}
    except FileNotFoundError:
        status_message.error("Error: Corrected_Time_Table.csv not found.")
        st.stop()
    except Exception as e:
        status_message.error(f"Error loading dataset: {e}")
        st.stop()

    # Convert time columns to minutes
    def convert_time_to_minutes(time_str):
        try:
            if pd.isna(time_str):
                return 0
            time_str = str(time_str).strip()
            if not time_str:
                return 0
            try:
                time_obj = pd.to_datetime(time_str, format="%H:%M").time()
            except ValueError:
                try:
                    time_obj = pd.to_datetime(time_str).time()
                except ValueError:
                    return 0
            return time_obj.hour * 60 + time_obj.minute
        except Exception:
            return 0

    time_columns = ["Scheduled Arrival Time", "Scheduled Departure Time", "Actual Arrival Time", "Actual Departure Time"]
    for col in time_columns:
        df[col] = df[col].apply(convert_time_to_minutes)

    # Handle missing values and ensure string type for categorical columns
    for col in ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]:
        if df[col].isnull().any():
            print(f"Warning: Missing values found in '{col}'. Filling with 'Unknown'.")
            df[col] = df[col].fillna("Unknown")
        df[col] = df[col].astype(str)

    # Encode categorical features
    categorical_cols = ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]
    label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    # Features for RL
    FEATURES = [
        "Scheduled Arrival Time", "Scheduled Departure Time", "Distance",
        "Delay Status", "Station Congestion", "Track Availability",
        "Peak Hour Indicator", "Halt Time (min)", "Original Delay"
    ]

    for feature in FEATURES:
        if feature not in df.columns:
            status_message.error(f"Error: Feature '{feature}' not found in the dataset.")
            st.stop()
        if feature not in categorical_cols and df[feature].isnull().any():
            print(f"Warning: Missing values found in numerical feature '{feature}'. Filling with 0.")
            df[feature] = df[feature].fillna(0)

    for feature in FEATURES:
        if feature not in categorical_cols:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

    # Ensure SEQ is numeric
    if "SEQ" in df.columns:
        df["SEQ"] = pd.to_numeric(df["SEQ"], errors='coerce').fillna(0)
    else:
        status_message.error("Error: 'SEQ' column not found in the dataset.")
        st.stop()

    # Validate Original Delay (prevent negative values)
    if (df["Original Delay"] < 0).any():
        print("Warning: Negative Original Delay values found. Setting to 0.")
        df["Original Delay"] = df["Original Delay"].clip(lower=0)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values.astype(np.float32))
    y = df["Original Delay"].values

    if len(X) == 0:
        status_message.warning("No valid data rows found after preprocessing.")
        return pd.DataFrame(), {}, pd.DataFrame(), {}

    # Custom Gym Environment
    class TrainSchedulingEnv(gym.Env):
        def __init__(self, df, X, discrete=True):
            super(TrainSchedulingEnv, self).__init__()
            self.df = df
            self.X = X
            self.discrete = discrete
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(FEATURES),), dtype=np.float32)
            if discrete:
                self.action_space = gym.spaces.Discrete(len(DISCRETE_ADJUSTMENTS))
            else:
                self.action_space = gym.spaces.Box(low=CONTINUOUS_ACTION_BOUNDS[0], high=CONTINUOUS_ACTION_BOUNDS[1], shape=(1,), dtype=np.float32)
            self.current_train = None
            self.current_indices = []
            self.current_step = 0
            self.prev_delay = 0

        def reset(self):
            self.current_train = random.choice(self.df["Train No"].unique())
            self.current_indices = self.df[self.df["Train No"] == self.current_train].index.tolist()
            self.current_indices.sort(key=lambda i: self.df.loc[i, "SEQ"])
            self.current_step = 0
            self.prev_delay = 0
            if not self.current_indices:
                return np.zeros(len(FEATURES), dtype=np.float32)
            state = self.X[self.current_indices[0]].copy()
            return state

        def step(self, action):
            if self.current_step >= len(self.current_indices):
                return np.zeros(len(FEATURES), dtype=np.float32), 0, True, {}

            idx = self.current_indices[self.current_step]
            state = self.X[idx]
            original_delay = self.df.loc[idx, "Original Delay"]
            congestion = state[FEATURES.index("Station Congestion")]
            track_availability = state[FEATURES.index("Track Availability")]

            if self.discrete:
                if not self.action_space.contains(action):
                    print(f"Warning: Invalid action received: {action}. Defaulting to 0.")
                    action = 0
                adjustment = DISCRETE_ADJUSTMENTS[action]
            else:
                adjustment = np.clip(action[0], CONTINUOUS_ACTION_BOUNDS[0], CONTINUOUS_ACTION_BOUNDS[1])

            # Apply delay adjustment with partial propagation, capped at original_delay
            new_delay = max(0, min(original_delay, original_delay + adjustment + 0.5 * self.prev_delay))
            delay_reduction = original_delay - new_delay

            # Reward function with penalty for increased delays
            congestion_penalty = 0.3 * congestion * abs(adjustment)
            track_penalty = 0.2 * (1 - track_availability) * abs(adjustment)
            over_optimization_penalty = 0.1 * (adjustment ** 2)
            increase_penalty = -10.0 * (new_delay - original_delay) if new_delay > original_delay else 0
            reward = delay_reduction - congestion_penalty - track_penalty - over_optimization_penalty + increase_penalty

            # Debug log for cases where optimized_delay > original_delay
            if new_delay > original_delay:
                print(f"Train {self.current_train}, Station {self.df.loc[idx, 'Station Name']}: Optimized Delay ({new_delay}) > Original Delay ({original_delay}), Adjustment: {adjustment}, Prev Delay: {self.prev_delay}")

            # Update state
            state[FEATURES.index("Original Delay")] = new_delay
            self.prev_delay = new_delay
            self.current_step += 1

            done = self.current_step >= len(self.current_indices)
            info = {"adjustment": adjustment}

            if not done:
                next_idx = self.current_indices[self.current_step]
                state = self.X[next_idx].copy()
                state[FEATURES.index("Original Delay")] = self.df.loc[next_idx, "Original Delay"] + 0.5 * self.prev_delay

            return state, reward, done, info

    status_message.text("Training RL models...")
    env_discrete = TrainSchedulingEnv(df, X, discrete=True)
    env_continuous = TrainSchedulingEnv(df, X, discrete=False)

    # Train models
    models = {
        "DQN": CustomDQN("MlpPolicy", env_discrete, verbose=0, exploration_fraction=0.8, exploration_final_eps=0.1, replay_buffer_class=PrioritizedReplayBuffer),
        "A2C": A2C("MlpPolicy", env_discrete, verbose=0),
        "PPO": PPO("MlpPolicy", env_discrete, verbose=0),
        "DDPG": DDPG("MlpPolicy", env_continuous, verbose=0),
        "SAC": SAC("MlpPolicy", env_continuous, verbose=0)
    }

    # Dictionary to store timing results
    timing_results = {"Training": {}, "Evaluation": {}}

    for name, model in models.items():
        # Time training
        status_message.text(f"Training {name}...")
        train_start = time.time()
        model.learn(total_timesteps=50000)
        train_time = time.time() - train_start
        timing_results["Training"][name] = train_time
        status_message.text(f"Trained {name} in {train_time:.2f} seconds")

    # Custom Ensemble Policy
    class EnsemblePolicy:
        def __init__(self, dqn_model, sac_model):
            self.dqn_model = dqn_model
            self.sac_model = sac_model

        def predict(self, state):
            original_delay = state[FEATURES.index("Original Delay")]
            congestion = state[FEATURES.index("Station Congestion")]
            if original_delay > 4 or congestion < 0.5:
                action, _ = self.dqn_model.predict(state, deterministic=True)
                adjustment = DISCRETE_ADJUSTMENTS[action]
            else:
                action, _ = self.sac_model.predict(state, deterministic=True)
                adjustment = action[0]
            return adjustment

    ensemble = EnsemblePolicy(models["DQN"], models["SAC"])
    models["Ensemble"] = ensemble

    # Evaluate models
    results = {}
    for name, model in models.items():
        # Time evaluation
        status_message.text(f"Evaluating {name}...")
        eval_start = time.time()
        optimized_delays = []
        adjustments = []
        rewards = []
        for train_no in df["Train No"].unique():
            indices = df[df["Train No"] == train_no].index.tolist()
            indices.sort(key=lambda i: df.loc[i, "SEQ"])
            state = X[indices[0]].copy()
            prev_delay = 0
            for idx in indices:
                if name == "Ensemble":
                    adjustment = model.predict(state)
                else:
                    action, _ = model.predict(state, deterministic=True)
                    adjustment = DISCRETE_ADJUSTMENTS[action] if name in ["DQN", "A2C", "PPO"] else action[0]
                original_delay = df.loc[idx, "Original Delay"]
                new_delay = max(0, min(original_delay, original_delay + adjustment + 0.5 * prev_delay))
                optimized_delays.append(new_delay)
                adjustments.append(adjustment)
                congestion = state[FEATURES.index("Station Congestion")]
                track_availability = state[FEATURES.index("Track Availability")]
                delay_reduction = original_delay - new_delay
                reward = delay_reduction - 0.3 * congestion * abs(adjustment) - 0.2 * (1 - track_availability) * abs(adjustment) - 0.1 * (adjustment ** 2)
                rewards.append(reward)
                # Debug log for evaluation
                if new_delay > original_delay:
                    print(f"Evaluation - Train {train_no}, Station {df.loc[idx, 'Station Name']}: Optimized Delay ({new_delay}) > Original Delay ({original_delay}), Adjustment: {adjustment}, Prev Delay: {prev_delay}")
                prev_delay = new_delay
                if idx != indices[-1] and indices.index(idx) + 1 < len(indices):
                    state = X[indices[indices.index(idx) + 1]].copy()
                    state[FEATURES.index("Original Delay")] = df.loc[indices[indices.index(idx) + 1], "Original Delay"] + 0.5 * prev_delay
        eval_time = time.time() - eval_start
        timing_results["Evaluation"][name] = eval_time
        status_message.text(f"Evaluated {name} in {eval_time:.2f} seconds")
        results[name] = {
            "optimized_delays": optimized_delays,
            "adjustments": adjustments,
            "avg_reward": np.mean(rewards),
            "total_delay_reduction": df["Original Delay"].sum() - sum(optimized_delays),
            "action_counts": pd.Series(adjustments).value_counts().to_dict()
        }

    df_result = df.copy()
    best_model = max(results, key=lambda k: results[k]["total_delay_reduction"])
    df_result["Optimized Delay"] = results[best_model]["optimized_delays"]
    df_result["Adjustment"] = results[best_model]["adjustments"]

    # Calculate total execution time
    total_time = time.time() - start_time
    timing_results["Total"] = total_time

    status_message.text("Prediction analysis complete.")
    status_message.empty()
    return df_result, results, df, timing_results

# --- Streamlit App Layout ---
st.title("Advanced Train Delay Optimization")
st.write("This application uses multiple RL algorithms (DQN, DDPG, A2C, SAC, PPO, and a custom Ensemble) to optimize train delays.")

df_optimized, model_results, df_raw, timing_results = load_data_train_and_predict()

if df_optimized.empty:
    st.error("Could not load or process data. Please check the input file and console logs.")
else:
    # --- Timing Results ---
    st.subheader("Execution Time Analysis")
    st.write(f"**Total Execution Time**: {timing_results['Total']:.2f} seconds")
    st.write("**Training Times**:")
    for model, time_taken in timing_results["Training"].items():
        st.write(f"- {model}: {time_taken:.2f} seconds")
    st.write("**Evaluation Times**:")
    for model, time_taken in timing_results["Evaluation"].items():
        st.write(f"- {model}: {time_taken:.2f} seconds")

    # --- Overall Metrics ---
    st.subheader("Overall Delay Optimization Metrics (Best Model)")
    best_model = max(model_results, key=lambda k: model_results[k]["total_delay_reduction"])
    total_original_delay = df_optimized["Original Delay"].sum()
    total_optimized_delay = df_optimized["Optimized Delay"].sum()
    avg_original_delay = df_optimized["Original Delay"].mean()
    avg_optimized_delay = df_optimized["Optimized Delay"].mean()
    total_reduction = total_original_delay - total_optimized_delay
    avg_reduction_per_stop = avg_original_delay - avg_optimized_delay
    percentage_reduction = 0
    if total_original_delay > 0:
        percentage_reduction = (total_reduction / total_original_delay) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Avg Original Delay (min)", value=f"{avg_original_delay:.2f}")
    with col2:
        st.metric(label="Avg Optimized Delay (min)", value=f"{avg_optimized_delay:.2f}", delta=f"{-avg_reduction_per_stop:.2f} min")
    with col3:
        st.metric(label="Overall % Reduction", value=f"{percentage_reduction:.2f}%")

    st.write(f"**Best Model**: {best_model}")
    st.write(f"Total Original Delay: {total_original_delay:.0f} min")
    st.write(f"Total Optimized Delay: {total_optimized_delay:.0f} min")
    st.write(f"**Total Delay Reduced**: {total_reduction:.0f} min")

    # --- Model Comparison ---
    st.subheader("Model Performance Comparison")
    comparison_data = {
        "Model": [],
        "Avg Reward": [],
        "Total Delay Reduction (min)": [],
        "Action Diversity (Unique Actions)": []
    }
    for name, result in model_results.items():
        comparison_data["Model"].append(name)
        comparison_data["Avg Reward"].append(result["avg_reward"])
        comparison_data["Total Delay Reduction (min)"].append(result["total_delay_reduction"])
        comparison_data["Action Diversity (Unique Actions)"].append(len(result["action_counts"]))
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)

    # --- Action Distribution ---
    st.subheader("Action Distribution (Best Model)")
    action_counts = model_results[best_model]["action_counts"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(action_counts.keys()),
        y=list(action_counts.values()),
        marker_color='purple'
    ))
    fig.update_layout(
        title=f"Action Distribution for {best_model}",
        xaxis_title="Adjustment (minutes)",
        yaxis_title="Count",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Individual Train Visualization ---
    st.subheader("Individual Train Delay Comparison")
    unique_trains = df_optimized.drop_duplicates(subset=["Train No", "Train Name"])
    unique_trains["Train No"] = unique_trains["Train No"].astype(str)
    dropdown_options = unique_trains.apply(
        lambda row: {"label": f"{row['Train No']} - {row['Train Name']}", "value": row['Train No']},
        axis=1
    ).tolist()

    if not dropdown_options:
        st.warning("No trains found in the data with unique Train No and Name.")
        selected_train_value = None
    else:
        default_train_value_str = str(df_optimized["Train No"].iloc[0]) if not df_optimized.empty else None
        default_index = next((i for i, opt in enumerate(dropdown_options) if opt['value'] == default_train_value_str), 0) if default_train_value_str else 0
        selected_option = st.selectbox(
            "Select Train:",
            options=dropdown_options,
            format_func=lambda option: option['label'],
            index=default_index
        )
        selected_train_value = selected_option['value'] if selected_option else None

    if selected_train_value is not None:
        filtered_df = df_optimized[df_optimized["Train No"].astype(str) == selected_train_value].sort_values(by="SEQ")
        if filtered_df.empty:
            st.warning(f"No station data found for Train Number: {selected_train_value}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=filtered_df["Station Name"], y=filtered_df["Original Delay"],
                                name="Original Delay", marker=dict(color="red")))
            fig.add_trace(go.Bar(x=filtered_df["Station Name"], y=filtered_df["Optimized Delay"],
                                name="Optimized Delay", marker=dict(color="green")))
            fig.update_layout(
                title=f"Original vs Optimized Delay for Train {selected_train_value} (Model: {best_model})",
                xaxis_title="Station Name",
                yaxis_title="Delay (min)",
                xaxis=dict(tickangle=-45),
                barmode="group",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Interactive Train Scheduling Simulation ---
    st.subheader("Interactive Train Scheduling Simulation")
    if selected_train_value is not None:
        sim_filtered_df = df_optimized[df_optimized["Train No"].astype(str) == selected_train_value].sort_values(by="SEQ")
        if sim_filtered_df.empty:
            st.warning(f"No data available for simulation for Train Number: {selected_train_value}.")
        else:
            if 'sim_step' not in st.session_state:
                st.session_state.sim_step = 0
            if 'sim_train' not in st.session_state or st.session_state.sim_train != selected_train_value:
                st.session_state.sim_step = 0
                st.session_state.sim_train = selected_train_value

            total_stations = len(sim_filtered_df)
            current_step = st.session_state.sim_step

            col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 3])
            with col_nav1:
                if st.button("Previous Station", disabled=(current_step <= 0)):
                    st.session_state.sim_step = max(0, current_step - 1)
                    st.rerun()
            with col_nav2:
                if st.button("Next Station", disabled=(current_step >= total_stations - 1)):
                    st.session_state.sim_step = min(total_stations - 1, current_step + 1)
                    st.rerun()

            if current_step < total_stations:
                current_row = sim_filtered_df.iloc[current_step]
                station = current_row["Station Name"]
                original_delay = current_row["Original Delay"]
                optimized_delay = current_row["Optimized Delay"]
                adjustment = current_row["Adjustment"]

                # Timeline visualization
                timeline_data = []
                base_date = datetime(2025, 5, 18, 0, 0)
                for i, row in sim_filtered_df.iterrows():
                    status = "Future"
                    if i < current_step:
                        status = "Past"
                    elif i == current_step:
                        status = "Current"
                    try:
                        arrival_minutes = float(row["Scheduled Arrival Time"]) if float(row["Scheduled Arrival Time"]) > 0 else float(row["SEQ"]) * 10
                    except (ValueError, TypeError):
                        st.warning(f"Non-numeric value in 'Scheduled Arrival Time' or 'SEQ' for station {row['Station Name']}.")
                        arrival_minutes = float(row["SEQ"]) * 10 if pd.notnull(row["SEQ"]) else 0
                    start_time = base_date + timedelta(minutes=arrival_minutes)
                    end_time = start_time + timedelta(minutes=5)
                    timeline_data.append(dict(
                        Task=f"{row['Station Name']} (SEQ {row['SEQ']})",
                        Start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        Finish=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        Resource=status,
                        Original_Delay=row["Original Delay"],
                        Optimized_Delay=row["Optimized Delay"]
                    ))
                colors = {
                    'Past': 'rgb(255, 165, 0)',
                    'Current': 'rgb(0, 200, 0)',
                    'Future': 'rgb(70, 130, 180)'
                }
                timeline_fig = ff.create_gantt(
                    timeline_data,
                    colors=colors,
                    index_col="Resource",
                    show_colorbar=True,
                    group_tasks=True,
                    title="Train Route Timeline"
                )
                for i, trace in enumerate(timeline_fig.data):
                    trace.hoverinfo = "text"
                    trace.text = f"Station: {timeline_data[i]['Task']}<br>" \
                                 f"Original Delay: {timeline_data[i]['Original_Delay']:.0f} min<br>" \
                                 f"Optimized Delay: {timeline_data[i]['Optimized_Delay']:.0f} min"
                timeline_fig.update_layout(
                    xaxis_title="Scheduled Arrival Time",
                    yaxis_title="Station (Sequence)",
                    showlegend=True,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True, autorange="reversed")
                )
                st.plotly_chart(timeline_fig, use_container_width=True)

                # Explanation
                st.write(f"**Station: {station} (Step {current_step + 1}/{total_stations})**")
                st.write(f"- **Original Delay**: {original_delay:.0f} minutes")
                st.write(f"- **Decision**: Adjust delay by {adjustment:.2f} minutes")
                st.write(f"- **Optimized Delay**: {optimized_delay:.0f} minutes")
                st.write(f"The {best_model} model chose an adjustment of {adjustment:.2f} minutes based on features like congestion and track availability.")
            else:
                st.info("Simulation complete. Select a new train or restart.")
    else:
        st.info("Please select a train to start the simulation.")