import gym
import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
import streamlit as st
from stable_baselines3 import DQN
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
from itertools import product


# --- Helper Functions ---
def generate_dynamic_adjustments(original_delay, congestion_level, track_availability):
    """
    Generate scenario-based delay adjustments based on delay, congestion, and track availability.
    Args:
        original_delay (float): Current delay in minutes.
        congestion_level (int): Encoded congestion level (0: low, 1: medium, 2: high).
        track_availability (int): Encoded track availability (0: low, 1: high).
    Returns:
        list: List of possible delay adjustments (non-positive).
    """
    base_adjustments = [0]  # Always allow no adjustment
    if original_delay <= 5:
        adjustments = [-1, -2] if track_availability == 1 else [-1]
    elif original_delay <= 15:
        adjustments = [-2, -3, -4] if congestion_level <= 1 else [-2, -3]
    else:
        adjustments = [-3, -4, -5, -6] if congestion_level == 0 and track_availability == 1 else [-3, -4]

    return sorted(base_adjustments + adjustments, reverse=True)


# --- Data Loading, Preprocessing, RL Environment, Training, Prediction ---
@st.cache_resource(show_spinner=False)
def load_data_train_and_predict(_hyperparams):
    """
    Loads data, preprocesses, trains DQN model with hyperparameter tuning, and performs predictions.
    Args:
        _hyperparams (dict): Hyperparameters for DQN (learning_rate, exploration_fraction).
    Returns:
        tuple: (df_result, q_df, best_params, train_groups)
    """
    status_message = st.empty()
    status_message.text("Loading and preprocessing data...")

    # Load Dataset
    try:
        df = pd.read_csv("Corrected_Time_Table.csv", low_memory=False)
        if df.empty:
            status_message.warning("Dataset is empty.")
            return pd.DataFrame(), pd.DataFrame(), {}, {}
    except FileNotFoundError:
        status_message.error("Error: Corrected_Time_Table.csv not found.")
        st.stop()
    except Exception as e:
        status_message.error(f"Error loading dataset: {e}")
        st.stop()

    # Convert time columns to minutes
    def convert_time_to_minutes(time_str):
        try:
            if pd.isna(time_str) or not str(time_str).strip():
                return np.nan
            time_obj = pd.to_datetime(str(time_str), errors='coerce').time()
            if time_obj is None:
                return np.nan
            return time_obj.hour * 60 + time_obj.minute
        except Exception:
            return np.nan

    time_columns = ["Scheduled Arrival Time", "Scheduled Departure Time", "Actual Arrival Time",
                    "Actual Departure Time"]
    for col in time_columns:
        df[col] = df[col].apply(convert_time_to_minutes)

    # Impute missing time values with median
    for col in time_columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Handle categorical columns
    categorical_cols = ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    # Features for RL
    FEATURES = [
        "Scheduled Arrival Time", "Scheduled Departure Time", "Distance",
        "Delay Status", "Station Congestion", "Track Availability",
        "Peak Hour Indicator", "Halt Time (min)", "Original Delay"
    ]

    # Impute missing numerical values with median
    for feature in FEATURES:
        if feature not in df.columns:
            status_message.error(f"Error: Feature '{feature}' not found.")
            st.stop()
        if feature not in categorical_cols and df[feature].isnull().any():
            median_value = df[feature].median()
            df[feature] = df[feature].fillna(median_value)

    # Ensure numeric types
    for feature in FEATURES:
        if feature not in categorical_cols:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(df[feature].median())

    X = df[FEATURES].values.astype(np.float32)
    y = df["Original Delay"].values

    # Group data by Train No for multi-step episodes
    train_groups = {train_no: group.sort_values("SEQ") for train_no, group in df.groupby("Train No")}

    # Custom Gym Environment
    class TrainSchedulingEnv(gym.Env):
        def __init__(self, train_groups, features):
            super(TrainSchedulingEnv, self).__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(5)  # Max number of adjustments
            self.train_groups = train_groups
            self.features = features
            self.current_train = None
            self.current_idx = 0
            self.state = None
            self.train_ids = list(train_groups.keys())
            self.max_steps = max(len(group) for group in train_groups.values())

        def reset(self):
            self.current_train = random.choice(self.train_ids)
            self.current_idx = 0
            self.state = self.train_groups[self.current_train][self.features].iloc[self.current_idx].values.astype(
                np.float32)
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            group = self.train_groups[self.current_train]
            row = group.iloc[self.current_idx]
            original_delay_idx = self.features.index("Original Delay")
            congestion_idx = self.features.index("Station Congestion")
            track_idx = self.features.index("Track Availability")

            adjustments = generate_dynamic_adjustments(
                self.state[original_delay_idx],
                self.state[congestion_idx],
                self.state[track_idx]
            )

            # Map action to available adjustments
            action = min(action, len(adjustments) - 1)
            adjustment = adjustments[action]
            original_delay = self.state[original_delay_idx]
            new_delay = max(0, min(original_delay + adjustment, original_delay))  # Ensure new_delay <= original_delay

            # Reward function
            delay_reduction = original_delay - new_delay
            congestion_penalty = -0.5 * row["Station Congestion"] * abs(adjustment)
            track_bonus = 0.2 * row["Track Availability"] * delay_reduction
            downstream_impact = -0.1 * new_delay if self.current_idx < len(group) - 1 else 0
            increase_penalty = -10.0 if new_delay > original_delay else 0.0  # Strong penalty for increasing delay
            reward = delay_reduction + congestion_penalty + track_bonus + downstream_impact + increase_penalty

            # Update state
            self.state[original_delay_idx] = new_delay
            self.current_idx += 1

            # Check if episode is done
            done = self.current_idx >= len(group) or self.current_idx >= self.max_steps
            if not done:
                self.state = group[self.features].iloc[self.current_idx].values.astype(np.float32)

            return np.array(self.state, dtype=np.float32), reward, done, {"adjustment": adjustment}

    status_message.text("Tuning and training DQN model...")

    # Hyperparameter tuning
    learning_rates = [1e-3, 5e-4, 1e-4]
    exploration_fractions = [0.3, 0.5, 0.7]
    best_params = None
    best_reward = -np.inf
    env = TrainSchedulingEnv(train_groups, FEATURES)

    for lr, exp_frac in product(learning_rates, exploration_fractions):
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            exploration_fraction=exp_frac,
            exploration_final_eps=0.05,
            verbose=0
        )
        model.learn(total_timesteps=50000)
        rewards = []
        for _ in range(10):
            obs = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        avg_reward = np.mean(rewards)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = {"learning_rate": lr, "exploration_fraction": exp_frac}
            best_model = model

    status_message.text("Predicting optimized delays...")

    # Prediction
    optimized_delays = []
    q_value_logs = []
    with torch.no_grad():
        for train_no, group in train_groups.items():
            states = group[FEATURES].values.astype(np.float32)
            for i in range(len(group)):
                state = states[i]
                adjustments = generate_dynamic_adjustments(
                    state[FEATURES.index("Original Delay")],
                    state[FEATURES.index("Station Congestion")],
                    state[FEATURES.index("Track Availability")]
                )
                state_tensor = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32).to(best_model.device)
                q_values = best_model.policy.q_net(state_tensor).cpu().numpy()[0]

                # Truncate Q-values to match available adjustments
                valid_q_values = q_values[:len(adjustments)]
                action_index = np.argmax(valid_q_values)
                delay_adjustment = adjustments[action_index]
                original_delay = group.iloc[i]["Original Delay"]
                optimized_delay = max(0, min(original_delay + delay_adjustment,
                                             original_delay))  # Ensure optimized_delay <= original_delay

                # Compute reward for logging
                delay_reduction = original_delay - optimized_delay
                congestion_penalty = -0.5 * group.iloc[i]["Station Congestion"] * abs(delay_adjustment)
                track_bonus = 0.2 * group.iloc[i]["Track Availability"] * delay_reduction
                downstream_impact = -0.1 * optimized_delay if i < len(group) - 1 else 0
                increase_penalty = -10.0 if optimized_delay > original_delay else 0.0
                reward = delay_reduction + congestion_penalty + track_bonus + downstream_impact + increase_penalty

                optimized_delays.append(optimized_delay)
                q_value_logs.append({
                    "Train No": train_no,
                    "Station": group.iloc[i]["Station Name"],
                    "Original Delay": original_delay,
                    "Q-Values": valid_q_values.tolist(),
                    "Selected Action Index": action_index,
                    "Delay Adjustment (mins)": delay_adjustment,
                    "Optimized Delay": optimized_delay,
                    "Adjustments Available": adjustments,
                    "Reward": reward
                })

    df_result = df.copy()
    df_result["Optimized Delay"] = optimized_delays
    q_df = pd.DataFrame(q_value_logs)

    status_message.text("Processing complete.")
    status_message.empty()

    return df_result, q_df, best_params, train_groups


# --- Streamlit App ---
st.title("Train Delay Optimization using DQN")
st.write("Optimizes train delays using a DQN with multi-step episodes and scenario-based adjustments.")

# Hyperparameters for tuning
hyperparams = {"learning_rate": 1e-3, "exploration_fraction": 0.5}
df_optimized, q_df_analysis, best_params, train_groups = load_data_train_and_predict(hyperparams)

if df_optimized.empty:
    st.error("Could not load or process data.")
else:
    # Metrics
    st.subheader("Overall Delay Optimization Metrics")
    total_original_delay = df_optimized["Original Delay"].sum()
    total_optimized_delay = df_optimized["Optimized Delay"].sum()
    avg_original_delay = df_optimized["Original Delay"].mean()
    avg_optimized_delay = df_optimized["Optimized Delay"].mean()
    total_reduction = total_original_delay - total_optimized_delay
    percentage_reduction = (total_reduction / total_original_delay) * 100 if total_original_delay > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Original Delay (min)", f"{avg_original_delay:.2f}")
    with col2:
        st.metric("Avg Optimized Delay (min)", f"{avg_optimized_delay:.2f}",
                  delta=f"{-(avg_original_delay - avg_optimized_delay):.2f}")
    with col3:
        st.metric("Overall % Reduction", f"{percentage_reduction:.2f}%")

    st.write(f"Total Delay Reduced: {total_reduction:.0f} min")
    st.write(f"Best Hyperparameters: {best_params}")

    # Train Selection
    st.subheader("Individual Train Delay Comparison")
    unique_trains = df_optimized.drop_duplicates(subset=["Train No", "Train Name"])
    unique_trains["Train No"] = unique_trains["Train No"].astype(str)
    dropdown_options = unique_trains.apply(
        lambda row: {"label": f"{row['Train No']} - {row['Train Name']}", "value": row['Train No']},
        axis=1
    ).tolist()

    if dropdown_options:
        default_train_value = str(df_optimized["Train No"].iloc[0])
        default_index = next((i for i, opt in enumerate(dropdown_options) if opt['value'] == default_train_value), 0)
        selected_option = st.selectbox(
            "Select Train:",
            options=dropdown_options,
            format_func=lambda opt: opt['label'],
            index=default_index
        )
        selected_train_value = selected_option['value']

        filtered_df = df_optimized[df_optimized["Train No"].astype(str) == selected_train_value].sort_values("SEQ")
        if not filtered_df.empty:
            fig = go.Figure()
            filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values(
                "Station")
            fig.add_trace(go.Bar(
                x=filtered_df["Station Name"],
                y=filtered_df["Original Delay"],
                name="Original Delay",
                marker=dict(color="red"),
                customdata=filtered_df[
                    ["Station Congestion", "Track Availability", "Peak Hour Indicator", "Delay Status"]].assign(
                    Action=filtered_q_df["Delay Adjustment (mins)"],
                    Reward=filtered_q_df["Reward"]
                ),
                hovertemplate=(
                        "<b>Station:</b> %{x}<br>" +
                        "<b>Original Delay:</b> %{y} min<br>" +
                        "<b>Congestion:</b> %{customdata[0]}<br>" +
                        "<b>Track Availability:</b> %{customdata[1]}<br>" +
                        "<b>Peak Hour:</b> %{customdata[2]}<br>" +
                        "<b>Delay Status:</b> %{customdata[3]}<br>" +
                        "<b>Action:</b> %{customdata[4]} min<br>" +
                        "<b>Reward:</b> %{customdata[5]:.2f}<extra></extra>"
                )
            ))
            fig.add_trace(go.Bar(
                x=filtered_df["Station Name"],
                y=filtered_df["Optimized Delay"],
                name="Optimized Delay",
                marker=dict(color="green"),
                customdata=filtered_df[
                    ["Station Congestion", "Track Availability", "Peak Hour Indicator", "Delay Status"]].assign(
                    Action=filtered_q_df["Delay Adjustment (mins)"],
                    Reward=filtered_q_df["Reward"]
                ),
                hovertemplate=(
                        "<b>Station:</b> %{x}<br>" +
                        "<b>Optimized Delay:</b> %{y} min<br>" +
                        "<b>Congestion:</b> %{customdata[0]}<br>" +
                        "<b>Track Availability:</b> %{customdata[1]}<br>" +
                        "<b>Peak Hour:</b> %{customdata[2]}<br>" +
                        "<b>Delay Status:</b> %{customdata[3]}<br>" +
                        "<b>Action:</b> %{customdata[4]} min<br>" +
                        "<b>Reward:</b> %{customdata[5]:.2f}<extra></extra>"
                )
            ))
            fig.update_layout(
                title=f"Delay Comparison for Train {selected_train_value}",
                xaxis_title="Station Name",
                yaxis_title="Delay (min)",
                xaxis=dict(tickangle=-45),
                barmode="group",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data for Train {selected_train_value}.")
    else:
        st.warning("No trains found.")

    # Detailed Station Analysis
    st.subheader("Station-Level Analysis for Selected Train")
    if not q_df_analysis.empty and dropdown_options:
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values(
            "Station")
        if not filtered_q_df.empty:
            st.write(f"Detailed analysis for Train {selected_train_value}:")
            display_df = filtered_q_df[[
                "Station", "Original Delay", "Optimized Delay",
                "Delay Adjustment (mins)", "Reward"
            ]].rename(columns={
                "Station": "Station Name",
                "Delay Adjustment (mins)": "Action Taken (min)"
            })
            st.dataframe(display_df.reset_index(drop=True))
        else:
            st.warning(f"No analysis data for Train {selected_train_value}.")
    else:
        st.info("No analysis data available.")

    # Q-Value Analysis
    st.subheader("DQN Prediction Analysis (Sample)")
    if not q_df_analysis.empty:
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values(
            "Station")
        if not filtered_q_df.empty:
            st.write(f"Q-value analysis for Train {selected_train_value}:")
            st.dataframe(filtered_q_df.head(10).reset_index(drop=True))
        else:
            st.write("Showing first 10 entries:")
            st.dataframe(q_df_analysis.head(10).reset_index(drop=True))

    # Q-Value Plot
    if not q_df_analysis.empty:
        sample = q_df_analysis.iloc[0]
        fig_mpl, ax = plt.subplots(figsize=(10, 5))
        adjustments = sample["Adjustments Available"]
        q_values = sample["Q-Values"]
        bars = ax.bar([str(adj) for adj in adjustments], q_values, color='purple')
        for bar in bars:
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va=va, ha='center')
        ax.set_title(f"Q-values at {sample['Station']} (Train {sample['Train No']})")
        ax.set_xlabel("Delay Adjustment (minutes)")
        ax.set_ylabel("Q-value")
        ax.grid(axis='y', linestyle='--')
        st.subheader("Sample Q-Value Visualization")
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)