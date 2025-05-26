import gym
import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
import streamlit as st

# Import a continuous action algorithm, PPO
from stable_baselines3 import PPO
# Import MlpPolicy, typically found within the algorithm's policies submodule for SB3 >= 1.0
try:
    from stable_baselines3.ppo.policies import MlpPolicy
except ImportError:
    # Fallback for older versions if needed
     from stable_baselines3.common.policies import MlpPolicy

# Import LabelEncoder from scikit-learn
from sklearn.preprocessing import LabelEncoder # <-- Added this import

import matplotlib.pyplot as plt
import torch

# Define the range for the continuous delay adjustment action space
MIN_ADJUSTMENT = -5.0 # Max possible reduction (e.g., 5 minutes)
MAX_ADJUSTMENT = 0.0  # No change (0 reduction)

# --- Data Loading, Preprocessing, RL Environment, Training, Prediction (Cached) ---

@st.cache_resource(show_spinner=False)
def load_data_train_and_predict(min_adj, max_adj):
    """
    Loads data, preprocesses, trains PPO model for continuous actions,
    and performs predictions with dynamic adjustment outcomes.
    This function is cached by Streamlit.

    Args:
        min_adj (float): Minimum possible continuous delay adjustment.
        max_adj (float): Maximum possible continuous delay adjustment.

    Returns:
        tuple: A tuple containing:
            - df_result (pd.DataFrame): The original DataFrame with 'Optimized Delay' column added.
            - prediction_analysis_df (pd.DataFrame): DataFrame with detailed prediction steps.
    """
    status_message = st.empty()
    status_message.text("Loading and preprocessing data...")

    try:
        df = pd.read_csv("Corrected_Time_Table.csv", low_memory=False)
        if df.empty:
            status_message.warning("Dataset is empty.")
            return pd.DataFrame(), pd.DataFrame()
    except FileNotFoundError:
        status_message.error("Error: Corrected_Time_Table.csv not found.")
        st.stop()
    except Exception as e:
         status_message.error(f"Error loading dataset: {e}")
         st.stop()

    def convert_time_to_minutes(time_str):
        try:
            if pd.isna(time_str): return 0
            time_str = str(time_str).strip()
            if not time_str: return 0
            try: time_obj = pd.to_datetime(time_str, format="%H:%M").time()
            except ValueError:
                 try: time_obj = pd.to_datetime(time_str).time()
                 except ValueError: return 0
            return time_obj.hour * 60 + time_obj.minute
        except Exception as e:
            return 0

    time_columns = ["Scheduled Arrival Time", "Scheduled Departure Time", "Actual Arrival Time", "Actual Departure Time"]
    for col in time_columns:
        df[col] = df[col].apply(convert_time_to_minutes)

    for col in ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]:
        if df[col].isnull().any():
            print(f"Warning: Missing values found in '{col}'. Filling with 'Unknown'.")
            df[col] = df[col].fillna("Unknown")
        df[col] = df[col].astype(str)

    categorical_cols = ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]
    df_encoded = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    max_encoded_congestion = df_encoded["Station Congestion"].max() if df_encoded["Station Congestion"].nunique() > 1 else 0
    max_encoded_track = df_encoded["Track Availability"].max() if df_encoded["Track Availability"].nunique() > 1 else 0


    FEATURES = [
        "Scheduled Arrival Time", "Scheduled Departure Time", "Distance",
        "Delay Status", "Station Congestion", "Track Availability",
        "Peak Hour Indicator", "Halt Time (min)", "Original Delay"
    ]

    for feature in FEATURES:
        if feature not in df_encoded.columns:
            status_message.error(f"Error: Feature '{feature}' not found in the dataset.")
            st.stop()
        if feature not in categorical_cols and df_encoded[feature].isnull().any():
             print(f"Warning: Missing values found in numerical feature '{feature}'. Filling with 0.")
             df_encoded[feature] = df_encoded[feature].fillna(0)

    for feature in FEATURES:
        if feature not in categorical_cols:
             df_encoded[feature] = pd.to_numeric(df_encoded[feature], errors='coerce').fillna(0)

    X = df_encoded[FEATURES].values.astype(np.float32)
    y = df_encoded["Original Delay"].values

    if len(X) == 0:
         status_message.warning("No valid data rows found after preprocessing.")
         return pd.DataFrame(), pd.DataFrame()

    try:
        idx_original_delay = FEATURES.index("Original Delay")
        idx_congestion = FEATURES.index("Station Congestion")
        idx_track = FEATURES.index("Track Availability")
    except ValueError as e:
         status_message.error(f"Required feature index not found in FEATURES list: {e}")
         st.stop()


    class TrainSchedulingEnv(gym.Env):
        def __init__(self, min_adj, max_adj, features_list, idx_delay, idx_cong, idx_track, max_cong, max_track):
            super(TrainSchedulingEnv, self).__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(features_list),), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=min_adj, high=max_adj, shape=(1,), dtype=np.float32)

            self.state = None
            self.current_idx = 0
            self.min_adj = min_adj
            self.max_adj = max_adj
            self.features_list = features_list
            self.idx_original_delay = idx_delay
            self.idx_congestion = idx_cong
            self.idx_track = idx_track
            self.max_encoded_congestion = max_cong
            self.max_encoded_track = max_track


        def reset(self):
            self.current_idx = random.randint(0, len(X) - 1)
            self.state = X[self.current_idx].copy()
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            intended_adjustment = action[0]
            intended_adjustment = np.clip(intended_adjustment, self.min_adj, self.max_adj)

            original_delay = self.state[self.idx_original_delay]
            current_congestion = self.state[self.idx_congestion]
            # current_track = self.state[self.idx_track]

            # --- Dynamic Calculation of Actual Adjustment based on Conditions ---
            condition_influence_strength = 0.8
            if self.max_encoded_congestion > 0:
                feasibility_factor = 1.0 - (current_congestion / self.max_encoded_congestion) * condition_influence_strength
            else:
                feasibility_factor = 1.0


            if intended_adjustment < 0:
                actual_adjustment = intended_adjustment * feasibility_factor
            else:
                 actual_adjustment = intended_adjustment

            actual_adjustment = np.clip(actual_adjustment, self.min_adj, self.max_adj)

            new_delay = max(0.0, original_delay + actual_adjustment)

            # --- Reward Calculation ---
            reward = original_delay - new_delay

            if self.idx_original_delay is not None:
                 self.state[self.idx_original_delay] = new_delay

            done = True
            info = {}

            return np.array(self.state, dtype=np.float32), reward, done, info


    status_message.text("Training PPO model...")
    env = TrainSchedulingEnv(MIN_ADJUSTMENT, MAX_ADJUSTMENT, FEATURES, idx_original_delay, idx_congestion, idx_track, max_encoded_congestion, max_encoded_track)

    model = PPO(MlpPolicy, env, verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
               )

    model.learn(total_timesteps=50000)

    status_message.text("PPO model training complete. Analyzing predictions...")

    # --- Prediction Phase ---
    prediction_logs = []

    with torch.no_grad():
        for i in range(len(df_encoded)):
            state = np.array(X[i], dtype=np.float32)

            intended_adjustment_array, _ = model.predict(state, deterministic=True)
            intended_adjustment = intended_adjustment_array[0]

            original_delay = y[i]
            current_congestion = state[idx_congestion]
            # current_track = state[idx_track]

            # --- Apply Dynamic Logic for Logging (Matches Environment) ---
            condition_influence_strength = 0.8
            if max_encoded_congestion > 0:
                feasibility_factor = 1.0 - (current_congestion / max_encoded_congestion) * condition_influence_strength
            else:
                feasibility_factor = 1.0

            if intended_adjustment < 0:
                 actual_adjustment = intended_adjustment * feasibility_factor
            else:
                 actual_adjustment = intended_adjustment

            actual_adjustment = np.clip(actual_adjustment, MIN_ADJUSTMENT, MAX_ADJUSTMENT)

            optimized_delay = max(0.0, original_delay + actual_adjustment)

            prediction_logs.append({
                "Train No": df.iloc[i]["Train No"],
                "Station": df.iloc[i]["Station Name"],
                "Original Delay": df.iloc[i]["Original Delay"],
                "Intended Adjustment (mins)": intended_adjustment,
                "Feasibility Factor": feasibility_factor,
                "Actual Adjustment (mins)": actual_adjustment,
                "Optimized Delay": optimized_delay,
                "Station Congestion (Encoded)": current_congestion,
                # "Track Availability (Encoded)": current_track,
            })

    df_result = df.copy()
    df_result["Optimized Delay"] = [log["Optimized Delay"] for log in prediction_logs]

    prediction_analysis_df = pd.DataFrame(prediction_logs)

    status_message.text("Prediction analysis complete.")
    status_message.empty()

    return df_result, prediction_analysis_df

# --- End of Cached Function ---


# --- Streamlit App Layout ---

st.title("Train Delay Optimization using PPO (Continuous Actions)")

st.write("This application uses a Proximal Policy Optimization (PPO) agent to predict optimized train delays with dynamic continuous adjustments based on simulated conditions.")

# Call the cached function
df_optimized, prediction_analysis_df = load_data_train_and_predict(MIN_ADJUSTMENT, MAX_ADJUSTMENT)


if df_optimized.empty:
    st.error("Could not load or process data. Please check the input file and console logs.")
else:
    # --- Display Overall Metrics ---
    st.subheader("Overall Delay Optimization Metrics (Across Dataset)")

    total_original_delay = df_optimized["Original Delay"].sum()
    total_optimized_delay = df_optimized["Optimized Delay"].sum()
    avg_original_delay = df_optimized["Original Delay"].mean()
    avg_optimized_delay = df_optimized["Optimized Delay"].mean()
    total_reduction = total_original_delay - total_optimized_delay
    avg_reduction_per_stop = avg_original_delay - avg_optimized_delay

    percentage_reduction = 0
    if total_original_delay > 0:
        percentage_reduction = (total_reduction / total_original_delay) * 100
    elif total_original_delay == 0 and total_optimized_delay > 0:
        percentage_reduction = -np.inf
    else:
        percentage_reduction = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Avg Original Delay (min)", value=f"{avg_original_delay:.2f}")
    with col2:
        delta_val = avg_reduction_per_stop
        delta_color = "inverse" if delta_val < 0 else "normal" if delta_val > 0 else "off"
        st.metric(label="Avg Optimized Delay (min)", value=f"{avg_optimized_delay:.2f}", delta=f"{-delta_val:.2f} min avg change", delta_color=delta_color)
    with col3:
        perc_delta_val = percentage_reduction
        perc_delta_color = "inverse" if perc_delta_val < 0 else "normal" if perc_delta_val > 0 else "off"
        st.metric(label="Overall % Reduction", value=f"{percentage_reduction:.2f}%", delta_color=perc_delta_color)

    st.write(f"Total Original Delay across all recorded stops: {total_original_delay:.0f} min")
    st.write(f"Total Optimized Delay across all recorded stops: {total_optimized_delay:.0f} min")
    st.write(f"**Total Delay Change:** {total_reduction:.0f} min")


    # --- Individual Train Visualization ---
    st.subheader("Individual Train Delay Comparison")
    st.write("Select a train to visualize original vs. optimized delays per station.")

    unique_trains = df_optimized.drop_duplicates(subset=["Train No", "Train Name"])
    unique_trains["Train No"] = unique_trains["Train No"].astype(str)

    dropdown_options = unique_trains.apply(
        lambda row: {"label": f"{row['Train No']} - {row['Train Name']}", "value": row["Train No"]},
        axis=1
    ).tolist()

    if not dropdown_options:
         st.warning("No trains found in the data.")
         selected_train_value = None
    else:
        default_train_value_str = str(df_optimized["Train No"].iloc[0]) if not df_optimized.empty else None
        default_index = 0
        if default_train_value_str is not None:
            try:
                default_index = next((i for i, opt in enumerate(dropdown_options) if opt['value'] == default_train_value_str), 0)
            except Exception: default_index = 0

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

            fig.update_layout(title=f"Original vs Optimized Delay for Train {selected_train_value}",
                              xaxis_title="Station Name", yaxis_title="Delay (min)",
                              xaxis=dict(tickangle=-45),
                              barmode="group",
                              legend=dict(x=0.01, y=0.99))

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please select a train.")

    # --- Display Prediction Analysis Details ---
    st.subheader("Prediction Analysis Details")
    st.write("Details of the agent's predicted intended adjustment and the actual calculated adjustment based on simulated conditions.")

    if not prediction_analysis_df.empty:
        filtered_analysis_df = prediction_analysis_df[prediction_analysis_df["Train No"].astype(str) == selected_train_value].sort_values(by="Station")

        if not filtered_analysis_df.empty:
             st.write(f"Showing analysis for Train {selected_train_value}:")
             display_cols = [
                 "Station",
                 "Original Delay",
                 "Intended Adjustment (mins)",
                 "Feasibility Factor",
                 "Actual Adjustment (mins)",
                 "Optimized Delay",
                 "Station Congestion (Encoded)"
             ]
             display_cols = [col for col in display_cols if col in filtered_analysis_df.columns]
             st.dataframe(filtered_analysis_df[display_cols].reset_index(drop=True))
        else:
            st.write(f"No detailed analysis data found for the selected Train {selected_train_value}.")
            st.write("Showing overall first 10 entries instead:")
            display_cols = [col for col in display_cols if col in prediction_analysis_df.columns]
            st.dataframe(prediction_analysis_df[display_cols].head(10).reset_index(drop=True))
    else:
        st.info("Prediction analysis data is not available.")

    # --- Matplotlib plot for the first entry (sample prediction breakdown) ---
    if not prediction_analysis_df.empty:
        sample = prediction_analysis_df.iloc[0]
        fig_mpl, ax = plt.subplots(figsize=(10, 5))

        labels = ['Original Delay', 'Intended Adj.', 'Feasibility Factor', 'Actual Adj.', 'Optimized Delay']
        values = [
            sample["Original Delay"],
            sample["Intended Adjustment (mins)"],
            sample["Feasibility Factor"],
            sample["Actual Adjustment (mins)"],
            sample["Optimized Delay"]
        ]
        colors = ['red', 'blue', 'orange', 'purple', 'green']

        ax.bar(labels, values, color=colors)

        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

        ax.set_title(f"Prediction Breakdown for Sample Station: {sample['Station']} (Train {sample['Train No']})")
        ax.set_ylabel("Value")
        ax.grid(axis='y', linestyle='--')
        plt.xticks(rotation=0)

        st.subheader("Sample Prediction Process Visualization (First Dataset Entry)")
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)