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
import plotly.figure_factory as ff
from datetime import datetime, timedelta

# Define delay_adjustments in the global scope
delay_adjustments = [0, -1, -2, -3, -4]

# --- Data Loading, Preprocessing, RL Environment, Training, Prediction (Cached) ---
@st.cache_resource(show_spinner=False)
def load_data_train_and_predict(adj_list):
    status_message = st.empty()
    status_message.text("Loading and preprocessing data...")

    # Load Dataset
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
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

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

    X = df[FEATURES].values.astype(np.float32)
    y = df["Original Delay"].values

    if len(X) == 0:
        status_message.warning("No valid data rows found after preprocessing.")
        return pd.DataFrame(), pd.DataFrame()

    # Custom Gym Environment
    class TrainSchedulingEnv(gym.Env):
        def __init__(self, adj_list):
            super(TrainSchedulingEnv, self).__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(FEATURES),), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(adj_list))
            self.state = None
            self.current_idx = 0
            self.adj_list = adj_list

        def reset(self):
            self.current_idx = random.randint(0, len(X) - 1)
            self.state = X[self.current_idx].copy()
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            if not self.action_space.contains(action):
                print(f"Warning: Invalid action received: {action}. Defaulting to 0.")
                action = 0

            adjustment = self.adj_list[action]
            original_delay_idx = FEATURES.index("Original Delay")
            original_delay = self.state[original_delay_idx]
            new_delay = max(0, original_delay + adjustment)
            reward = original_delay - new_delay
            self.state[original_delay_idx] = new_delay
            done = True
            info = {}
            return np.array(self.state, dtype=np.float32), reward, done, info

    status_message.text("Training DQN model...")
    env = TrainSchedulingEnv(adj_list)
    model = DQN("MlpPolicy", env, verbose=0, exploration_fraction=0.5, exploration_final_eps=0.05)
    model.learn(total_timesteps=20000)
    status_message.text("DQN model training complete. Analyzing predictions...")

    optimized_delays = []
    q_value_logs = []

    with torch.no_grad():
        for i in range(len(df)):
            state = np.array(X[i], dtype=np.float32)
            q_net = model.policy.q_net
            state_tensor = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32).to(model.device)
            q_values_tensor = q_net(state_tensor)
            q_values = q_values_tensor.cpu().numpy()[0]
            action_index = np.argmax(q_values)
            delay_adjustment = adj_list[action_index]
            original_delay = y[i]
            optimized_delay = max(0, original_delay + delay_adjustment)
            optimized_delays.append(optimized_delay)
            q_value_logs.append({
                "Train No": df.iloc[i]["Train No"],
                "Station": df.iloc[i]["Station Name"],
                "Original Delay": original_delay,
                "Q-Values (0,-1,-2,-3,-4 adj)": q_values.tolist(),
                "Selected Action Index": action_index,
                "Delay Adjustment (mins)": delay_adjustment,
                "Optimized Delay": optimized_delay,
            })

    df_result = df.copy()
    df_result["Optimized Delay"] = optimized_delays
    q_df = pd.DataFrame(q_value_logs)
    status_message.text("Prediction analysis complete.")
    status_message.empty()
    return df_result, q_df

# --- Streamlit App Layout ---
st.title("Train Delay Optimization using DQN")
st.write("This application uses a Deep Q-Network (DQN) to predict optimized train delays.")

df_optimized, q_df_analysis = load_data_train_and_predict(delay_adjustments)

if df_optimized.empty:
    st.error("Could not load or process data. Please check the input file and console logs.")
else:
    # --- Overall Metrics ---
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Avg Original Delay (min)", value=f"{avg_original_delay:.2f}")
    with col2:
        st.metric(label="Avg Optimized Delay (min)", value=f"{avg_optimized_delay:.2f}", delta=f"{-avg_reduction_per_stop:.2f} min")
    with col3:
        st.metric(label="Overall % Reduction", value=f"{percentage_reduction:.2f}%")

    st.write(f"Total Original Delay across all recorded stops: {total_original_delay:.0f} min")
    st.write(f"Total Optimized Delay across all recorded stops: {total_optimized_delay:.0f} min")
    st.write(f"**Total Delay Reduced:** {total_reduction:.0f} min")

    # --- Individual Train Visualization ---
    st.subheader("Individual Train Delay Comparison")
    st.write("Select a train to visualize the original and optimized delays per station.")

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
            fig.update_layout(title=f"Original vs Optimized Delay for Train {selected_train_value}",
                            xaxis_title="Station Name", yaxis_title="Delay (min)",
                            xaxis=dict(tickangle=-45), barmode="group",
                            legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select a train from the dropdown to see its detailed schedule.")

    # --- Interactive Train Scheduling Simulation ---
    st.subheader("Interactive Train Scheduling Simulation")
    st.write("Step through the scheduling process for a selected train to see how the DQN optimizes delays at each station. The visualization shows Q-values for each possible delay adjustment and the train's progress along its route.")

    if selected_train_value is not None:
        # Filter data for the selected train
        sim_filtered_df = df_optimized[df_optimized["Train No"].astype(str) == selected_train_value].sort_values(by="SEQ")
        sim_filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values(by="Station")

        if sim_filtered_df.empty or sim_filtered_q_df.empty:
            st.warning(f"No data available for simulation for Train Number: {selected_train_value}.")
        else:
            # Initialize session state for simulation step
            if 'sim_step' not in st.session_state:
                st.session_state.sim_step = 0
            if 'sim_train' not in st.session_state or st.session_state.sim_train != selected_train_value:
                st.session_state.sim_step = 0
                st.session_state.sim_train = selected_train_value

            total_stations = len(sim_filtered_df)
            current_step = st.session_state.sim_step

            # Navigation buttons
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
                # Get current station data
                current_row = sim_filtered_df.iloc[current_step]
                current_q_row = sim_filtered_q_df.iloc[current_step]
                station = current_row["Station Name"]
                original_delay = current_row["Original Delay"]
                optimized_delay = current_row["Optimized Delay"]
                q_values = current_q_row["Q-Values (0,-1,-2,-3,-4 adj)"]
                selected_action = current_q_row["Selected Action Index"]
                delay_adjustment = current_q_row["Delay Adjustment (mins)"]

                # Q-value bar chart
                q_fig = go.Figure()
                colors = ['blue' if i != selected_action else 'green' for i in range(len(delay_adjustments))]
                q_fig.add_trace(go.Bar(
                    x=[str(adj) for adj in delay_adjustments],
                    y=q_values,
                    marker_color=colors,
                    text=[f"{val:.2f}" for val in q_values],
                    textposition='auto'
                ))
                q_fig.update_layout(
                    title=f"Q-Values for Delay Adjustments at {station}",
                    xaxis_title="Delay Adjustment (minutes)",
                    yaxis_title="Q-Value",
                    showlegend=False
                )
                st.plotly_chart(q_fig, use_container_width=True)

                # Timeline visualization
                timeline_data = []
                base_date = datetime(2025, 5, 18, 0, 0)  # Base date for timeline
                for i, row in sim_filtered_df.iterrows():
                    status = "Future"
                    if i < current_step:
                        status = "Past"
                    elif i == current_step:
                        status = "Current"
                    # Use Scheduled Arrival Time (in minutes) to compute start time, fallback to SEQ
                    try:
                        arrival_minutes = float(row["Scheduled Arrival Time"]) if float(row["Scheduled Arrival Time"]) > 0 else float(row["SEQ"]) * 10
                    except (ValueError, TypeError):
                        st.warning(f"Non-numeric value found in 'Scheduled Arrival Time' ({row['Scheduled Arrival Time']}) or 'SEQ' ({row['SEQ']}) for station {row['Station Name']}. Using SEQ-based fallback.")
                        arrival_minutes = float(row["SEQ"]) * 10 if pd.notnull(row["SEQ"]) else 0
                    start_time = base_date + timedelta(minutes=arrival_minutes)
                    # Assume a short duration (e.g., 5 minutes) for each station stop
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
                    'Past': 'rgb(255, 165, 0)',      # Orange for past stations
                    'Current': 'rgb(0, 200, 0)',     # Bright green for current station
                    'Future': 'rgb(70, 130, 180)'    # Steel blue for future stations
                }
                timeline_fig = ff.create_gantt(
                    timeline_data,
                    colors=colors,
                    index_col="Resource",
                    show_colorbar=True,
                    group_tasks=True,
                    title="Train Route Timeline"
                )
                # Add hover text with delay details
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
                    yaxis=dict(showgrid=True, autorange="reversed")  # Gridlines and SEQ order
                )
                st.plotly_chart(timeline_fig, use_container_width=True)

                # Explanation
                st.write(f"**Station: {station} (Step {current_step + 1}/{total_stations})**")
                st.write(f"- **Original Delay**: {original_delay:.0f} minutes")
                st.write(f"- **DQN Decision**: Adjust delay by {delay_adjustment:.0f} minutes (Action {selected_action})")
                st.write(f"- **Optimized Delay**: {optimized_delay:.0f} minutes")
                st.write(f"The DQN evaluated the Q-values for each possible adjustment ({', '.join([str(adj) for adj in delay_adjustments])} minutes). The highest Q-value ({q_values[selected_action]:.2f}) corresponds to the adjustment of {delay_adjustment} minutes, indicating the best action to minimize delay while considering factors like congestion and track availability.")
            else:
                st.info("Simulation complete. Select a new train or restart to explore another route.")
    else:
        st.info("Please select a train to start the scheduling simulation.")

    # --- DQN Prediction Analysis ---
    st.subheader("DQN Prediction Analysis (Sample)")
    st.write("Below are the Q-values predicted by the DQN for the first entry in the dataset or for stations of the selected train.")

    if not q_df_analysis.empty:
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values(by="Station")
        if not filtered_q_df.empty:
            st.write(f"Showing analysis for Train {selected_train_value}:")
            st.dataframe(filtered_q_df.head(10).reset_index(drop=True))
        else:
            st.write(f"No detailed analysis data found for the selected Train {selected_train_value}.")
            st.write("Showing overall first 10 entries instead:")
            st.dataframe(q_df_analysis.head(10).reset_index(drop=True))
    else:
        st.info("Q-value analysis data is not available.")

    # --- Sample Q-Value Visualization ---
    if not q_df_analysis.empty:
        sample = q_df_analysis.iloc[0]
        fig_mpl, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar([str(adj) for adj in delay_adjustments], sample["Q-Values (0,-1,-2,-3,-4 adj)"], color='purple')
        for bar in bars:
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va=va, ha='center')
        ax.set_title(f"Sample Q-values for Delay Adjustments at {sample['Station']} (Train {sample['Train No']})")
        ax.set_xlabel("Delay Adjustment (minutes)")
        ax.set_ylabel("Q-value")
        ax.grid(axis='y', linestyle='--')
        st.subheader("Sample Q-Value Visualization (First Dataset Entry)")
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)