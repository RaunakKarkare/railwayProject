import gym
import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
import streamlit as st

from stable_baselines3 import DQN
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch # Import torch for device handling with cached model

# Define delay_adjustments in the global scope so it's accessible everywhere
delay_adjustments = [0, -1, -2, -3, -4]

# --- Data Loading, Preprocessing, RL Environment, Training, Prediction (Cached) ---

# @st.cache_resource caches the function's output (model and dataframes)
# It uses the function arguments and the function's code to determine if a re-run is needed.
@st.cache_resource(show_spinner=False) # Hide default spinner, use custom one inside
def load_data_train_and_predict(adj_list):
    """
    Loads data, preprocesses, trains DQN model, and performs predictions.
    This function is cached by Streamlit.

    Args:
        adj_list (list): The list of possible delay adjustments (used for action space).

    Returns:
        tuple: A tuple containing:
            - df_result (pd.DataFrame): The original DataFrame with 'Optimized Delay' column added.
            - q_df (pd.DataFrame): DataFrame containing Q-value analysis for predictions.
    """
    # Use a custom status message for this potentially long step
    status_message = st.empty()
    status_message.text("Loading and preprocessing data...")

    # Load Dataset
    try:
        df = pd.read_csv("Corrected_Time_Table.csv", low_memory=False)
        if df.empty:
            status_message.warning("Dataset is empty.")
            return pd.DataFrame(), pd.DataFrame() # Return empty if data is empty
    except FileNotFoundError:
        status_message.error("Error: Corrected_Time_Table.csv not found.")
        st.stop() # Stop Streamlit execution
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
                # Try parsing with explicit format first
                time_obj = pd.to_datetime(time_str, format="%H:%M").time()
            except ValueError:
                 try:
                    # Fallback to a more flexible parse if needed
                    time_obj = pd.to_datetime(time_str).time()
                 except ValueError:
                    # Handle cases that still fail
                    return 0
            return time_obj.hour * 60 + time_obj.minute
        except Exception as e:
            # print(f"Debug: Error converting time '{time_str}': {e}") # Uncomment for detailed debug
            return 0

    time_columns = ["Scheduled Arrival Time", "Scheduled Departure Time", "Actual Arrival Time", "Actual Departure Time"]
    for col in time_columns:
        df[col] = df[col].apply(convert_time_to_minutes)

    # Handle potential missing values and ensure string type before encoding
    for col in ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]:
        if df[col].isnull().any():
            print(f"Warning: Missing values found in '{col}'. Filling with 'Unknown'.") # Console print
            df[col] = df[col].fillna("Unknown")
        df[col] = df[col].astype(str) # Ensure string type

    # Encode categorical features
    categorical_cols = ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]
    for col in categorical_cols:
        # LabelEncoder works on string or numeric, .fit_transform expects 1D array
        df[col] = LabelEncoder().fit_transform(df[col])

    # Features for RL
    FEATURES = [
        "Scheduled Arrival Time", "Scheduled Departure Time", "Distance",
        "Delay Status", "Station Congestion", "Track Availability",
        "Peak Hour Indicator", "Halt Time (min)", "Original Delay"
    ]

    # Ensure all FEATURES columns exist and handle NaNs in numerical features
    for feature in FEATURES:
        if feature not in df.columns:
            status_message.error(f"Error: Feature '{feature}' not found in the dataset.")
            st.stop()
        # Fill potential NaNs in numerical columns (those not in categorical_cols)
        if feature not in categorical_cols and df[feature].isnull().any():
             print(f"Warning: Missing values found in numerical feature '{feature}'. Filling with 0.") # Console print
             df[feature] = df[feature].fillna(0)

    # Ensure feature columns are numeric types for model input
    for feature in FEATURES:
        if feature not in categorical_cols: # Categorical already handled by LabelEncoder
             df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0) # Coerce to numeric, fill errors/NaNs

    X = df[FEATURES].values.astype(np.float32) # Ensure float32 for model input
    y = df["Original Delay"].values # Original delay can be any numeric type

    if len(X) == 0:
         status_message.warning("No valid data rows found after preprocessing.")
         return pd.DataFrame(), pd.DataFrame()


    # Custom Gym Environment
    class TrainSchedulingEnv(gym.Env):
        def __init__(self, adj_list):
            super(TrainSchedulingEnv, self).__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(FEATURES),), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(adj_list)) # Action space size based on adj_list length
            self.state = None
            self.current_idx = 0
            self.adj_list = adj_list # Store adj_list from constructor

        def reset(self):
            # Pick a random row index from the dataset for a new "episode"
            self.current_idx = random.randint(0, len(X) - 1)
            self.state = X[self.current_idx].copy() # Use .copy() to avoid modifying X directly
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            # Ensure action index is valid
            if not self.action_space.contains(action):
                 print(f"Warning: Invalid action received: {action}. Valid actions are 0 to {len(self.adj_list)-1}. Defaulting to 0.")
                 action = 0

            adjustment = self.adj_list[action]

            # Find the index of 'Original Delay' dynamically in FEATURES list
            try:
                original_delay_idx = FEATURES.index("Original Delay")
                original_delay = self.state[original_delay_idx]
            except ValueError:
                 print("Error: 'Original Delay' not found in FEATURES list during step.")
                 original_delay = 0 # Default if not found

            new_delay = max(0, original_delay + adjustment)

            # Reward: Encourage reducing delay (simple approach)
            # Reward = Amount of delay reduced in this step
            reward = original_delay - new_delay

            # Update the delay in the state
            if original_delay_idx is not None:
                self.state[original_delay_idx] = new_delay

            # In this environment, each "episode" is a single decision at one station
            done = True
            info = {}

            return np.array(self.state, dtype=np.float32), reward, done, info

    status_message.text("Training DQN model...")
    # Pass the adj_list to the environment constructor
    env = TrainSchedulingEnv(adj_list)

    # Initialize the DQN model
    model = DQN("MlpPolicy", env, verbose=0, exploration_fraction=0.5, exploration_final_eps=0.05)

    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=20000)

    status_message.text("DQN model training complete. Analyzing predictions...")

    # --- Prediction Phase ---
    # Use the trained model to predict optimized delays for the entire dataset

    optimized_delays = []
    q_value_logs = []

    # Use torch.no_grad() for inference to save memory and speed up
    with torch.no_grad():
        for i in range(len(df)):
            # Get the state for the current row
            state = np.array(X[i], dtype=np.float32)

            # Access the Q-network from the trained model's policy
            # SB3 stores the Q-network within the policy object
            q_net = model.policy.q_net

            # Prepare the state for the PyTorch model (add batch dimension, move to device)
            state_tensor = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32).to(model.device)

            # Get the predicted Q-values for all actions in this state
            q_values_tensor = q_net(state_tensor)

            # Convert Q-values back to numpy array, remove batch dimension, move to CPU
            q_values = q_values_tensor.cpu().numpy()[0]

            # Select the action with the highest predicted Q-value (greedy action)
            action_index = np.argmax(q_values)

            # Map the selected action index back to a delay adjustment value
            delay_adjustment = adj_list[action_index]

            # Get the original delay for this row
            original_delay = y[i]

            # Calculate the optimized delay (ensure it's not negative)
            optimized_delay = max(0, original_delay + delay_adjustment)
            optimized_delays.append(optimized_delay)

            # Log details for analysis DataFrame
            q_value_logs.append({
                "Train No": df.iloc[i]["Train No"],
                "Station": df.iloc[i]["Station Name"],
                "Original Delay": original_delay,
                # Store Q-values as a list in the DataFrame
                "Q-Values (0,-1,-2,-3,-4 adj)": q_values.tolist(),
                "Selected Action Index": action_index,
                "Delay Adjustment (mins)": delay_adjustment,
                "Optimized Delay": optimized_delay,
                # Optional: include state features in logs (can make DF large)
                # "Features": state.tolist()
            })

    # Add the calculated optimized delays back to a copy of the original dataframe
    df_result = df.copy()
    df_result["Optimized Delay"] = optimized_delays

    # Create a DataFrame from the Q-value logs
    q_df = pd.DataFrame(q_value_logs)

    status_message.text("Prediction analysis complete.")
    status_message.empty() # Clear the status message once done

    # Return the results to be cached
    return df_result, q_df

# --- End of Cached Function ---


# --- Streamlit App Layout ---

st.title("Train Delay Optimization using DQN")

st.write("This application uses a Deep Q-Network (DQN) to predict optimized train delays.")

# Call the cached function to load data, train the model, and get predictions
# This will run only once unless the function code or its arguments change.
df_optimized, q_df_analysis = load_data_train_and_predict(delay_adjustments)


# Check if data was loaded successfully
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

    # Calculate percentage reduction safely
    percentage_reduction = 0
    if total_original_delay > 0:
        percentage_reduction = (total_reduction / total_original_delay) * 100

    # Use Streamlit columns for a cleaner layout of metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Avg Original Delay (min)", value=f"{avg_original_delay:.2f}")
    with col2:
        # Use delta to show the change (reduction)
        st.metric(label="Avg Optimized Delay (min)", value=f"{avg_optimized_delay:.2f}", delta=f"{-avg_reduction_per_stop:.2f} min")
    with col3:
        st.metric(label="Overall % Reduction", value=f"{percentage_reduction:.2f}%")

    # Add some more details about total reduction
    st.write(f"Total Original Delay across all recorded stops: {total_original_delay:.0f} min")
    st.write(f"Total Optimized Delay across all recorded stops: {total_optimized_delay:.0f} min")
    st.write(f"**Total Delay Reduced:** {total_reduction:.0f} min")


    # --- Individual Train Visualization ---
    st.subheader("Individual Train Delay Comparison")
    st.write("Select a train from the dropdown below to visualize the original and optimized delays per station.")

    # Generate dropdown options from the processed dataframe
    unique_trains = df_optimized.drop_duplicates(subset=["Train No", "Train Name"])
    # Ensure the values are strings for consistent comparison with selectbox output if needed
    unique_trains["Train No"] = unique_trains["Train No"].astype(str)

    dropdown_options = unique_trains.apply(
        lambda row: {"label": f"{row['Train No']} - {row['Train Name']}", "value": row["Train No"]},
        axis=1
    ).tolist()

    # Handle case where dropdown options might be empty
    if not dropdown_options:
         st.warning("No trains found in the data with unique Train No and Name.")
         selected_train_value = None
    else:
        # Determine a default selected value (first train number) and its index
        # Ensure default value is also treated as string for lookup
        default_train_value_str = str(df_optimized["Train No"].iloc[0]) if not df_optimized.empty else None
        default_index = 0
        if default_train_value_str is not None:
            # Find the index based on the string value in the options list's 'value'
            try:
                default_index = next((i for i, opt in enumerate(dropdown_options) if opt['value'] == default_train_value_str), 0)
            except Exception:
                 default_index = 0 # Fallback


        # Create the Streamlit selectbox
        selected_option = st.selectbox(
            "Select Train:",
            options=dropdown_options,
            format_func=lambda option: option['label'], # Display the 'label' part of the dict
            index=default_index
        )
        selected_train_value = selected_option['value'] if selected_option else None # Get the selected 'value'

    # Data filtering and Plotly figure creation based on selection
    if selected_train_value is not None:
        # Filter the main optimized dataframe based on the selected train value (which is a string)
        # Ensure the column in df_optimized is also treated as string for direct comparison
        filtered_df = df_optimized[df_optimized["Train No"].astype(str) == selected_train_value].sort_values(by="SEQ")

        if filtered_df.empty:
            st.warning(f"No station data found for Train Number: {selected_train_value}.")
        else:
            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Bar(x=filtered_df["Station Name"], y=filtered_df["Original Delay"],
                                 name="Original Delay", marker=dict(color="red")))
            fig.add_trace(go.Bar(x=filtered_df["Station Name"], y=filtered_df["Optimized Delay"],
                                 name="Optimized Delay", marker=dict(color="green")))

            fig.update_layout(title=f"Original vs Optimized Delay for Train {selected_train_value}",
                              xaxis_title="Station Name", yaxis_title="Delay (min)",
                              xaxis=dict(tickangle=-45), # Rotate station names if needed
                              barmode="group", # Group bars for comparison
                              legend=dict(x=0.01, y=0.99)) # Position legend

            # Display the figure in Streamlit
            st.plotly_chart(fig, use_container_width=True) # use_container_width makes it responsive

    else:
        # This block will show if dropdown_options is empty or no selection is made initially
        st.info("Please select a train from the dropdown to see its detailed schedule.")

    # --- Optional: Display Q-value Analysis ---
    st.subheader("DQN Prediction Analysis (Sample)")
    st.write("Below are the Q-values predicted by the DQN for the first entry in the dataset or for stations of the selected train.")

    if not q_df_analysis.empty:
        # Filter q_df_analysis for the selected train value (which is a string)
        # Ensure the column in q_df_analysis is also treated as string
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values(by="Station")

        if not filtered_q_df.empty:
            st.write(f"Showing analysis for Train {selected_train_value}:")
            # Display first 10 rows for the filtered train
            st.dataframe(filtered_q_df.head(10).reset_index(drop=True)) # Reset index for cleaner display
        else:
             # This case should ideally not happen if filtered_df is not empty,
             # but as a fallback if filtering on q_df_analysis fails for some reason.
             st.write(f"No detailed analysis data found for the selected Train {selected_train_value}.")
             st.write("Showing overall first 10 entries instead:")
             st.dataframe(q_df_analysis.head(10).reset_index(drop=True))
    else:
        st.info("Q-value analysis data is not available.")


    # --- Matplotlib plot for the first entry (sample Q-values) ---
    # This plot visualizes the Q-values for the very first station/row in the dataset
    # as processed by the cached function, giving insight into a sample prediction.
    if not q_df_analysis.empty:
        sample = q_df_analysis.iloc[0] # Get the first row from the analysis dataframe
        fig_mpl, ax = plt.subplots(figsize=(10, 5)) # Create a matplotlib figure and axes

        # Use the global delay_adjustments list for bar labels
        bars = ax.bar([str(adj) for adj in delay_adjustments], sample["Q-Values (0,-1,-2,-3,-4 adj)"], color='purple')

        # Add Q-value text labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            # Position text slightly above or below the bar depending on value
            va = 'bottom' if yval >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va=va, ha='center')

        ax.set_title(f"Sample Q-values for Delay Adjustments at {sample['Station']} (Train {sample['Train No']})")
        ax.set_xlabel("Delay Adjustment (minutes)")
        ax.set_ylabel("Q-value")
        ax.grid(axis='y', linestyle='--')

        # Display the matplotlib figure in Streamlit
        st.subheader("Sample Q-Value Visualization (First Dataset Entry)")
        st.pyplot(fig_mpl)
        plt.close(fig_mpl) # Close the figure to free memory

# --- End of Streamlit App ---