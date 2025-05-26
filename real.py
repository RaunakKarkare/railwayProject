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
import requests
import time
from datetime import datetime
import streamlit as st_version  # For version checking


# --- Helper Functions ---
def generate_dynamic_adjustments(original_delay, congestion_level, track_availability, peak_hour_indicator):
    """
    Generate scenario-based delay adjustments based on delay, congestion, track availability, and peak hour.
    Args:
        original_delay (float): Current delay in minutes (positive or zero).
        congestion_level (int): Encoded congestion level (0: low, 1: medium, 2: high).
        track_availability (int): Encoded track availability (0: low, 1: high).
        peak_hour_indicator (int): Peak hour status (0: non-peak, 1: peak).
    Returns:
        list: List of possible delay adjustments (non-positive).
    """
    base_adjustments = [0]
    if original_delay <= 5:
        adjustments = [-1, -2] if track_availability == 1 and peak_hour_indicator == 0 else [-1]
    elif original_delay <= 15:
        adjustments = [-2, -3, -4] if congestion_level <= 1 and peak_hour_indicator == 0 else [-2, -3]
    else:
        adjustments = [-3, -4, -5,
                       -6] if congestion_level == 0 and track_availability == 1 and peak_hour_indicator == 0 else [-3,
                                                                                                                   -4]

    return sorted(base_adjustments + adjustments, reverse=True)


def convert_time_to_minutes(time_str):
    """
    Convert time string (HH:MM) to minutes since midnight.
    Args:
        time_str (str): Time in HH:MM format or invalid (e.g., "--", "00:00").
    Returns:
        float: Minutes since midnight, or np.nan if invalid.
    """
    try:
        if pd.isna(time_str) or not str(time_str).strip() or time_str in ["--", "00:00"]:
            return np.nan
        time_obj = pd.to_datetime(str(time_str), errors='coerce', format="%H:%M").time()
        if time_obj is None:
            return np.nan
        return time_obj.hour * 60 + time_obj.minute
    except Exception:
        return np.nan


def fetch_train_data(api_key, train_numbers, date="20250521", debug=False):
    """
    Fetch real-time train data from Indian Railway IRCTC API on RapidAPI.
    Args:
        api_key (str): RapidAPI key.
        train_numbers (list): List of train numbers to fetch.
        date (str): Departure date in YYYYMMDD format (default: today).
        debug (bool): If True, print raw API responses and DataFrame for debugging.
    Returns:
        pd.DataFrame: DataFrame with train data, filtered for crossed stations and next upcoming station.
    """
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "indian-railway-irctc.p.rapidapi.com",
        "X-Rapid-API": "rapid-api-database"
    }
    all_data = []

    # Static mapping of platform counts for key stations
    platform_counts = {
        "SEY": 1,  # Seoni
        "CHUA": 1,  # Chourai
        "CWA": 2,  # Chhindwara Jn
        "PUX": 1,  # Parasia
        "JNO": 1,  # Junnor Deo
        "NVG": 1,  # Nawagaon
        "BXY": 1,  # Bordahi
        "AMLA": 3,  # Amla Jn
        "BZU": 2  # Betul
    }

    for train_no in train_numbers:
        try:
            url = "https://indian-railway-irctc.p.rapidapi.com/api/trains/v1/train/status"
            params = {
                "train_number": train_no,
                "departure_date": date,
                "isH5": "true",
                "client": "web"
            }
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            if debug:
                st.write(f"Raw API response for train {train_no}: {data}")

            if not data.get("body", {}).get("stations"):
                st.warning(
                    f"No station data returned for train {train_no}. Status: {data.get('body', {}).get('train_status_message', 'Unknown status')}")
                continue

            if "waiting for an update" in data.get("body", {}).get("train_status_message", "").lower():
                st.warning(
                    f"Live updates unavailable for train {train_no}: {data.get('body', {}).get('train_status_message')}. Try a train currently running.")
                continue

            current_station = data.get("body", {}).get("current_station", "")
            current_seq = 0
            for station in data.get("body", {}).get("stations", []):
                if station.get("stationCode") == current_station:
                    current_seq = int(station.get("stnSerialNumber", 0))
                    break

            for station in data.get("body", {}).get("stations", []):
                station_seq = int(station.get("stnSerialNumber", 0))
                # Include stations up to current station and the next one
                if station_seq > current_seq + 1:
                    continue

                delay = 0
                if station.get("actual_arrival_time") != "--" and station.get("arrivalTime") != "--" and station.get(
                        "actual_arrival_time") != "00:00":
                    try:
                        actual = pd.to_datetime(station["actual_arrival_time"], format="%H:%M")
                        scheduled = pd.to_datetime(station["arrivalTime"], format="%H:%M")
                        delay = (actual - scheduled).total_seconds() / 60
                    except ValueError:
                        delay = 0

                # Handle haltTime parsing
                halt_time = 0
                try:
                    halt_str = str(station.get("haltTime", "0"))
                    if halt_str != "--" and halt_str:
                        if ":" in halt_str:
                            minutes = int(halt_str.split(":")[0])
                        else:
                            minutes = int(halt_str)
                        halt_time = max(0, minutes)
                except (ValueError, TypeError):
                    halt_time = 0

                # Station Congestion: Based on haltTime
                congestion_level = 2 if halt_time > 10 else 1 if halt_time > 2 else 0

                # Peak Hour Indicator: Based on Scheduled Arrival Time
                arrival_minutes = convert_time_to_minutes(station.get("arrivalTime", "00:00"))
                peak_hours = (7 * 60 <= arrival_minutes <= 9 * 60) or (17 * 60 <= arrival_minutes <= 20 * 60)
                peak_hour_indicator = 1 if peak_hours else 0

                # Track Availability: Based on platform counts
                station_code = station.get("stationCode", "")
                track_availability = 1 if platform_counts.get(station_code, 1) > 2 else 0

                # Convert distance to numeric
                distance = 0
                try:
                    distance = int(station.get("distance", "0"))
                except (ValueError, TypeError):
                    distance = 0

                station_info = {
                    "Train No": train_no,
                    "Train Name": data.get("body", {}).get("train_name", "Unknown"),
                    "Station Name": station.get("stationName", "Unknown"),
                    "Station Code": station.get("stationCode", "Unknown"),
                    "Scheduled Arrival Time": station.get("arrivalTime", "00:00"),
                    "Scheduled Departure Time": station.get("departureTime", "00:00"),
                    "Actual Arrival Time": station.get("actual_arrival_time", "00:00"),
                    "Actual Departure Time": station.get("actual_departure_time", "00:00"),
                    "Original Delay": max(0, delay),
                    "SEQ": station_seq,
                    "Distance": distance,
                    "Halt Time (min)": halt_time,
                    "Delay Status": "On Time" if delay <= 0 else "Delayed",
                    "Station Congestion": congestion_level,
                    "Track Availability": track_availability,
                    "Peak Hour Indicator": peak_hour_indicator
                }
                all_data.append(station_info)
        except requests.HTTPError as e:
            if response.status_code == 404:
                st.warning(
                    f"404 Not Found for train {train_no}: Invalid train number or date. Use 5-digit numbers (e.g., 12051) and verify on IRCTC/NTES.")
            elif response.status_code == 429:
                st.error(f"Rate limit exceeded for train {train_no}. Check RapidAPI dashboard or try later.")
            else:
                st.warning(f"Failed to fetch data for train {train_no}: {e}")
        except requests.RequestException as e:
            st.warning(f"Network error for train {train_no}: {e}")

    df = pd.DataFrame(all_data)
    if df.empty:
        st.error(
            "No data fetched for any train. Check train numbers, date, or API status. Try trains currently running.")
    elif debug:
        st.write("Fetched DataFrame dtypes:", df.dtypes)
        st.write("Fetched DataFrame:", df.head())
        if df["Original Delay"].eq(0).all():
            st.warning(
                "All delays are zero. The train may not have live updates or is running on time. Try another train or wait for updates.")
    return df


# --- Data Loading, Preprocessing, RL Environment, Training, Prediction ---
@st.cache_resource(show_spinner=False)
def load_data_train_and_predict(_hyperparams, api_key, train_numbers, date, debug=False):
    """
    Loads real-time data, preprocesses, trains DQN model, and performs predictions.
    Args:
        _hyperparams (dict): Hyperparameters for DQN.
        api_key (str): RapidAPI key.
        train_numbers (list): List of train numbers.
        date (str): Departure date in YYYYMMDD format.
        debug (bool): If True, print debugging information.
    Returns:
        tuple: (df_result, q_df, best_params, train_groups, last_updated)
    """
    status_message = st.empty()
    status_message.text("Fetching and preprocessing real-time data...")

    # Fetch real-time data
    df = fetch_train_data(api_key, train_numbers, date, debug=debug)
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    if df.empty:
        status_message.warning("No data fetched from API.")
        return pd.DataFrame(), pd.DataFrame(), {}, {}, last_updated

    # Convert time columns to minutes
    time_columns = ["Scheduled Arrival Time", "Scheduled Departure Time", "Actual Arrival Time",
                    "Actual Departure Time"]
    for col in time_columns:
        df[col] = df[col].apply(convert_time_to_minutes)

    # Impute missing time values with median
    for col in time_columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)

    # Handle categorical columns
    categorical_cols = ["Delay Status", "Station Congestion", "Track Availability", "Peak Hour Indicator"]
    for col in categorical_cols:
        if col == "Delay Status":
            # Delay Status is string ("On Time", "Delayed"); encode directly
            df[col] = df[col].fillna("On Time").astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            # Other categoricals are already integers
            df[col] = df[col].fillna(0).astype(int)
            df[col] = LabelEncoder().fit_transform(df[col])

    # Log dtypes and sample values for debugging
    if debug:
        st.write("DataFrame dtypes after categorical encoding:", df.dtypes)
        st.write("Sample categorical values:", df[categorical_cols].head())

    # Features for RL
    FEATURES = [
        "Scheduled Arrival Time", "Scheduled Departure Time", "Distance",
        "Delay Status", "Station Congestion", "Track Availability",
        "Peak Hour Indicator", "Halt Time (min)", "Original Delay"
    ]

    # Validate FEATURES exist in DataFrame
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        status_message.error(f"Missing features in DataFrame: {missing_features}")
        st.stop()

    # Impute missing numerical values with median
    for feature in FEATURES:
        if feature not in categorical_cols:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            if df[feature].isnull().any():
                median_value = df[feature].median()
                if pd.isna(median_value):
                    median_value = 0
                df[feature] = df[feature].fillna(median_value)

    # Ensure numeric types and non-negative delays
    for feature in FEATURES:
        if feature not in categorical_cols:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            if feature == "Original Delay":
                df[feature] = df[feature].clip(lower=0)

    X = df[FEATURES].values.astype(np.float32)
    y = df["Original Delay"].values

    # Group data by Train No for multi-step episodes
    train_groups = {train_no: group.sort_values("SEQ") for train_no, group in df.groupby("Train No")}

    # Custom Gym Environment
    class TrainSchedulingEnv(gym.Env):
        def __init__(self, train_groups, features):
            super(TrainSchedulingEnv, self).__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(5)
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
            group = self.train_groups[self.current_train]
            if not all(f in group.columns for f in self.features):
                raise ValueError(f"Features {self.features} not found in group columns: {group.columns}")
            self.state = group[self.features].iloc[self.current_idx].values.astype(np.float32)
            if len(self.state) != len(self.features):
                raise ValueError(f"State shape {len(self.state)} does not match features {len(self.features)}")
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            group = self.train_groups[self.current_train]
            row = group.iloc[self.current_idx]
            original_delay_idx = self.features.index("Original Delay")
            congestion_idx = self.features.index("Station Congestion")
            track_idx = self.features.index("Track Availability")
            peak_hour_idx = self.features.index("Peak Hour Indicator")

            original_delay = row["Original Delay"]
            adjustments = generate_dynamic_adjustments(
                original_delay,
                row["Station Congestion"],
                row["Track Availability"],
                row["Peak Hour Indicator"]
            )

            action = min(action, len(adjustments) - 1)
            adjustment = adjustments[action]
            if adjustment > 0:
                adjustment = 0
            new_delay = max(0, min(original_delay + adjustment, original_delay))

            # Reward function
            delay_reduction = original_delay - new_delay
            congestion_penalty = -0.5 * row["Station Congestion"] * abs(adjustment)
            track_bonus = 0.2 * row["Track Availability"] * delay_reduction
            peak_hour_penalty = -0.3 * row["Peak Hour Indicator"] * abs(adjustment)
            downstream_impact = -0.1 * new_delay if self.current_idx < len(group) - 1 else 0
            increase_penalty = -100.0 if new_delay > original_delay else 0.0
            reward = delay_reduction + congestion_penalty + track_bonus + peak_hour_penalty + downstream_impact + increase_penalty

            self.state = row[self.features].values.astype(np.float32)
            self.current_idx += 1

            done = self.current_idx >= len(group) or self.current_idx >= self.max_steps
            if not done:
                if self.current_idx < len(group):
                    self.state = group[self.features].iloc[self.current_idx].values.astype(np.float32)
                else:
                    done = True

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
                original_delay = group.iloc[i]["Original Delay"]
                adjustments = generate_dynamic_adjustments(
                    original_delay,
                    group.iloc[i]["Station Congestion"],
                    group.iloc[i]["Track Availability"],
                    group.iloc[i]["Peak Hour Indicator"]
                )
                state_tensor = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32).to(best_model.device)
                q_values = best_model.policy.q_net(state_tensor).cpu().numpy()[0]

                valid_q_values = q_values[:len(adjustments)]
                action_index = np.argmax(valid_q_values)
                delay_adjustment = adjustments[action_index]
                if delay_adjustment > 0:
                    delay_adjustment = 0
                optimized_delay = max(0, min(original_delay + delay_adjustment, original_delay))

                delay_reduction = original_delay - optimized_delay
                congestion_penalty = -0.5 * group.iloc[i]["Station Congestion"] * abs(delay_adjustment)
                track_bonus = 0.2 * group.iloc[i]["Track Availability"] * delay_reduction
                peak_hour_penalty = -0.3 * group.iloc[i]["Peak Hour Indicator"] * abs(delay_adjustment)
                downstream_impact = -0.1 * optimized_delay if i < len(group) - 1 else 0
                increase_penalty = -100.0 if optimized_delay > original_delay else 0.0
                reward = delay_reduction + congestion_penalty + track_bonus + peak_hour_penalty + downstream_impact + increase_penalty

                if optimized_delay > original_delay:
                    print(f"Warning: Train {train_no}, Station {group.iloc[i]['Station Name']}: "
                          f"Optimized Delay ({optimized_delay}) > Original Delay ({original_delay})")

                optimized_delays.append(optimized_delay)
                q_value_logs.append({
                    "Train No": train_no,
                    "Station": group.iloc[i]["Station Name"],
                    "Station Code": group.iloc[i]["Station Code"],
                    "Original Delay": original_delay,
                    "Q-Values": valid_q_values.tolist(),
                    "Selected Action Index": action_index,
                    "Delay Adjustment (mins)": delay_adjustment,
                    "Optimized Delay": optimized_delay,
                    "Adjustments Available": adjustments,
                    "Reward": reward,
                    "SEQ": group.iloc[i]["SEQ"]
                })

    df_result = df.copy()
    df_result["Optimized Delay"] = optimized_delays
    q_df = pd.DataFrame(q_value_logs)

    violations = df_result[df_result["Optimized Delay"] > df_result["Original Delay"]]
    if not violations.empty:
        print(f"Warning: Found {len(violations)} cases where Optimized Delay > Original Delay")
        print(violations[["Train No", "Station Name", "Original Delay", "Optimized Delay"]])

    status_message.text("Processing complete.")
    status_message.empty()

    return df_result, q_df, best_params, train_groups, last_updated


# --- Streamlit App ---
st.title("Real-Time Train Delay Optimization using DQN")
st.write("Optimizes train delays using real-time data from Indian Railway IRCTC API on RapidAPI.")
st.write("Shows delay comparison and analysis for stations crossed and the next upcoming station.")

# Log Streamlit version for debugging
st.write(f"Streamlit version: {st_version.__version__}")

# API Key and Input
api_key = st.text_input("Enter RapidAPI Key", type="password")
train_numbers = st.text_input("Enter Train Numbers (comma-separated, 5-digit, e.g., 12051,12309,20423)",
                              value="20423").split(",")
train_numbers = [num.strip() for num in train_numbers if num.strip()]
date = st.text_input("Enter Departure Date (YYYYMMDD, e.g., 20250521)", value="20250521")

if not api_key or not train_numbers or not date:
    st.error("Please provide a valid RapidAPI key, at least one 5-digit train number, and a departure date.")
    st.stop()

# Hyperparameters
hyperparams = {"learning_rate": 1e-3, "exploration_fraction": 0.5}
# Enable debug logging
df_optimized, q_df_analysis, best_params, train_groups, last_updated = load_data_train_and_predict(hyperparams, api_key,
                                                                                                   train_numbers, date,
                                                                                                   debug=True)

if df_optimized.empty:
    st.error("Could not load or process data. Check train numbers, date, or API status. Try trains currently running.")
else:
    st.write(f"**Last Data Update:** {last_updated}")

    # Metrics
    st.subheader("Overall Delay Optimization Metrics (Crossed and Next Station)")
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
    st.subheader("Individual Train Delay Comparison (Crossed and Next Station)")
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

        # Filter for crossed stations and next upcoming station
        filtered_df = df_optimized[df_optimized["Train No"].astype(str) == selected_train_value].sort_values("SEQ")
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values("SEQ")

        if not filtered_df.empty and not filtered_q_df.empty:
            merged_df = pd.merge(
                filtered_df[["Station Name", "Station Code", "Original Delay", "Optimized Delay", "Station Congestion",
                             "Track Availability", "Peak Hour Indicator", "Delay Status", "SEQ"]],
                filtered_q_df[["Station", "Station Code", "Delay Adjustment (mins)", "Reward", "SEQ"]],
                left_on=["Station Name", "Station Code", "SEQ"],
                right_on=["Station", "Station Code", "SEQ"],
                how="inner"
            ).sort_values("SEQ")

            if merged_df.empty:
                st.warning(f"No aligned data for Train {selected_train_value}.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=merged_df["Station Name"],
                    y=merged_df["Original Delay"],
                    name="Original Delay",
                    marker=dict(color="red"),
                    customdata=merged_df[["Station Congestion", "Track Availability", "Peak Hour Indicator",
                                          "Delay Status", "Delay Adjustment (mins)", "Reward"]],
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
                    x=merged_df["Station Name"],
                    y=merged_df["Optimized Delay"],
                    name="Optimized Delay",
                    marker=dict(color="green"),
                    customdata=merged_df[["Station Congestion", "Track Availability", "Peak Hour Indicator",
                                          "Delay Status", "Delay Adjustment (mins)", "Reward"]],
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
                    title=f"Delay Comparison for Train {selected_train_value} (Crossed and Next Station)",
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
    st.subheader("Station-Level Analysis for Selected Train (Crossed and Next Station)")
    if not q_df_analysis.empty and dropdown_options:
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values("SEQ")
        if not filtered_q_df.empty:
            st.write(f"Detailed analysis for Train {selected_train_value}:")
            display_df = filtered_q_df[[
                "Station", "Station Code", "Original Delay", "Optimized Delay",
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
        filtered_q_df = q_df_analysis[q_df_analysis["Train No"].astype(str) == selected_train_value].sort_values("SEQ")
        if not filtered_q_df.empty:
            st.write(f"Q-value analysis for Train {selected_train_value}:")
            st.dataframe(filtered_q_df.head(10).reset_index(drop=True))
        else:
            st.write("Showing first 10 entries:")
            st.dataframe(q_df_analysis.head(10).reset_index(drop=True))

    # Q-Value Plot
    if not q_df_analysis.empty:
        sample = q_df_analysis.iloc[0]
        if "Q-Values" not in sample or "Adjustments Available" not in sample:
            st.warning(f"Sample data missing 'Q-Values' or 'Adjustments Available' for Train {selected_train_value}.")
        else:
            fig_mpl, ax = plt.subplots(figsize=(10, 5))
            adjustments = sample["Adjustments Available"]
            q_values = sample["Q-Values"]
            if not isinstance(q_values, list) or not q_values:
                st.warning(f"Invalid Q-Values data for Train {selected_train_value}: {q_values}")
            else:
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

    # Auto-refresh
    st.write("Data will refresh every 5 minutes...")
    time.sleep(300)
    try:
        st.rerun()
    except AttributeError:
        st.warning(
            "Auto-refresh not supported in this Streamlit version. Please upgrade to Streamlit 1.29.0 or higher to use st.rerun.")