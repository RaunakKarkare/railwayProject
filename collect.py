
import pandas as pd
import requests
from datetime import datetime


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


def fetch_train_data(api_key, train_numbers, date, debug=False):
    """
    Fetch real-time train data from Indian Railway IRCTC API on RapidAPI.
    Args:
        api_key (str): RapidAPI key.
        train_numbers (list): List of train numbers to fetch.
        date (str): Departure date in YYYYMMDD format.
        debug (bool): If True, print debugging information.
    Returns:
        pd.DataFrame: DataFrame with train data.
    """
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "indian-railway-irctc.p.rapidapi.com",
        "X-Rapid-API": "rapid-api-database"
    }
    all_data = []

    platform_counts = {
        "SEY": 1, "CHUA": 1, "CWA": 2, "PUX": 1, "JNO": 1,
        "NVG": 1, "BXY": 1, "AMLA": 3, "BZU": 2
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
                print(f"Raw API response for train {train_no}: {data}")

            if not data.get("body", {}).get("stations"):
                print(
                    f"Warning: No station data returned for train {train_no}. Status: {data.get('body', {}).get('train_status_message', 'Unknown status')}")
                if debug:
                    print(f"API response details: Status Code: {response.status_code}, Response: {data}")
                continue

            if "waiting for an update" in data.get("body", {}).get("train_status_message", "").lower():
                print(
                    f"Warning: Live updates unavailable for train {train_no}: {data.get('body', {}).get('train_status_message')}")
                if debug:
                    print(f"API response details: Status Code: {response.status_code}, Response: {data}")
                continue

            current_station = data.get("body", {}).get("current_station", "")
            current_seq = 0
            for station in data.get("body", {}).get("stations", []):
                if station.get("stationCode") == current_station:
                    current_seq = int(station.get("stnSerialNumber", 0))
                    break

            for station in data.get("body", {}).get("stations", []):
                station_seq = int(station.get("stnSerialNumber", 0))
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

                congestion_level = 2 if halt_time > 10 else 1 if halt_time > 2 else 0
                arrival_minutes = convert_time_to_minutes(station.get("arrivalTime", "00:00"))
                peak_hours = (7 * 60 <= arrival_minutes <= 9 * 60) or (17 * 60 <= arrival_minutes <= 20 * 60)
                peak_hour_indicator = 1 if peak_hours else 0
                station_code = station.get("stationCode", "")
                track_availability = 1 if platform_counts.get(station_code, 1) > 2 else 0

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
            print(f"Error: Failed to fetch data for train {train_no}: HTTP Error {e}")
            if debug:
                print(f"HTTP Error details: Status Code: {response.status_code}, Response: {response.text}")
        except requests.RequestException as e:
            print(f"Error: Network error for train {train_no}: {e}")
            if debug:
                print(f"Network error details: {e}")

    df = pd.DataFrame(all_data)
    if df.empty:
        print("Error: No data fetched. Check train number, date, API key, or API status. Try trains like 12051, 12309.")
        if debug:
            print(
                "Debug: No data fetched. Ensure train number is a valid 5-digit number and has live updates on IRCTC/NTES.")
    elif debug:
        print("Fetched DataFrame dtypes:", df.dtypes)
        print("Fetched DataFrame:", df.head())
        if df["Original Delay"].eq(0).all():
            print("Warning: All delays are zero. The train may not have live updates or is running on time.")
    return df


if __name__ == "__main__":
    # Prompt for inputs
    print("Enter your RapidAPI key:")
    api_key = input().strip()
    if not api_key:
        print("Error: RapidAPI key is required.")
        exit(1)

    print("Enter a 5-digit train number (e.g., 16094):")
    train_number = input().strip()
    if not (train_number.isdigit() and len(train_number) == 5):
        print("Error: Train number must be a 5-digit number (e.g., 16094).")
        exit(1)
    train_numbers = [train_number]

    print("Enter departure date (YYYYMMDD, e.g., 20250523):")
    date = input().strip()
    if not (date.isdigit() and len(date) == 8):
        print("Error: Date must be in YYYYMMDD format (e.g., 20250523).")
        exit(1)
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        print("Error: Invalid date format. Use YYYYMMDD (e.g., 20250523).")
        exit(1)

    # Fetch data
    debug = True
    df = fetch_train_data(api_key, train_numbers, date, debug=debug)

    # Save to CSV
    if not df.empty:
        file_name = f"train_data_{train_number}_{date}.csv"
        df.to_csv(file_name, index=False)
        print(f"Data saved to {file_name}")
    else:
        print("No data to save.")