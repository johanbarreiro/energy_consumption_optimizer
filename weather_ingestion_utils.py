import requests
import json
from datetime import datetime
import sys
import pandas as pd
import numpy as np


def fetch_weather_data_from_api(location, start_date, end_date, api_key):
    """
    Fetch weather data for a specific location and date range.

    Parameters:
    location (str): Location for which to fetch the weather data.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    end_date (str): End date in the format 'YYYY-MM-DD'.
    api_key (str): Your API key for Visual Crossing Weather.

    Returns:
    dict: Parsed JSON data containing the weather information.
    """
    # Ensure date range is not more than a year
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    if (end - start).days > 365:
        print("Error: Date range cannot be more than a year apart.")
        return None

    # Construct the API URL
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    elements = "datetime,datetimeEpoch,temp,tempmax,tempmin,precip,windspeed,windgust,feelslike,feelslikemax,feelslikemin,pressure,stations,degreedays,accdegreedays"
    include = "fcst,obs,histfcst,stats"
    url = f"{base_url}/{location}/{start_date}/{end_date}?elements={elements}&include={include}&key={api_key}&contentType=json"

    try:
        # Fetch the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        jsonData = response.json()
        return jsonData

    except requests.exceptions.HTTPError as e:
        print('HTTP error:', e.response.text)
        sys.exit()
    except requests.exceptions.RequestException as e:
        print('Request error:', e)
        sys.exit()


def fetch_weather_data_from_csv(file_path, amount_of_columns='essential'):
    """
    Fetch weather data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing weather data.
    amount_of_columns (str): Specifies the amount of columns to fetch ('minimal', 'essential', 'all').
                             'minimal' will return a minimal set of columns,
                             'essential' will return a predefined set of columns,
                             'all' will return all columns.

    Returns:
    pandas.DataFrame: DataFrame containing the weather data, or None if an error occurs.
    """

    minimal_columns = [
        'datetime', 'temp', 'tempmax', 'tempmin', 'humidity'
    ]

    essential_columns = minimal_columns + [
        'precip', 'windspeed', 'uvindex', 'conditions'
    ]

    try:
        df = pd.read_csv(file_path)

        if amount_of_columns == 'minimal':
            # Check if minimal columns are present in the DataFrame
            missing_columns = [
                col for col in minimal_columns if col not in df.columns]
            if missing_columns:
                print(
                    f"Error: Missing minimal columns {missing_columns} in file '{file_path}'.")
                return None
            df = df[minimal_columns]

        elif amount_of_columns == 'essential':
            # Check if essential columns are present in the DataFrame
            missing_columns = [
                col for col in essential_columns if col not in df.columns]
            if missing_columns:
                print(
                    f"Error: Missing essential columns {missing_columns} in file '{file_path}'.")
                return None
            df = df[essential_columns]

        elif amount_of_columns == 'all':
            # Return all columns
            pass

        else:
            print(
                f"Error: Invalid amount_of_columns option '{amount_of_columns}'. Must be 'minimal', 'essential', or 'all'.")
            return None

        # Ensure datetime column is in datetime format
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        # Convert humidity from percentage to decimal
        if 'humidity' in df.columns:
            df['humidity'] = df['humidity'] / 100

        return df

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_path}'.")
        return None
    except pd.errors.DtypeWarning:
        print(f"Warning: Data type mismatch in file '{file_path}'.")
        return None


def approximate_hourly_weather(daily_df):
    """
    Approximate hourly weather conditions from daily data.

    Parameters:
    daily_df (pd.DataFrame): DataFrame containing daily weather data.

    Returns:
    pd.DataFrame: DataFrame with approximated hourly weather conditions.
    """
    # Ensure datetime column is in datetime format
    if 'datetime' in daily_df.columns:
        daily_df['datetime'] = pd.to_datetime(
            daily_df['datetime'], errors='coerce')
    else:
        raise ValueError(
            "The 'datetime' column is missing from the DataFrame.")

    # Set datetime as index
    daily_df.set_index('datetime', inplace=True)

    # Create a new DataFrame with hourly frequency
    hourly_index = pd.date_range(start=daily_df.index.min(
    ), end=daily_df.index.max() + pd.Timedelta(days=1), freq='H')[:-1]
    hourly_df = pd.DataFrame(index=hourly_index)

    # Merge the daily data with the hourly DataFrame
    merged_df = hourly_df.merge(
        daily_df, left_index=True, right_index=True, how='left')

    # Forward fill categorical columns and interpolate numeric columns
    for column in merged_df.columns:
        if column in ['tempmax', 'tempmin']:
            continue
        if np.issubdtype(merged_df[column].dtype, np.number):
            merged_df[column] = merged_df[column].interpolate(method='linear')
        else:
            merged_df[column] = merged_df[column].ffill()

    # Approximate hourly temperature using a sinusoidal function
    def calc_hourly_temp(row, hour):
        temp_range = row['tempmax'] - row['tempmin']
        return row['tempmin'] + (temp_range / 2) * (1 + np.cos((hour - 14) * np.pi / 12))

    hourly_temps = []
    for dt in hourly_df.index:
        day = dt.normalize()
        if day in daily_df.index:
            temp = calc_hourly_temp(daily_df.loc[day], dt.hour)
        else:
            temp = np.nan
        hourly_temps.append(temp)

    merged_df['temp'] = hourly_temps

    # Drop tempmax and tempmin columns
    merged_df.drop(columns=['tempmax', 'tempmin'], inplace=True)

    # Reset the index to include the datetime as a column
    merged_df.reset_index(inplace=True)
    merged_df.rename(columns={'index': 'datetime'}, inplace=True)

    # Round all values to 2 decimal places
    for column in merged_df.select_dtypes(include=[np.number]).columns:
        merged_df[column] = merged_df[column].round(2)

    return merged_df


def approximate_hourly_weather(daily_df):
    """
    Approximate hourly weather conditions from daily data.

    Parameters:
    daily_df (pd.DataFrame): DataFrame containing daily weather data.

    Returns:
    pd.DataFrame: DataFrame with approximated hourly weather conditions.
    """
    # Ensure datetime column is in datetime format
    if 'datetime' in daily_df.columns:
        daily_df['datetime'] = pd.to_datetime(daily_df['datetime'], errors='coerce')
    else:
        raise ValueError("The 'datetime' column is missing from the DataFrame.")

    # Set datetime as index
    daily_df.set_index('datetime', inplace=True)
    
    # Create a new DataFrame with hourly frequency
    hourly_index = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max() + pd.Timedelta(days=1), freq='H')[:-1]
    hourly_df = pd.DataFrame(index=hourly_index)
    
    # Merge the daily data with the hourly DataFrame
    merged_df = hourly_df.merge(daily_df, left_index=True, right_index=True, how='left')
    
    # Forward fill categorical columns and interpolate numeric columns
    for column in merged_df.columns:
        if column in ['tempmax', 'tempmin']:
            continue
        if np.issubdtype(merged_df[column].dtype, np.number):
            merged_df[column] = merged_df[column].interpolate(method='linear')
        else:
            merged_df[column] = merged_df[column].ffill()

    # Approximate hourly temperature using a sinusoidal function
    def calc_hourly_temp(row, hour):
        temp_range = row['tempmax'] - row['tempmin']
        return row['tempmin'] + (temp_range / 2) * (1 + np.cos((hour - 14) * np.pi / 12))

    hourly_temps = []
    for dt in hourly_df.index:
        day = dt.normalize()
        if day in daily_df.index:
            temp = calc_hourly_temp(daily_df.loc[day], dt.hour)
        else:
            temp = np.nan
        hourly_temps.append(temp)

    merged_df['temp'] = hourly_temps

    # Drop tempmax and tempmin columns
    merged_df.drop(columns=['tempmax', 'tempmin'], inplace=True)
    
    # Reset the index to include the datetime as a column
    merged_df.reset_index(inplace=True)
    merged_df.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Round all values to 2 decimal places
    for column in merged_df.select_dtypes(include=[np.number]).columns:
        merged_df[column] = merged_df[column].round(2)
    
    return merged_df