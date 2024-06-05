import os
import pandas as pd
import re
import numpy as np
from IPython.display import display


def get_dataframes(directory: str) -> dict:
    dataframes = {}

    # Debug: Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return dataframes

    # Debug: Print the directory being processed
    print(f"Processing directory: {directory}")

    for filename in os.listdir(directory):
        # Debug: Print each filename being processed
        print(f"Found file: {filename}")

        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            # Debug: Print the full filepath
            print(f"Reading CSV file: {filepath}")

            df = pd.read_csv(filepath)
            dataframe_name = os.path.splitext(filename)[0]
            # Add a column with the dataframe name
            df['DataFrame Name'] = dataframe_name
            dataframes[dataframe_name] = df

            print(f'DataFrame for {filename} created from {filepath}.')

    return dataframes


def restructure_dataframes(dict_of_dfs, date_column='Date', verbose=False):
    """
    Restructures DataFrames in the dictionary by moving specific columns to the beginning.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to restructure.
    date_column (str): The name of the date column to move to the beginning. Default is 'Date'.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: Dictionary of restructured DataFrames.
    """
    for name, df in dict_of_dfs.items():
        # Add a column with the dataframe name
        columns = list(df.columns)

        if 'DataFrame Name' in columns:
            columns.remove('DataFrame Name')
            columns.insert(0, 'DataFrame Name')
            if verbose:
                print(f"DataFrame: {name} - 'DataFrame Name' column moved")
        else:
            df.insert(0, 'DataFrame Name', name)

        # Ensure date_column is at the beginning
        if date_column in columns:
            columns.remove(date_column)
            columns.insert(1, date_column)
            if verbose:
                print(f"DataFrame: {name} - '{date_column}' column moved")

        # Ensure total_consumption(kWh)_hour is after the date column
        if 'total_consumption(kWh)_hour' in columns:
            columns.remove('total_consumption(kWh)_hour')
            columns.insert(2, 'total_consumption(kWh)_hour')
            if verbose:
                print(
                    f"DataFrame: {name} - 'total_consumption(kWh)_hour' column moved")

        # Ensure total_consumption(kW) is after total_consumption(kWh)_hour
        if 'total_consumption(kW)' in columns:
            columns.remove('total_consumption(kW)')
            columns.insert(3, 'total_consumption(kW)')
            if verbose:
                print(
                    f"DataFrame: {name} - 'total_consumption(kW)' column moved")

        # Reorder the dataframe columns
        df = df[columns]

        # Update the dataframe in the dictionary
        dict_of_dfs[name] = df

    return dict_of_dfs


def check_column_names(dict_of_dfs):
    for name, df in dict_of_dfs.items():
        print(f"DataFrame: {name}")
        print(f"Columns: {list(df.columns)}\n")


def column_presence_checker(dict_of_dfs):
    all_columns = set()
    for df in dict_of_dfs.values():
        all_columns.update(df.columns)

    # Sort the columns to conserve order
    all_columns = sorted(list(all_columns))

    summary_data = []

    for name, df in dict_of_dfs.items():
        row = {'DataFrame Name': name}
        for column in all_columns:
            if column in df.columns:
                if column == 'DataFrame Name':
                    # Ensure 'DataFrame Name' column shows the actual dataframe name
                    # Assuming all values in this column are the same
                    row[column] = df[column].unique()[0]
                else:
                    row[column] = str(df[column].dtype)
            else:
                row[column] = 'Column not present'
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def convert_date_columns(dict_of_dfs, date_column, verbose=False):
    """
    Converts the specified date column to datetime in each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to convert date columns.
    date_column (str): The name of the date column to convert to datetime.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: Dictionary of DataFrames with the date column converted to datetime.
    """
    for name, df in dict_of_dfs.items():
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            dict_of_dfs[name] = df
            if verbose:
                print(
                    f"DataFrame: {name} - Converted {date_column} to datetime")
        else:
            if verbose:
                print(f"DataFrame: {name} - {date_column} column not found")

    return dict_of_dfs


def transform_date_features_dict(dict_of_dfs, date_column):
    """
    Transforms the date column in each DataFrame within the dictionary to extract and encode date-related features.

    Parameters:
    dfs_dict (dict): Dictionary of DataFrames.
    date_column (str): Name of the date column in each DataFrame.

    Returns:
    dict: Dictionary with DataFrames containing the transformed date features.
    """
    transformed_dict_of_dfs = {}

    for key, df in dict_of_dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])

        # Extract date features
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0)

        # Cyclical encoding for month and day of week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Drop original columns if not needed
        df.drop(['month', 'day', 'day_of_week',
                'week_of_year'], axis=1, inplace=True)

        transformed_dict_of_dfs[key] = df

    return transformed_dict_of_dfs


def aggregate_data(dict_of_dfs, date_column, time_grain):
    """
    Aggregates data from minute granularity to the specified time grain.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames with minute granularity.
    date_column (str): The name of the date column in the DataFrames.
    time_grain (str): The time grain for aggregation ('hour', 'day', 'month').

    Returns:
    dict: Dictionary of aggregated DataFrames.
    """
    # Define resampling rules
    resample_rule = {
        'hour': 'H',
        'day': 'D',
        'month': 'M'
    }

    if time_grain not in resample_rule:
        raise ValueError(
            f"Invalid time grain: {time_grain}. Must be one of {list(resample_rule.keys())}")

    # Resample each DataFrame
    aggregated_dfs = {}
    for name, df in dict_of_dfs.items():
        # Ensure the date column is datetime type and set as index for resampling
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)

        # Resample and aggregate using the mean
        resampled_df = df.resample(resample_rule[time_grain]).mean()

        # Reset the index to move the date column back to a column
        resampled_df.reset_index(inplace=True)

        aggregated_dfs[name] = resampled_df

    return aggregated_dfs


def nan_checker(dict_of_dfs):
    all_columns = set()
    for df in dict_of_dfs.values():
        all_columns.update(df.columns)

    all_columns = sorted(list(all_columns))

    summary_data = []

    for name, df in dict_of_dfs.items():
        row = {'DataFrame Name': name}
        for column in all_columns:
            if column in df.columns:
                # Check the NaN status for the column
                total_rows = len(df)
                num_nans = df[column].isna().sum()
                if num_nans == 0:
                    row[column] = 'No NaNs'
                elif num_nans == total_rows:
                    row[column] = 'All NaNs'
                else:
                    row[column] = 'Some NaNs'
            else:
                row[column] = 'Column not present'
        summary_data.append(row)

    # Convert the summary data to a dataframe
    summary_df = pd.DataFrame(summary_data)

    # Display the summary dataframe
    return summary_df


def convert_kw_to_kwh(dict_of_dfs, grain='minute'):
    """
    Converts all kW columns to kWh in each DataFrame in the dictionary based on the specified grain and renames the columns.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames with kW data.
    grain (str): The grain of the data ('minute', 'hour', 'day'). Default is 'minute'.

    Returns:
    dict: Dictionary of DataFrames with kWh data and renamed columns.
    """
    # Define conversion factors for different grains
    conversion_factors = {
        'minute': 1/60,  # kW to kWh per minute
        'hour': 1,       # kW to kWh per hour
        'day': 24,       # kW to kWh per day
    }

    if grain not in conversion_factors:
        raise ValueError(
            f"Invalid grain: {grain}. Must be one of {list(conversion_factors.keys())}")

    conversion_factor = conversion_factors[grain]

    converted_dfs = {}

    for name, df in dict_of_dfs.items():
        # Create a copy of the DataFrame to avoid modifying the original
        df_converted = df.copy()

        # Convert kW to kWh for all columns that contain 'kW' in their name
        kw_columns = [col for col in df_converted.columns if '(kW)' in col]
        for col in kw_columns:
            # Convert kW to kWh based on the specified grain
            df_converted[col] = df_converted[col] * conversion_factor
            # Rename the column to reflect the conversion to kWh
            new_col_name = col.replace('(kW)', '(kWh)')
            df_converted.rename(columns={col: new_col_name}, inplace=True)

        converted_dfs[name] = df_converted

    return converted_dfs


def add_total_consumption_column(dict_of_dfs, verbose=False):
    """
    Adds a total consumption column (total_consumption(kW)) to each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to add the total consumption column.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: Dictionary of DataFrames with the total consumption column added.
    """
    for name, df in dict_of_dfs.items():
        # Identify columns with 'kW' in their name
        kw_columns = [col for col in df.columns if 'kW' in col]

        # Calculate the total consumption, treating NaNs as zeros
        df['total_consumption(kW)'] = df[kw_columns].fillna(0).sum(axis=1)

        # Update the dataframe in the dictionary
        dict_of_dfs[name] = df

        if verbose:
            print(f"DataFrame: {name} - Added total_consumption(kW) column")

    return dict_of_dfs


def add_kwh_column(dict_of_dfs, time_grain, date_column, verbose=False):
    """
    Adds a new column for total consumption in kWh to each DataFrame in the dictionary,
    resampling according to the specified time grain.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    time_grain (str): The time grain for resampling ('minute', 'hour', 'day', 'week', 'month').
    date_column (str): The name of the date column in the DataFrames.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: The updated dictionary with the new kWh column added.
    """
    # Define conversion factors for different time grains, considering each row is per minute
    conversion_factors = {
        'minute': 1/60,  # kW to kWh per minute
        'hour': 1,       # kW to kWh per hour
        'day': 24,       # kW to kWh per day
        'week': 24*7,    # kW to kWh per week
        # kW to kWh per month (approximate month length as 30 days)
        'month': 24*30
    }

    # Check if the time grain is valid
    if time_grain not in conversion_factors:
        raise ValueError(
            f"Invalid time grain: {time_grain}. Must be one of {list(conversion_factors.keys())}")

    # Get the conversion factor for the specified time grain
    conversion_factor = conversion_factors[time_grain]

    for name, df in dict_of_dfs.items():
        if 'total_consumption(kW)' in df.columns:
            # Ensure the date column is datetime type and set as index for resampling
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)

            if time_grain == 'minute':
                # For minute-level data, multiply the kW values directly by the conversion factor
                df[f'total_consumption(kWh)_{time_grain}'] = df['total_consumption(kW)'] * \
                    conversion_factor
            else:
                # Resample and sum based on the time grain, then multiply by the conversion factor
                resampled_df = df['total_consumption(kW)'].resample(
                    time_grain[0].upper()).sum() * conversion_factor
                resampled_df = resampled_df.rename(
                    f'total_consumption(kWh)_{time_grain}')

                # Merge the resampled data back to the original dataframe
                df = df.merge(resampled_df, left_index=True,
                              right_index=True, how='left')

                # Forward fill the merged column to align with original minute-level granularity
                df[f'total_consumption(kWh)_{time_grain}'] = df[f'total_consumption(kWh)_{time_grain}'].fillna(
                    method='ffill')

            df.reset_index(inplace=True)
            dict_of_dfs[name] = df
            if verbose:
                print(
                    f"DataFrame: {name} - Added total_consumption(kWh) column for time grain: {time_grain}")
        else:
            if verbose:
                print(
                    f"DataFrame: {name} - total_consumption(kW) column not found")

    return dict_of_dfs


def impute_missing_values(dict_of_dfs):
    for name, df in dict_of_dfs.items():
        numeric_cols = df.select_dtypes(include=['number']).columns
        dict_of_dfs[name][numeric_cols] = df[numeric_cols].apply(
            lambda x: x.fillna(x.mean()), axis=0)

    for name in dict_of_dfs:
        print(f'DataFrame: {name} - NaNs filled with column means')

    return dict_of_dfs


def merge_datasets_by_floor(dict_of_dfs):
    """
    Merges datasets within a dictionary of DataFrames based on year and floor number,
    with output DataFrame names as just the floor number.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames with names containing year and floor information.

    Returns:
    dict: Dictionary of merged DataFrames with keys as 'Floor' (e.g., 'Floor1').
    """
    merged_dfs = {}

    # Function to extract floor from DataFrame name
    def extract_floor(name):
        match = re.match(r"(\d{4})(Floor\d+)", name)
        if match:
            _, floor = match.groups()
            return floor
        return None

    # Group DataFrames by floor
    groups = {}
    for name, df in dict_of_dfs.items():
        key = extract_floor(name)
        if key:
            if key not in groups:
                groups[key] = []
            groups[key].append(df)

    # Merge DataFrames within each group
    for key, dfs in groups.items():
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_dfs[key] = merged_df

    return merged_dfs


def preview_dict_of_dfs(dict_of_dfs, n=5):
    """
    Previews the first n rows of each DataFrame in the dictionary.

    Parameters:
    dfs_dict (dict): Dictionary of DataFrames.
    n (int): Number of rows to preview from each DataFrame.

    Returns:
    None
    """
    for key, df in dict_of_dfs.items():
        print(f"Preview of DataFrame for {key}:")
        display(df.head(n))
        print("\n")

def collect_columns_by_unit(dict_of_dfs):
    """
    Collects column names by unit of measurement for each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to analyze.

    Returns:
    dict: A dictionary of dictionaries with stored lists of column names according to unit.
    """
    units = {
        'energy': '(kWh)',
        'power': '(kW)',
        'temperature': '(degC)',
        'relative humidity': '(RH%)',
        'ambient lighting': '(lux)'
    }

    collected_columns = {}

    for name, df in dict_of_dfs.items():
        collected_columns[name] = {
            'energy': [],
            'power': [],
            'temperature': [],
            'relative humidity': [],
            'ambient lighting': []
        }
        
        for unit, indicator in units.items():
            columns = [col for col in df.columns if indicator in col]
            collected_columns[name][unit].extend(columns)

    return collected_columns