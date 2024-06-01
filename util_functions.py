import argparse
import os
import pandas as pd


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


def restructure_dataframes(dict_of_dfs, date_column='Date'):
    for name, df in dict_of_dfs.items():
        # Add a column with the dataframe name
        columns = list(df.columns)

        if 'DataFrame Name' in columns:
            columns.remove('DataFrame Name')
            columns.insert(0, 'DataFrame Name')
        else:
            df.insert(0, 'DataFrame Name', name)

        # Ensure date_column is at the beginning
        if date_column in columns:
            columns.remove(date_column)
            columns.insert(1, date_column)

        # Ensure total_consumption(kWh)_hour is after the date column
        if 'total_consumption(kWh)_hour' in columns:
            columns.remove('total_consumption(kWh)_hour')
            columns.insert(2, 'total_consumption(kWh)_hour')

        # Ensure total_consumption(kW) is after total_consumption(kWh)_hour
        if 'total_consumption(kW)' in columns:
            columns.remove('total_consumption(kW)')
            columns.insert(3, 'total_consumption(kW)')

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

    all_columns = sorted(list(all_columns))  # Sort the columns to conserve order

    summary_data = []

    for name, df in dict_of_dfs.items():
        row = {'DataFrame Name': name}
        for column in all_columns:
            if column in df.columns:
                if column == 'DataFrame Name':
                    # Ensure 'DataFrame Name' column shows the actual dataframe name
                    row[column] = df[column].unique()[0]  # Assuming all values in this column are the same
                else:
                    row[column] = str(df[column].dtype)
            else:
                row[column] = 'Column not present'
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def convert_date_columns(dict_of_dfs, date_column):
    for name, df in dict_of_dfs.items():
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            dict_of_dfs[name] = df
            print(f'DataFrame: {name} - Converted {date_column} to datetime')
        else:
            print(f'DataFrame: {name} - {date_column} column not found')

    return dict_of_dfs


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


def add_total_consumption_column(dict_of_dfs):
    for name, df in dict_of_dfs.items():
        # Identify columns with 'kW' in their name
        kw_columns = [col for col in df.columns if 'kW' in col]

        # Calculate the total consumption, treating NaNs as zeros
        df['total_consumption(kW)'] = df[kw_columns].fillna(0).sum(axis=1)

        # Update the dataframe in the dictionary
        dict_of_dfs[name] = df

    for name in dict_of_dfs:
        print(f'DataFrame: {name} - Added total_consumption(kW) column')

    return dict_of_dfs


def add_kwh_column(dict_of_dfs, time_grain, date_column):
    """
    Adds a new column for total consumption in kWh to each DataFrame in the dictionary,
    resampling according to the specified time grain.
    
    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    time_grain (str): The time grain for resampling ('minute', 'hour', 'day', 'week', 'month').
    date_column (str): The name of the date column in the DataFrames.
    
    Returns:
    dict: The updated dictionary with the new kWh column added.
    """
    # Define conversion factors for different time grains, considering each row is per minute
    conversion_factors = {
        'minute': 1/60,  # kW to kWh per minute
        'hour': 1,       # kW to kWh per hour
        'day': 24,       # kW to kWh per day
        'week': 24*7,    # kW to kWh per week
        'month': 24*30   # kW to kWh per month (approximate month length as 30 days)
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
                df[f'total_consumption(kWh)_{time_grain}'] = df['total_consumption(kW)'] * conversion_factor
            else:
                # Resample and sum based on the time grain, then multiply by the conversion factor
                resampled_df = df['total_consumption(kW)'].resample(time_grain[0].upper()).sum() * conversion_factor
                resampled_df = resampled_df.rename(f'total_consumption(kWh)_{time_grain}')

                # Merge the resampled data back to the original dataframe
                df = df.merge(resampled_df, left_index=True, right_index=True, how='left')

                # Forward fill the merged column to align with original minute-level granularity
                df[f'total_consumption(kWh)_{time_grain}'] = df[f'total_consumption(kWh)_{time_grain}'].fillna(method='ffill')

            df.reset_index(inplace=True)
            dict_of_dfs[name] = df
            print(f"DataFrame: {name} - Added total_consumption(kWh) column for time grain: {time_grain}")
        else:
            print(f"DataFrame: {name} - total_consumption(kW) column not found")

    return dict_of_dfs


def impute_missing_values(dict_of_dfs):
    for name, df in dict_of_dfs.items():
        numeric_cols = df.select_dtypes(include=['number']).columns
        dict_of_dfs[name][numeric_cols] = df[numeric_cols].apply(
            lambda x: x.fillna(x.mean()), axis=0)

    for name in dict_of_dfs:
        print(f'DataFrame: {name} - NaNs filled with column means')

    return dict_of_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load CSV files into dataframes.")
    parser.add_argument('-d', '--directory', type=str,
                        help='The directory containing CSV files.')

    args = parser.parse_args()

    if args.directory:
        directory = args.directory
    else:
        directory = input("Please enter the directory containing CSV files: ")

    dataframes = get_dataframes(directory)
    print(f"Total dataframes created: {len(dataframes)}")
