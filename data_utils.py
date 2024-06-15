from IPython.display import display
import pandas as pd
import numpy as np
import os
import re


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


def aggregate_dataframes(dict_of_dfs, date_column='Date', output_grain='hour', input_grain='minute'):
    """
    Aggregates data from the specified input grain to the specified output grain.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    date_column (str): The name of the date column in the DataFrames.
    output_grain (str): The time grain for aggregation ('hour', 'day', 'month'). Default is 'hour'.
    input_grain (str): The grain of the input data ('minute', 'hour', 'day'). Default is 'minute'.

    Returns:
    dict: Dictionary of aggregated DataFrames.
    """


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


def impute_missing_values(dict_of_dfs):
    for name, df in dict_of_dfs.items():
        numeric_cols = df.select_dtypes(include=['number']).columns
        dict_of_dfs[name][numeric_cols] = df[numeric_cols].apply(
            lambda x: x.fillna(x.mean()), axis=0)

    for name in dict_of_dfs:
        print(f'DataFrame: {name} - NaNs filled with column means')

    return dict_of_dfs

def get_min_max_dates(dict_of_dfs, date_column='Time'):
    """
    Extracts the earliest and latest dates from each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    date_column (str): Name of the date column to find the earliest and latest dates. Default is 'Time'.

    Returns:
    dict: Dictionary with DataFrame names as keys and dictionaries containing the earliest and latest dates as values.
    """
    date_info_dict = {}

    for name, df in dict_of_dfs.items():
        if date_column in df.columns:
            min_date = df[date_column].min()
            max_date = df[date_column].max()
            date_info_dict[name] = {'earliest_date': min_date, 'latest_date': max_date}
        else:
            print(f"Warning: DataFrame {name} does not contain the specified date column '{date_column}'. Skipping...")

    return date_info_dict

