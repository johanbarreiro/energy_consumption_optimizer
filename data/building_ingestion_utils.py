import os
import re
from IPython.display import display
import pandas as pd
import json
import numpy as np


def get_dataframes(directory: str, delim: str, verbose=False) -> dict:
    """
    Reads CSV files from the specified directory and returns a dictionary of DataFrames.

    Parameters:
    directory (str): The directory containing CSV files.
    delim (str): The delimiter used in the CSV files.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: A dictionary where keys are the filenames (without extension) and values are the DataFrames.
    """
    dict_of_dfs = {}

    # Check if the directory exists
    if not os.path.exists(directory):
        if verbose:
            print(f"The directory '{directory}' does not exist.")
        return dict_of_dfs

    # Print the directory being processed
    if verbose:
        print(f"Processing directory: {directory}")

    for filename in os.listdir(directory):
        # Print each filename being processed
        if verbose:
            print(f"Found file: {filename}")

        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            # Print the full filepath
            if verbose:
                print(f"Reading CSV file: {filepath}")

            df = pd.read_csv(filepath, sep=delim)

            # Remove 'Unnamed' columns
            unnamed_columns = [
                col for col in df.columns if col.startswith('Unnamed')]
            if unnamed_columns and verbose:
                print(f"Removing columns: {unnamed_columns}")
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            dataframe_name = os.path.splitext(filename)[0]
            # Add a column with the dataframe name
            df['DataFrame Name'] = dataframe_name
            dict_of_dfs[dataframe_name] = df

            if verbose:
                print(f'DataFrame for {filename} created from {filepath}.')

    return dict_of_dfs


def rename_columns_in_dict_of_dfs(dict_of_dfs, old_string, new_string, verbose=False):
    """
    Rename columns in each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    old_string (str): The string to find in the column names.
    new_string (str): The string to replace the old_string with.
    verbose (bool): If True, print detailed information about the changes.

    Returns:
    dict: Dictionary of DataFrames with renamed columns.
    """
    renamed_dict_of_dfs = {}

    for key, df in dict_of_dfs.items():
        # Create a dictionary to map old column names to new column names
        new_column_names = {col: col.replace(
            old_string, new_string) for col in df.columns}

        # Print changes if verbose is True
        if verbose:
            for old_col, new_col in new_column_names.items():
                if old_col != new_col:
                    print(
                        f"Renaming column '{old_col}' to '{new_col}' in DataFrame '{key}'")

        # Rename the columns in the DataFrame
        renamed_df = df.rename(columns=new_column_names)

        # Add the renamed DataFrame to the new dictionary
        renamed_dict_of_dfs[key] = renamed_df

    return renamed_dict_of_dfs


def read_units_dict_from_json(file_path, verbose=False):
    """
    Read the units dictionary from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing the units dictionary.
    verbose (bool): If True, print detailed information about the operation. Default is False.

    Returns:
    dict: The units dictionary.
    """
    if verbose:
        print(f"Attempting to read the units dictionary from '{file_path}'")

    try:
        with open(file_path, 'r') as json_file:
            units_dict = json.load(json_file)

        if verbose:
            print(f"Successfully read the units dictionary from '{file_path}'")
            print(f"Dictionary content: {units_dict}")

        return units_dict
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{file_path}': {e}")
        return None


def add_entry_to_units_dict(file_path, new_entry_key, new_entry_value, verbose=False):
    """
    Add a new entry to the units dictionary stored in a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing the units dictionary.
    new_entry_key (str): The key for the new entry to be added.
    new_entry_value (str): The value for the new entry to be added.
    verbose (bool): If True, print detailed information about the operation. Default is False.
    """
    if verbose:
        print(
            f"Attempting to add entry '{new_entry_key}: {new_entry_value}' to '{file_path}'")

    try:
        # Read the existing dictionary from the JSON file
        with open(file_path, 'r') as json_file:
            units_dict = json.load(json_file)

        if verbose:
            print(
                f"Successfully read the current units dictionary from '{file_path}'")

        # Add the new entry to the dictionary
        units_dict[new_entry_key] = new_entry_value

        if verbose:
            print(f"Updated dictionary content: {units_dict}")

        # Write the updated dictionary back to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(units_dict, json_file, indent=4)

        if verbose:
            print(
                f"Successfully added entry '{new_entry_key}: {new_entry_value}' to '{file_path}'")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{file_path}'.")


def add_units_to_column_names(dict_of_dfs, units_dict, verbose=False):
    """
    Adds units to the column names in each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to add units to column names.
    units_dict (dict): Dictionary mapping column name keywords to their respective units.
    verbose (bool): If True, print detailed information about the operation. Default is False.

    Returns:
    dict: Dictionary of DataFrames with updated column names.
    """
    updated_dfs = {}

    for name, df in dict_of_dfs.items():
        if verbose:
            print(f"Processing DataFrame: {name}")

        new_columns = []
        for col in df.columns:
            new_col = col
            for keyword, unit in units_dict.items():
                if keyword in col and not col.endswith(f"({unit})"):
                    new_col = f"{col} ({unit})"
                    if verbose:
                        print(
                            f"Renaming column '{col}' to '{new_col}' in DataFrame '{name}'")
                    break
            new_columns.append(new_col)

        df.columns = new_columns
        updated_dfs[name] = df

        if verbose:
            print(
                f"Updated column names for DataFrame '{name}': {df.columns.tolist()}")

    return updated_dfs


def create_total_column_by_units(dict_of_dfs, col_string, verbose=False):
    """
    Sum columns containing a specific string in their names to create a total column.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    col_string (str): String contained in column names to be summed.
    verbose (bool): If True, print detailed information about the operations.

    Returns:
    dict: Dictionary of DataFrames with new total columns added.
    """
    # Load units_dict from JSON
    units_dict = read_units_dict_from_json('data/unit_dict.json', verbose=verbose)

    if units_dict is None:
        raise ValueError("Units dictionary could not be read. Please check the JSON file.")

    for name, df in dict_of_dfs.items():
        if verbose:
            print(f"Processing DataFrame: {name}")
        for unit_key, unit_value in units_dict.items():
            if col_string in unit_key:
                columns_to_sum = [col for col in df.columns if unit_value in col and not col.startswith('Total_')]
                if columns_to_sum:
                    total_col_name = f'Total_{unit_key}_{unit_value}'
                    if verbose:
                        print(f"Columns to sum for '{unit_value}': {columns_to_sum}")
                    df[total_col_name] = df[columns_to_sum].sum(axis=1)
                    dict_of_dfs[name] = df
                    if verbose:
                        print(f"DataFrame: {name} - Created column: {total_col_name} summing columns: {columns_to_sum}")
                else:
                    if verbose:
                        print(f"No columns found to sum for unit: {unit_value}")
                break  # Assuming one unit matches col_string, no need to check further
    return dict_of_dfs
