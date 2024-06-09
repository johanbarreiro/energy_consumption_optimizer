import os
import re
from IPython.display import display
import pandas as pd

def get_dataframes(directory: str, verbose=False) -> dict:
    """
    Reads CSV files from the specified directory and returns a dictionary of DataFrames.

    Parameters:
    directory (str): The directory containing CSV files.
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

            df = pd.read_csv(filepath)
            dataframe_name = os.path.splitext(filename)[0]
            # Add a column with the dataframe name
            df['DataFrame Name'] = dataframe_name
            dict_of_dfs[dataframe_name] = df

            if verbose:
                print(f'DataFrame for {filename} created from {filepath}.')

    return dict_of_dfs
