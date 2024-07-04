import datetime as dt
import pandas as pd
import os
import requests

from OMIEData.DataImport.omie_marginalprice_importer import OMIEMarginalPriceFileImporter
from OMIEData.Enums.all_enums import DataTypeInMarginalPriceFile


def import_omie_marginal_price_data(date_ini, date_end, verbose=False):
    """
    Imports OMIE marginal price data within the specified date range.

    Parameters:
    date_ini (datetime.datetime): Start date for importing data.
    date_end (datetime.datetime): End date for importing data.
    verbose (bool): If True, print detailed information about the import process. Default is True.

    Returns:
    pd.DataFrame: DataFrame containing the imported OMIE marginal price data.
    """
    if verbose:
        print(f"Attempting to import OMIE marginal price data from {date_ini} to {date_end}...")

    # Import data from OMIE
    df = OMIEMarginalPriceFileImporter(date_ini=date_ini, date_end=date_end).read_to_dataframe(verbose=verbose)

    # Sort values by 'DATE'
    df.sort_values(by='DATE', axis=0, inplace=True)

    if verbose:
        print("Import successful.")
        print(f"DataFrame shape: {df.shape}")
        print(f"Start date: {df['DATE'].min()}")
        print(f"End date: {df['DATE'].max()}")

    return df


def fetch_omie_data(start_date, end_date, verbose=False):
    """
    Fetches OMIE (Spanish energy prices) data for a given date range.

    Parameters:
    start_date (datetime): The start date for the data fetch.
    end_date (datetime): The end date for the data fetch.
    verbose (bool): If True, print detailed information about the operations.

    Returns:
    pd.DataFrame: A DataFrame containing the fetched data.
    """
    base_url = "https://www.omie.es/sites/default/files/dados/AGNO_{year}/MES_{month}/TXT/INT_PBC_EV_H_{start_date}_{end_date}.TXT"
    date_range = pd.date_range(start=start_date, end=end_date)
    data_frames = []

    for single_date in date_range:
        year = single_date.year
        month = f"{single_date.month:02d}"
        day = f"{single_date.day:02d}"
        url = base_url.format(year=year, month=month, start_date=f"{day}_{month}_{year}",
                              end_date=f"{day}_{month}_{year}")

        if verbose:
            print(f"Fetching data for date: {single_date.strftime('%Y-%m-%d')} from URL: {url}")

        response = requests.get(url)
        if response.status_code == 200:
            if verbose:
                print(f"Successfully fetched data for date: {single_date.strftime('%Y-%m-%d')}")
            # Read the data into a pandas DataFrame
            data = response.text
            df = pd.read_csv(StringIO(data), sep=';', skiprows=1, header=None)
            data_frames.append(df)
        else:
            if verbose:
                print(
                    f"Failed to fetch data for date: {single_date.strftime('%Y-%m-%d')} with status code: {response.status_code}")

    if data_frames:
        # Combine all data frames into one
        combined_df = pd.concat(data_frames, ignore_index=True)
        if verbose:
            print(f"Successfully combined data frames for the date range {start_date} to {end_date}")
        return combined_df
    else:
        if verbose:
            print(f"No data fetched for the date range {start_date} to {end_date}")
        return pd.DataFrame()


import pandas as pd


def merge_price_data(dataframes_dict, csv_directory, df_datetime_col, csv_datetime_col, price_col, verbose=False):
    """
    Merge hourly price data from multiple CSV files in a directory into a dictionary of dataframes.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are dataframe identifiers and values are pandas dataframes.
    - csv_directory (str): Path to the directory containing the CSV files.
    - df_datetime_col (str): The name of the datetime column in the dataframes.
    - csv_datetime_col (str): The name of the datetime column in the CSV files.
    - price_col (str): The name of the price column in the CSV files to be merged.
    - verbose (bool): If True, print debugging information.

    Returns:
    - dict: A dictionary with the same keys as input, with each dataframe updated to include the price data.
    """

    # Read all CSV files in the directory and concatenate them into a single DataFrame
    csv_files = [os.path.join(csv_directory, file) for file in os.listdir(csv_directory) if file.endswith('.csv')]
    csv_data_list = [pd.read_csv(file) for file in csv_files]
    csv_data = pd.concat(csv_data_list, ignore_index=True)

    # Strip whitespace from column names
    csv_data.columns = csv_data.columns.str.strip()

    # Convert the datetime column in the CSV to datetime format
    csv_data[csv_datetime_col] = pd.to_datetime(csv_data[csv_datetime_col])

    # Ensure the CSV data is sorted by datetime
    csv_data = csv_data.sort_values(by=csv_datetime_col)

    # Set the datetime column as the index
    csv_data.set_index(csv_datetime_col, inplace=True)

    # Print the initial CSV data to debug if verbose is True
    if verbose:
        print("Initial CSV Data:")
        print(csv_data.head())
        print("CSV Columns:", csv_data.columns.tolist())

    # Check if the price column exists in the CSV data
    if price_col not in csv_data.columns:
        raise KeyError(
            f"'{price_col}' not found in the CSV data columns. Available columns are: {csv_data.columns.tolist()}")

    # Resample to hourly intervals (if necessary)
    hourly_csv_data = csv_data.resample('h').first()

    # Print the resampled CSV data to debug if verbose is True
    if verbose:
        print("Resampled CSV Data:")
        print(hourly_csv_data.head())
        print("Resampled CSV Columns:", hourly_csv_data.columns.tolist())

    # Check if the price column exists in the resampled data
    if price_col not in hourly_csv_data.columns:
        raise KeyError(
            f"'{price_col}' not found in the resampled CSV data columns. Available columns are: {hourly_csv_data.columns.tolist()}")

    # Process each dataframe in the dictionary
    for key, df in dataframes_dict.items():
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Convert the datetime column in the dataframe to datetime format
        df[df_datetime_col] = pd.to_datetime(df[df_datetime_col])

        # Align the datetime column to hourly intervals
        df['datetime_hourly'] = df[df_datetime_col].dt.floor('h')

        # Print the dataframe before merging to debug if verbose is True
        if verbose:
            print(f"Dataframe '{key}' before merging:")
            print(df.head())
            print("Dataframe Columns:", df.columns.tolist())

        # Merge the CSV data with the dataframe, handling suffixes
        merged_df = pd.merge(df, hourly_csv_data[[price_col]], left_on='datetime_hourly', right_index=True, how='left',
                             suffixes=('', '_csv'))

        # Print the merged dataframe to debug if verbose is True
        if verbose:
            print(f"Merged Dataframe '{key}':")
            print(merged_df.head())
            print("Merged Dataframe Columns:", merged_df.columns.tolist())

        # Check if the price column already exists in the dataframe
        if price_col in df.columns:
            # Update only the null values in the existing price column
            df['price_mWh'] = df[price_col].combine_first(merged_df[price_col])
        else:
            # Add the price column to the dataframe
            df['price_mWh'] = merged_df[price_col]


        # Drop the temporary hourly datetime column
        df.drop(columns=['datetime_hourly'], inplace=True)

        # Update the dictionary with the modified dataframe
        dataframes_dict[key] = df

    return dataframes_dict



