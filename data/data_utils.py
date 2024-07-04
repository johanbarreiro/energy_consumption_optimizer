from IPython.display import display
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime


def convert_and_transform_date_columns(dict_of_dfs: dict[str, pd.DataFrame], date_column: str,
                                       datetime_format: str = '%Y/%m/%d %H:%M:%S', verbose: bool = False) -> dict[
    str, pd.DataFrame]:
    """
    Converts the specified date column to datetime and transforms it to extract date-related features
    in each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to convert and transform date columns.
    date_column (str): The name of the date column to convert to datetime.
    datetime_format (str): Format of the datetime column. Default is '%Y/%m/%d %H:%M:%S'.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: Dictionary of DataFrames with the date column converted to datetime and transformed.
    """
    transformed_dict_of_dfs = {}

    for name, df in dict_of_dfs.items():
        if date_column in df.columns:
            try:
                df[date_column] = pd.to_datetime(
                    df[date_column], format=datetime_format, errors='coerce')
                # Ensure all entries are pandas.Timestamp
                df[date_column] = df[date_column].apply(lambda x: pd.Timestamp(x) if pd.notnull(x) else pd.NaT)
                if verbose:
                    print(
                        f"DataFrame: {name} - Converted {date_column} to datetime")

                # Extract date features
                df['month'] = df[date_column].dt.month
                df['day'] = df[date_column].dt.day
                df['day_of_week'] = df[date_column].dt.dayofweek
                df['week_of_year'] = df[date_column].dt.isocalendar().week
                df['hour'] = df[date_column].dt.hour
                df['day_of_month'] = df[date_column].dt.day

                df['is_weekend'] = df['day_of_week'].apply(
                    lambda x: 1 if x >= 5 else 0)

                # Cyclical encoding for hour, month, day of week, and day of month
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['day_of_week_sin'] = np.sin(
                    2 * np.pi * df['day_of_week'] / 7)
                df['day_of_week_cos'] = np.cos(
                    2 * np.pi * df['day_of_week'] / 7)
                df['day_of_month_sin'] = np.sin(
                    2 * np.pi * df['day_of_month'] / 31)
                df['day_of_month_cos'] = np.cos(
                    2 * np.pi * df['day_of_month'] / 31)

                # Drop original columns if not needed
                df.drop(['month', 'day', 'day_of_week', 'week_of_year', 'hour', 'day_of_month'], axis=1, inplace=True)

                transformed_dict_of_dfs[name] = df

            except Exception as e:
                print(
                    f"DataFrame: {name} - Error converting {date_column} to datetime: {e}")
        else:
            if verbose:
                print(f"DataFrame: {name} - {date_column} column not found")

    return transformed_dict_of_dfs


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


def impute_missing_values(dict_of_dfs, method='linear', verbose=False):
    for name, df in dict_of_dfs.items():
        numeric_cols = df.select_dtypes(include=['number']).columns

        if method == 'linear':
            dict_of_dfs[name][numeric_cols] = df[numeric_cols].interpolate(method='linear', axis=0)
        elif method == 'mean':
            dict_of_dfs[name][numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
        elif method == 'median':
            dict_of_dfs[name][numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()), axis=0)
        else:
            raise ValueError(f"Unsupported imputation method: {method}")

        if verbose:
            print(f'DataFrame: {name} - NaNs filled using {method} method')

    return dict_of_dfs


def clean_data_by_column(data_dict, column_suffix='_Power Factor (real)',
                         negative_value_strategy='abs', outlier_strategy='median',
                         missing_value_strategy='median', verbose=False):
    """
    Cleans the data in a dictionary of dataframes based on specified column suffix.

    Args:
        data_dict (dict): Dictionary of pandas DataFrames containing data.
        column_suffix (str): Suffix to identify columns to be cleaned.
        negative_value_strategy (str): Strategy to handle negative values. Options are 'abs' to take absolute value,
                                       'remove' to drop negative values.
        outlier_strategy (str): Strategy to handle outliers. Options are 'median' to replace outliers with the median value,
                                'remove' to drop outliers.
        missing_value_strategy (str): Strategy to handle missing values. Options are 'median' to fill missing values with the median,
                                      'mean' to fill with the mean, 'drop' to drop rows with missing values.
        verbose (bool): If True, print detailed information during the cleaning process.

    Returns:
        dict: Dictionary of cleaned pandas DataFrames.
    """
    for df_name, df in data_dict.items():
        if verbose:
            print(f"Processing DataFrame: {df_name}")

        # Identify columns based on suffix
        target_cols = [col for col in df.columns if col.endswith(column_suffix)]

        for col in target_cols:
            if verbose:
                print(f"  Cleaning column: {col}")

            # Handle negative values
            if negative_value_strategy == 'abs':
                negative_count = (df[col] < 0).sum()
                df[col] = df[col].apply(lambda x: abs(x) if x < 0 else x)
                if verbose:
                    print(f"    Negative values handled by abs: {negative_count}")
            elif negative_value_strategy == 'remove':
                negative_count = (df[col] < 0).sum()
                df = df[df[col] >= 0]
                if verbose:
                    print(f"    Negative values removed: {negative_count}")

            # Handle outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if outlier_strategy == 'median':
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                median_value = df[col].median()
                df[col] = df[col].apply(lambda x: x if (x >= lower_bound and x <= upper_bound) else median_value)
                if verbose:
                    print(f"    Outliers handled by median: {outliers_count}")
            elif outlier_strategy == 'remove':
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                if verbose:
                    print(f"    Outliers removed: {outliers_count}")

            # Handle missing values
            if missing_value_strategy == 'median':
                missing_count = df[col].isna().sum()
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                if verbose:
                    print(f"    Missing values handled by median: {missing_count}")
            elif missing_value_strategy == 'mean':
                missing_count = df[col].isna().sum()
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                if verbose:
                    print(f"    Missing values handled by mean: {missing_count}")
            elif missing_value_strategy == 'drop':
                missing_count = df[col].isna().sum()
                df.dropna(subset=[col], inplace=True)
                if verbose:
                    print(f"    Rows with missing values dropped: {missing_count}")

        # Update the dataframe in the dictionary
        data_dict[df_name] = df

    return data_dict


def calculate_active_energy(dict_of_dfs, verbose=False):
    """
    Calculate Active Energy (kWh) from Active Power (kW) for each DataFrame in a dictionary of DataFrames.

    This function iterates through each DataFrame in the provided dictionary, identifies columns with the
    suffix '_Active Power (kW)', and computes the Active Energy values based on the formula:
    Active Energy (kWh) = Active Power (kW) * 0.25. The new Active Energy columns are added to the respective DataFrames.

    Parameters:
    dict_of_dfs (dict): A dictionary where keys are strings and values are pandas DataFrames.
    Data should have the grain of 15 minutes.
    Each DataFrame should contain columns with the suffix '_Active Power (kW)'.
    verbose (bool): If True, the function will print the name of each new Active Energy column created
                    and the DataFrame it was added to. Default is False.

    Returns:
    dict: The same dictionary of DataFrames, with each DataFrame updated to include the new Active Energy columns.
    """
    for name, df in dict_of_dfs.items():
        # Find all columns that match the naming convention for Active Power
        active_power_cols = [col for col in df.columns if '_Active Power (kW)' in col]

        # Calculate Active Energy for each Active Power column
        for active_power_col in active_power_cols:
            prefix = active_power_col.replace('_Active Power (kW)', '')
            active_energy_col = f'{prefix}_Active Energy (kWh)'
            df[active_energy_col] = df[active_power_col] * 0.25

            if verbose:
                print(f'Calculated {active_energy_col} in DataFrame: {name}')

    return dict_of_dfs


def calculate_apparent_power(dict_of_dfs):
    """
    Calculate the kVa values from kW and Power Factor for each DataFrame in a dictionary of DataFrames.

    This function iterates through each DataFrame in the provided dictionary, identifies columns with
    suffixes '_Power Factor (real)' and '_Active Power (kW)', and computes the kVa values based on the
    formula: kVa = kW / Power Factor. The new kVa columns are added to the respective DataFrames.

    Parameters:
    dict_of_dfs (dict): A dictionary where keys are strings and values are pandas DataFrames.
                        Each DataFrame should contain columns with suffixes '_Power Factor (real)'
                        and '_Active Power (kW)'.

    Returns:
    dict: The same dictionary of DataFrames, with each DataFrame updated to include the new kVa columns.

    """
    for key, df in dict_of_dfs.items():
        # Find all columns with the suffix _Power Factor (real)
        power_factor_cols = [col for col in df.columns if col.endswith('_Power Factor (real)')]

        for pf_col in power_factor_cols:
            # Extract the base name (prefix) of the column
            base_name = pf_col[:-20]  # removes '_Power Factor (real)'

            # Identify the corresponding Active Power (kW) column
            kw_col = f"{base_name}_Active Power (kW)"

            # Calculate the kVa column if the Active Power column exists
            if kw_col in df.columns:
                kva_col = f"{base_name}_Apparent Power (kVa)"
                df[kva_col] = df[kw_col] / df[pf_col]

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


def add_sum_column(dict_of_dfs, text, exclude_cols):
    """
    Adds a column to each dataframe in the dictionary that sums all columns
    with a certain text in the column name, excluding specified columns.

    Parameters:
    dict_of_dfs (dict): Dictionary where keys are names and values are dataframes.
    text (str): Text to search for in the column names.
    exclude_cols (list): List of columns to exclude from the sum.
    """
    for key, df in dict_of_dfs.items():
        # Find columns that contain the specific text and are not in exclude_cols
        cols_to_sum = [col for col in df.columns if text in col and col not in exclude_cols]

        # Add a new column with the sum of the specified columns
        df[f'sum_of_{text}'] = df[cols_to_sum].sum(axis=1)

        # Update the dictionary with the modified dataframe
        dict_of_dfs[key] = df
    return dict_of_dfs


def drop_columns_with_keywords(dict_of_dfs, keywords):
    """
    Drops columns from each dataframe in the dictionary that contain any of the specified keywords in the column name.

    Parameters:
    dict_of_dfs (dict): Dictionary where keys are names and values are dataframes.
    keywords (list): List of keywords to search for in the column names.
    """
    for key, df in dict_of_dfs.items():
        # Find columns that contain any of the specified keywords
        cols_to_drop = [col for col in df.columns if any(keyword in col for keyword in keywords)]

        # Drop the columns
        df.drop(columns=cols_to_drop, inplace=True)

        # Update the dictionary with the modified dataframe
        dict_of_dfs[key] = df


def aggregate_to_hourly(dict_of_dfs, datetime_column=None, verbose=False):
    aggregated_dfs = {}

    for name, df in dict_of_dfs.items():
        # Set the specified column as the datetime index if provided
        if datetime_column:
            if datetime_column in df.columns:
                df = df.set_index(datetime_column)
            else:
                raise ValueError(f"DataFrame {name} does not have the specified datetime column: {datetime_column}")

        # Ensure the dataframe index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame {name} does not have a DatetimeIndex.")

        # Identify numerical and non-numerical columns
        non_numerical_cols = ['DataFrame Name', 'Time', 'is_weekend', 'hour_sin', 'hour_cos', 'day_of_month_sin',
                              'day_of_month_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
        existing_non_numerical_cols = [col for col in non_numerical_cols if col in df.columns]
        numerical_cols = df.select_dtypes(include=['number']).columns.difference(existing_non_numerical_cols)

        # Create a dictionary to store aggregated columns
        agg_dict = {}
        for col in numerical_cols:
            if '(kWh)' in col:
                agg_dict[col] = 'sum'
            elif '(kW)' in col:
                agg_dict[col] = 'mean'
            elif '(kVa)' in col:
                agg_dict[col] = 'mean'
            elif 'Power Factor' in col:
                agg_dict[col] = 'mean'
            elif '(m3)' in col and 'm3/h' not in col:
                agg_dict[col] = 'sum'
            elif '(m3/h)' in col:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'mean'

        # Resample and aggregate the numerical columns
        resampled_numerical_df = df[numerical_cols].resample('h').agg(agg_dict)

        # Downsample the non-numerical columns by taking the first value in each hour
        resampled_non_numerical_df = df[existing_non_numerical_cols].resample('h').first()

        # Combine the numerical and non-numerical dataframes
        resampled_df = pd.concat([resampled_numerical_df, resampled_non_numerical_df], axis=1)

        # Reset index to get the datetime column back
        resampled_df.reset_index(inplace=True)

        # Ensure the datetime column is the first column
        resampled_df.index.name = datetime_column
        datetime_col = resampled_df.pop(resampled_df.index.name)
        resampled_df.insert(0, datetime_column, datetime_col)

        # Store the aggregated dataframe in the dictionary
        aggregated_dfs[name] = resampled_df

        if verbose:
            print(f'DataFrame {name} aggregated to hourly intervals.')

    return aggregated_dfs


def add_energy_cost_column(dict_of_dfs, price_col_mwh, sum_col_kwh):
    """
    Adds an energy_cost_kWh column to each dataframe in the dictionary by multiplying
    sum_kWh by price_mWh (converted to kWh).

    Parameters:
    dict_of_dfs (dict): Dictionary where keys are names and values are dataframes.
    """
    for key, df in dict_of_dfs.items():
        if sum_col_kwh in df.columns and price_col_mwh in df.columns:
            # Calculate the energy cost in kWh
            df['energy_cost_kWh'] = df[sum_col_kwh] * (df[price_col_mwh] / 1000)

            # Update the dictionary with the modified dataframe
            dict_of_dfs[key] = df
        else:
            print(f"Columns {sum_col_kwh} or {price_col_mwh} not found in dataframe {key}.")


def save_dataframes_to_csv(dataframes_dict, output_directory, verbose=False):
    """
    Save each dataframe in a dictionary to a CSV file in the specified output directory with the current date and time.

    Parameters:
    - dataframes_dict (dict): A dictionary where keys are dataframe identifiers and values are pandas dataframes.
    - output_directory (str): Path to the directory where the processed dataframes will be saved.
    - verbose (bool): If True, print debugging information.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get the current date and time in ISO format
    current_datetime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Save each dataframe in the dictionary to a CSV file
    for key, df in dataframes_dict.items():
        output_file_path = os.path.join(output_directory, f"{current_datetime}_{key}_processed.csv")
        df.to_csv(output_file_path, index=False)

        # Print the saved file path to debug if verbose is True
        if verbose:
            print(f"Dataframe '{key}' saved to '{output_file_path}'")
