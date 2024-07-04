
def add_total_consumption_columns_and_by_zone(dict_of_dfs, verbose=False):
    """
    Adds total consumption columns (total_consumption(kW) and total consumption by zone) to each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames to add the total consumption columns.
    verbose (bool): If True, print detailed information about the operations. Default is False.

    Returns:
    dict: Dictionary of DataFrames with the total consumption columns added.
    """
    zone_pattern = re.compile(r'(z\d+)_.*\(kW\)')

    for name, df in dict_of_dfs.items():
        # Identify columns with 'kW' in their name
        kw_columns = [col for col in df.columns if 'kW' in col]

        # Calculate the total consumption, treating NaNs as zeros
        df['total_consumption(kW)'] = df[kw_columns].fillna(0).sum(axis=1)

        # Identify zones and calculate total consumption for each zone
        zones = {}
        for col in kw_columns:
            match = zone_pattern.match(col)
            if match:
                zone = match.group(1)
                if zone not in zones:
                    zones[zone] = []
                zones[zone].append(col)

        for zone, columns in zones.items():
            df[f'{zone}_total_consumption(kW)'] = df[columns].fillna(
                0).sum(axis=1)

        # Update the dataframe in the dictionary
        dict_of_dfs[name] = df

        if verbose:
            print(
                f"DataFrame: {name} - Added total_consumption(kW) and zone-specific total consumption columns")

    return dict_of_dfs


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

def calculate_total_ac_consumption_by_zone(dict_of_dfs):
    """
    Calculate total AC consumption by zone for each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames with AC consumption data.

    Returns:
    dict: Dictionary of DataFrames with total AC consumption by zone added.
    """
    ac_zone_pattern = re.compile(r'(z\d+).*AC')

    for name, df in dict_of_dfs.items():
        # Find all columns that match the pattern
        ac_columns = [col for col in df.columns if ac_zone_pattern.search(col)]

        # Extract unique zones from the column names
        zones = set(ac_zone_pattern.search(col).group(1)
                    for col in ac_columns if ac_zone_pattern.search(col))

        for zone in zones:
            # Find all columns for the current zone
            zone_columns = [col for col in ac_columns if zone in col]
            if zone_columns:  # Only add the total column if there are relevant columns
                # Calculate the total consumption for the current zone
                df[f'{zone}_total_AC_consumption(kW)'] = df[zone_columns].sum(
                    axis=1)

        dict_of_dfs[name] = df

    return dict_of_dfs

def check_if_total_active_energy(df, suspect_column, energy_columns, verbose=False):
    """
    Check if the suspect column represents the total active energy by comparing it with the sum of other energy columns.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the columns.
    - suspect_column (str): The name of the suspect total active energy column.
    - energy_columns (list of str): A list of column names representing individual energy usage.
    - verbose (bool): If True, print detailed comparison results.

    Returns:
    - bool: True if the suspect column matches the sum of other energy columns, False otherwise.
    """

    # Check if the suspect column and all energy columns exist in the DataFrame
    missing_columns = [col for col in [suspect_column] + energy_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found in the DataFrame: {missing_columns}")

    # Sum the individual energy columns
    df['Sum_Energy_Columns'] = df[energy_columns].sum(axis=1)

    # Compare the suspect column with the sum of other energy columns
    columns_match = df[suspect_column].equals(df['Sum_Energy_Columns'])

    if verbose:
        if columns_match:
            print(f"The suspect column '{suspect_column}' matches the sum of the energy columns: {energy_columns}.")
        else:
            print(f"The suspect column '{suspect_column}' does not match the sum of the energy columns: {energy_columns}.")
            # Provide additional info on mismatched rows
            mismatched_rows = df[df[suspect_column] != df['Sum_Energy_Columns']]
            print(f"Mismatched rows:\n{mismatched_rows[[suspect_column] + ['Sum_Energy_Columns']].head()}")

    # Drop the temporary sum column
    df.drop(columns=['Sum_Energy_Columns'], inplace=True)

    return columns_match