# Description: This module contains utility functions for the optimization process.

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
