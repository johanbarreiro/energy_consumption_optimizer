import pandas as pd
import re
import sympy as sp
from scipy.optimize import least_squares

def define_total_floor_consumption(df):
    """
    Define the equation for total floor consumption.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to define the constraint.

    Returns:
    sympy.Expr: The equation for total floor consumption.
    """
    # Filter out columns that represent totals
    consumption_columns = [col for col in df.columns if ('kW' in col or 'kWh' in col) and 'total' not in col.lower()]
    total_consumption = sp.Add(*[sp.symbols(col) for col in consumption_columns])
    return total_consumption

def define_total_zone_consumption(df):
    """
    Define the equations for total consumption for each zone.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to define the constraints.

    Returns:
    dict: A dictionary of equations for total consumption by zone.
    """
    # Define a pattern to match zone columns
    zone_pattern = re.compile(r'(Z\d+)', re.IGNORECASE)
    
    # Filter columns to include only those that match the zone pattern and do not contain 'total'
    zone_columns = [col for col in df.columns if zone_pattern.search(col) and 'total' not in col.lower()]
    
    # Extract unique zones from the filtered column names
    zones = set(zone_pattern.search(col).group(1) for col in zone_columns)

    # Define equations for total consumption by zone
    zone_consumptions = {}
    for zone in zones:
        zone_cols = [col for col in zone_columns if zone in col]
        zone_consumptions[zone] = sp.Add(*[sp.symbols(col) for col in zone_cols])
    
    return zone_consumptions


def define_total_ac_consumption_by_zone(df):
    """
    Define the equations for total AC consumption by zone.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to define the constraints.

    Returns:
    dict: A dictionary of equations for total AC consumption by zone.
    """
    ac_zone_pattern = re.compile(r'(Z\d+).*AC')
    ac_columns = [col for col in df.columns if ac_zone_pattern.search(col)]
    zones = set(ac_zone_pattern.search(col).group(1) for col in ac_columns)

    ac_consumptions = {}
    for zone in zones:
        zone_ac_cols = [col for col in ac_columns if zone in col]
        ac_consumptions[zone] = sp.Add(*[sp.symbols(col) for col in zone_ac_cols])
    
    return ac_consumptions

def store_constraints(dict_of_dfs):
    """
    Store all the constraint equations for each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.

    Returns:
    dict: Dictionary with constraint equations for each DataFrame.
    """
    constraints = {}
    for name, df in dict_of_dfs.items():
        constraints[name] = {
            'total_floor_consumption': define_total_floor_consumption(df),
            'total_zone_consumption': define_total_zone_consumption(df),
            'total_ac_consumption_by_zone': define_total_ac_consumption_by_zone(df)
        }
    
    return constraints

def extract_data_for_fitting(df, consumption_columns):
    data = df[consumption_columns].values
    return data

def fit_coefficients(df, equation, consumption_columns):
    data = extract_data_for_fitting(df, consumption_columns)
    def objective_function(coefficients):
        subs = {sp.symbols(col): coef for col, coef in zip(consumption_columns, coefficients)}
        equation_with_coefficients = equation.subs(subs)
        evaluated = np.array([float(equation_with_coefficients.evalf(subs={sp.symbols(col): val for col, val in zip(consumption_columns, row)})) for row in data])
        residuals = evaluated - data.sum(axis=1)
        return residuals
    initial_guess = np.ones(len(consumption_columns))
    result = least_squares(objective_function, initial_guess)
    return result.x

def fit_constraints_for_all(dict_of_dfs, constraints):
    """
    Fit coefficients to the constraint equations for each DataFrame in the dictionary.

    Parameters:
    dict_of_dfs (dict): Dictionary of DataFrames.
    constraints (dict): Dictionary with constraint equations for each DataFrame.

    Returns:
    dict: Dictionary with fitted coefficients for each constraint in each DataFrame.
    """
    fitted_coefficients = {}
    for name, df in dict_of_dfs.items():
        consumption_columns = [col for col in df.columns if ('kW' in col or 'kWh' in col) and 'total' not in col.lower()]
        fitted_coefficients[name] = {}
        
        # Fit coefficients for total floor consumption
        total_floor_consumption = constraints[name]['total_floor_consumption']
        fitted_coefficients[name]['total_floor_consumption'] = fit_coefficients(df, total_floor_consumption, consumption_columns)
        
        # Fit coefficients for total zone consumption
        total_zone_consumption = constraints[name]['total_zone_consumption']
        fitted_coefficients[name]['total_zone_consumption'] = {}
        for zone, equation in total_zone_consumption.items():
            fitted_coefficients[name]['total_zone_consumption'][zone] = fit_coefficients(df, equation, consumption_columns)
        
        # Fit coefficients for total AC consumption by zone
        total_ac_consumption_by_zone = constraints[name]['total_ac_consumption_by_zone']
        fitted_coefficients[name]['total_ac_consumption_by_zone'] = {}
        for zone, equation in total_ac_consumption_by_zone.items():
            fitted_coefficients[name]['total_ac_consumption_by_zone'][zone] = fit_coefficients(df, equation, consumption_columns)
    
    return fitted_coefficients