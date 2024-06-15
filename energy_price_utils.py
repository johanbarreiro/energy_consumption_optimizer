import datetime as dt

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