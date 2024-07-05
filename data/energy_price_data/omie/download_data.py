import datetime as dt
import matplotlib.pyplot as plt

from OMIEData.DataImport.omie_marginalprice_importer import OMIEMarginalPriceFileImporter
from OMIEData.Enums.all_enums import DataTypeInMarginalPriceFile

# Define the start and end dates
dateIni = dt.datetime(2020, 1, 1)
dateEnd = dt.datetime(2020, 3, 22)

# Download the data and convert to dataframe
print("Downloading data, this may take some time...")
omie_object = OMIEMarginalPriceFileImporter(date_ini=dateIni, date_end=dateEnd) # .read_to_dataframe(verbose=True)

omie_object.read_to_dataframe(verbose=True)

# Sort the dataframe by date
df.sort_values(by='DATE', axis=0, inplace=True)

# Print the dataframe
print(df)

# Plot the data if necessary
plt.figure(figsize=(10, 6))
plt.plot(df['DATE'], df['PRICE'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('OMIE Marginal Prices')
plt.show()