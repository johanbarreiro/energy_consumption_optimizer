import os
import re
import pandas as pd
from datetime import datetime
import argparse

def extract_data_from_content(filepath):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('MARGINALPDBC;'):
                continue  # Skip the header line
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            if line.endswith('*'):
                line = line[:-1]  # Remove the trailing asterisk if present
            if line:
                parts = line.split(';')
                if len(parts) == 7:
                    year, month, day, hour, value1, value2, space = parts
                    if hour == '25': # Skip the last hour of the day on dst end - needs to be fixed
                        continue
                    # Adjusting date to include hour
                    date = datetime(int(year), int(month), int(day), int(hour)-1) # subtract 1 to obtain hour range 0-23
                    data.append([date, float(value1), float(value2)])  # Adjusting to 3 columns
    return data

def main(directory, verbose=False):
    parent_directory = os.path.abspath(os.path.join(directory, os.pardir))
    output_directory = os.path.join(parent_directory, 'spain_daily_market_hourly_prices')
    os.makedirs(output_directory, exist_ok=True)

    filename_pattern = re.compile(r'marginalpdbc_(\d{8})\.\d')
    files = os.listdir(directory)

    all_data = []
    for file in files:
        filename_match = filename_pattern.match(file)
        if filename_match:
            filepath = os.path.join(directory, file)
            data = extract_data_from_content(filepath)
            if data:
                all_data.extend(data)
            if verbose:
                print(f"Processed: {file}")

    columns = ['Date', 'Price1', 'Price2']  # Adjusted columns
    df = pd.DataFrame(all_data, columns=columns)

    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year

    for year, group in df.groupby('Year'):
        output_file = os.path.join(output_directory, f'marginalpdbc_{year}.csv')
        try:
            group.to_csv(output_file, index=False)
            if verbose:
                print(f"Saved: {output_file}")
        except Exception as e:
            print(f"Error saving {output_file}: {str(e)}")

    if verbose:
        print(f"Data has been successfully extracted and saved to CSV files in {output_directory}.")
    else:
        print("Data has been successfully extracted and saved to CSV files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from marginalpdbc files and save to CSV by year.")
    parser.add_argument('directory', type=str, help='Path to the directory containing the files.')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output.')
    args = parser.parse_args()
    main(args.directory, verbose=args.verbose)