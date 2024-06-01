import argparse
import os
import pandas as pd


def get_dataframes(directory: str) -> dict:
    dataframes = {}

    # Debug: Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return dataframes

    # Debug: Print the directory being processed
    print(f"Processing directory: {directory}")

    for filename in os.listdir(directory):
        # Debug: Print each filename being processed
        print(f"Found file: {filename}")

        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            # Debug: Print the full filepath
            print(f"Reading CSV file: {filepath}")

            df = pd.read_csv(filepath)
            dataframes[os.path.splitext(filename)[0]] = df

            print(f'DataFrame for {filename} created from {filepath}.')

    return dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load CSV files into dataframes.")
    parser.add_argument('-d', '--directory', type=str,
                        help='The directory containing CSV files.')

    args = parser.parse_args()

    if args.directory:
        directory = args.directory
    else:
        directory = input("Please enter the directory containing CSV files: ")

    dataframes = get_dataframes(directory)
    print(f"Total dataframes created: {len(dataframes)}")
