"#!/usr/bin/env python3"
import sys
import pandas as pd


def extract_target_values(df, column_name, relevant_target_values):
    """
    :param df: The pandas dataframe to be extracted from.
    :param column_name: The name of the column to filter values by.
    :param relevant_target_values: The relevant values to include in the
           filtered dataset.

    :return: A pandas dataframe containing only rows where the `column_name`
             matches any one of the provided `relevant_target_values`.
    """
    return df.loc[df[column_name].isin(relevant_target_values)]


def filter_dataset(raw_data_path):
    """
    filters the provided raw dataset.
    :param raw_data_path: path to the raw, unprocessed dataset.
    :return: a filtered pandas dataframe.
    """
    raw_dataset_filepath = "data/unprocessed-crime-data.csv"  # raw, unprocessed dataset location.
    raw_df = pd.read_csv(raw_dataset_filepath)  # loading the raw data.
    column_name = "Primary Type"  # the column to filter on
    relevant_target_values = ["BATTERY", "THEFT", "CRIMINAL DAMAGE"]  # can be changed for experimentation
    df = extract_target_values(raw_df, column_name, relevant_target_values)  # filtering the raw dataset

    # store the filtered dataset (not necessary, but useful  JiC).
    df.to_csv(index=False, path_or_buf="data/filtered-crime-data.csv")

    return df  # returning the filtered dataframe


def main():
    # Keeping name upper-cased to let the world know this var shouldn't be changed
    RAW_DATA_PATH = "data/filtered-crime-data.csv"

    df = filter_dataset(RAW_DATA_PATH)

    print(df.describe())
    print(len(df))


if __name__ == "__main__":
    main()
