import argparse
import logging
import os
import tarfile
import urllib.request

import pandas as pd
from six.moves import urllib


def logging_setup(log_level="INFO", log_path=None, console_log=True):
    """
    Configures logging for the script.

    Parameters
    ----------
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".
    log_path : str, optional
        Path to the log file. If None, logs won't be written to a file.
    console_log : bool, optional
        Whether to log messages to the console. Default is True.
    """
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    log_level = log_level.upper()
    if log_level not in valid_log_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Choose from {valid_log_levels}"
        )

    handlers = []
    if console_log:
        handlers.append(logging.StreamHandler())
    if log_path:
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def fetch_housing_data(housing_url, housing_path):
    """
    Downloads and extracts the housing data.

    Parameters
    ----------
    housing_url : str
        URL of the housing data.
    housing_path : str
        The Directory where the raw housing data will be saved and extracted.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    logging.info(f"Starting download of housing data from {housing_url}...")

    logging.info("Downloading and Extracting the housing data...")

    with tarfile.open(tgz_path, "r:gz") as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    logging.info(f"Successfully extracted the Housing data to {housing_path}.")


def split_data(housing_path, output_dir):
    """
    Splits the dataset into training and validation sets.

    Parameters
    ----------
    housing_path : str
        Path to the directory containing the raw housing data.
    output_dir : str
        Directory where the processed data will be saved.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    logging.info(f"Reading the housing dataset from {csv_path}")
    housing_df = pd.read_csv(csv_path)

    logging.info("Splitting the dataset into training and validation sets")
    train_df = housing_df.sample(frac=0.8, random_state=42)
    validation_df = housing_df.drop(train_df.index)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    validation_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    logging.info(f"Training and Validation datasets saved to directory: {output_dir}.")


def main():
    """
    Function to define the parser, add arguments and make function calls
    """
    parser = argparse.ArgumentParser(description="Preparing the Housing data.")
    parser.add_argument(
        "--housing-url",
        default="https://raw.githubusercontent.com/ageron/handson-ml/master/"
        + "datasets/housing/housing.tgz",
        help="URL of the housing data (default: Housing Dataset URL)",
    )
    parser.add_argument(
        "--housing-path",
        default="data/raw/",
        help="The path where to save the raw housing dataset (default: data/raw/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="The path where to save the processed datasets (default: data/processed/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-path",
        default="logs/ingest_data.log",
        help="The path where to save the log file (default: None)",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="To disable console logging",
    )

    args = parser.parse_args()

    if args.log_path:
        log_dir = os.path.dirname(args.log_path)
        os.makedirs(log_dir, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)

    console_log = not args.no_console_log
    logging_setup(args.log_level, args.log_path, console_log)

    logging.info("Starting the data ingestion process.")
    fetch_housing_data(args.housing_url, args.housing_path)
    split_data(args.housing_path, args.output_dir)
    logging.info("Data ingestion and processing completed successfully.")


if __name__ == "__main__":
    main()
