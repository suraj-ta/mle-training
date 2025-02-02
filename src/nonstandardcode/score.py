import argparse
import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error


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


def preprocess_data(data, target_column=None):
    """
    Preprocesses the dataset for prediction.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to preprocess.
    target_column : str, optional
        The target column to drop from features (default: None).

    Returns
    -------
    pd.DataFrame or tuple
        Preprocessed features (and target if target_column is provided).
    """
    logging.info("Preprocessing the test dataset")
    categorical_columns = ["ocean_proximity"]
    data = pd.get_dummies(data, columns=categorical_columns)
    data = data.copy()

    if target_column and target_column in data.columns:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    return data


def score_model(model_path, test_path, target_column):
    """
    Scores the trained model on the test dataset.

    Parameters
    ----------
    model_path : str
        Path to the trained model.
    test_path : str
        Path to the test dataset.
    target_column : str
        The target column to predict.
    """
    logging.info(f"Loading the test dataset from {test_path}.")
    test_data = pd.read_csv(test_path)

    X_test, y_test = preprocess_data(test_data, target_column)

    logging.info(f"Loading the trained model from {model_path}.")
    model = joblib.load(model_path)

    logging.info("Making the predictions.")
    predictions = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, predictions)
    logging.info(f"Test RMSE: {rmse:.4f}")


def main():
    """
    Function to create parser, add arguments and make function call to score_model.
    """
    parser = argparse.ArgumentParser(
        description="Scoring a trained model on the test dataset."
    )
    parser.add_argument(
        "--model-path",
        default="model/random_forest_model.pkl",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--test-path",
        default="data/processed/val.csv",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--target-column",
        default="median_house_value",
        help="The target column in the test dataset.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--log-path", default="logs/score.log")
    parser.add_argument("--no-console-log", action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logging_setup(args.log_level, args.log_path, not args.no_console_log)

    score_model(args.model_path, args.test_path, args.target_column)


if __name__ == "__main__":
    main()
