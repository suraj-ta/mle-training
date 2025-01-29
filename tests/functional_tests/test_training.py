import os
import subprocess


def test_train_model():
    """
    Test to check the model is trained and saved in the provided directory.
    """
    result = subprocess.run(
        [
            "python3",
            "src/nonstandardcode/train.py",
            "--train-path",
            "data/processed/train.csv",
            "--val-path",
            "data/processed/val.csv",
            "--log-level",
            "INFO",
            "--log-path",
            "logs/train.log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, f"Training script failed: {result.stderr.decode()}"
    assert os.path.exists("model/random_forest_model.pkl"), "Model was not saved"
