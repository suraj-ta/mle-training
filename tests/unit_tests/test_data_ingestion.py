import os
import subprocess


def test_ingest_data():
    """
    Test to check if the processed dataset files are getting created.
    """
    result = subprocess.run(
        [
            "python3",
            "src/nonstandardcode/ingest_data.py",
            "--output-dir",
            "data/processed",
            "--log-level",
            "INFO",
            "--log-path",
            "logs/ingest_data.log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, f"Ingestion script failed: {result.stderr.decode()}"
    assert os.path.exists("data/processed/train.csv"), "train.csv was not created"
    assert os.path.exists("data/processed/val.csv"), "val.csv was not created"
