import subprocess


def test_score_model():
    """
    Test to check the model is trained first and
    then perform scoring on the validation data
    """
    subprocess.run(
        [
            "python3",
            "src/nonstandardcode/train.py",
            "--log-level",
            "DEBUG",
            "--log-path",
            "logs/train2.log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    result = subprocess.run(
        [
            "python3",
            "src/nonstandardcode/score.py",
            "--model-path",
            "model/random_forest_model.pkl",
            "--test-path",
            "data/processed/val.csv",
            "--log-path",
            "logs/score.log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, f"Scoring script failed: {result.stderr.decode()}"
