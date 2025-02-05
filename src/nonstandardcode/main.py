import mlflow
from ingest_data import main as ingesting_data
from score import main as scoring_model
from train import main as training_model

Experiment_name = "Housing_Prices"
mlflow.set_experiment(Experiment_name)
mlflow.set_tracking_uri("http://localhost:5000")


def main():
    with mlflow.start_run(run_name="Housing_Prices_Workflow"):

        with mlflow.start_run(run_name="Data Ingestion", nested=True):
            ingesting_data()
            mlflow.log_artifacts("data/processed")
        with mlflow.start_run(run_name="Model Training", nested=True):
            training_model()
            mlflow.log_artifacts("model")
        with mlflow.start_run(run_name="Model Scoring", nested=True):
            scoring_model()


if __name__ == "__main__":
    main()
