import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow

#setting the tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("green-taxi-experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        mlflow.set_tag("developer", "isabelle")
        mlflow.log_param("train-data-path", "./output/train.pkl")
        mlflow.log_param("valid-data-path", "./output/val.pkl")

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mlflow.log_param("model", rf)

        mlflow.log_param("max_depth", 10)
        min_samples_split = rf.get_params()['min_samples_split']
        mlflow.log_param("min_samples_split", min_samples_split)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)


        model_path = "models/rf_model.pkl"
        with open(model_path, "wb") as f_out:
            pickle.dump(rf, f_out)

        mlflow.log_artifact(local_path=model_path, artifact_path="models_pickle")


if __name__ == '__main__':
    
    run_train()
