import mlflow
from azureml.core import Workspace

ws = Workspace.from_config()

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

experiment_name = "experiment_with_mlflow"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_metric("alpha", 0.03)
