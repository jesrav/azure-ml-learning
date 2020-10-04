from pathlib import Path

import mlflow.azureml
from azureml.core import Workspace
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

ws = Workspace.from_config()

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

model_path = Path("learning-scripts/06_azure_mlflow/output/add_n_model")

experiment_name = "experiment-with-mlflow"
mlflow.set_experiment(experiment_name)

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python=3.8",
        "scikit-learn=0.19.1" "pip",
        {
            "pip": [
                "mlflow",
                "azureml-mlflow",
            ],
        },
    ],
    "name": "mlflow-example",
}

# Define the model class
class AddN(mlflow.pyfunc.PythonModel):
    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)


with mlflow.start_run():
    add5_model = AddN(n=5)
    mlflow.pyfunc.log_model(
        registered_model_name="mlflow_model",
        python_model=add5_model,
        conda_env=conda_env,
        artifact_path="model",
    )


import pandas as pd

model = mlflow.pyfunc.load_model(
    "azureml://experiments/experiment-with-mlflow/runs/f3863733-c09b-42d3-9e38-6cbf7e239345/artifacts/model"
)
model.predict(pd.DataFrame([4]))
