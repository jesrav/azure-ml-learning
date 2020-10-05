from pathlib import Path

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core import Workspace, Model, Environment

ws = Workspace.from_config()

model = ws.models["diabetes_model"]
environment = Environment.get(workspace=ws, name="training_environment")

service_name = "diabetes-service"
scoring_directory = Path("learning-scripts/05_deploy_web_service")

inference_config = InferenceConfig(
    environment=environment,
    source_directory=scoring_directory,
    entry_script="score.py",
)

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
