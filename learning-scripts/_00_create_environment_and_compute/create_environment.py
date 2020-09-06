from pathlib import Path
from azureml.core import Workspace
from azureml.core import Environment

CONDA_YAML_PATH = Path("learning-scripts/_00_create_environment_and_compute/conda.yaml")
ws = Workspace.from_config()

env = Environment.from_conda_specification(
    name="training_environment", file_path=CONDA_YAML_PATH
)

env.register(workspace=ws)
