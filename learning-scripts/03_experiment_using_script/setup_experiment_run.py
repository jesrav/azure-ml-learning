import azureml.core
from azureml.core import Workspace, Experiment

# Load the workspace from the saved config file
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="diabetes-experiment")
