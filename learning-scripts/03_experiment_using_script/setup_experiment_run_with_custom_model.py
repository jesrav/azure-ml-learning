from pathlib import Path
from azureml.core import Environment, Experiment, Workspace
from azureml.train.estimator import Estimator

ws = Workspace.from_config()

compute_name = "aml-cluster"

training_env = Environment.get(workspace=ws, name="training_environment")

experiment_folder = Path(
    "learning-scripts/03_experiment_using_script/train_custom_model"
)
entry_script = "train_custom_model.py"

estimator = Estimator(
    source_directory=experiment_folder,
    entry_script=entry_script,
    inputs=[ws.datasets["diabetes"].as_named_input("diabetes")],
    environment_definition=training_env,
    compute_target=compute_name,
)

experiment = Experiment(workspace=ws, name="diabetes-experiment")
run = experiment.submit(config=estimator)
