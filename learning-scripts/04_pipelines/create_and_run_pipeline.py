from pathlib import Path
from azureml.core import Dataset, Workspace, Environment, Experiment
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.core.runconfig import RunConfiguration
from azureml.train.estimator import Estimator

ws = Workspace.from_config()

compute_name = "aml-cluster"

training_env = Environment.get(workspace=ws, name="training_environment")

# Get the diabetes dataset
raw_ds = Dataset.get_by_name(ws, "diabetes")

data_store = ws.get_default_datastore()

# create a new run config object. The same run config is used in both
# pipeline steps, but you could use a separate one for each.
run_config = RunConfiguration()
run_config.environment = training_env
run_config.target = compute_name

# Paths top scripts
preprocess_step_directory = Path("learning-scripts/04_pipelines/preprocess")
train_step_directory = Path("learning-scripts/04_pipelines/train")
register_step_directory = Path("learning-scripts/04_pipelines/register_model")


# Define a PipelineData object for the preprocessed data
preprocessed_folder = PipelineData("preprocessed_folder", datastore=data_store)

# Define a PipelineData object for the model artifact
model_folder = PipelineData("model_folder", datastore=data_store)

# Step to run a Python script
preprocess_step = PythonScriptStep(
    name="Preprocess data",
    source_directory=str(preprocess_step_directory),
    script_name="preprocess.py",
    compute_target=compute_name,
    runconfig=run_config,
    inputs=[raw_ds.as_named_input("diabetes")],
    outputs=[preprocessed_folder],
    arguments=["--output_folder", preprocessed_folder],
    allow_reuse=True,
)

train_step = PythonScriptStep(
    name="train model",
    script_name="train.py",
    arguments=["--input_folder", preprocessed_folder, "--output_folder", model_folder],
    inputs=[preprocessed_folder],
    outputs=[model_folder],
    source_directory=train_step_directory,
    runconfig=run_config,
    allow_reuse=True,
)

register_step = PythonScriptStep(
    name="Register Model",
    source_directory=register_step_directory,
    script_name="register_model.py",
    arguments=["--model_folder", model_folder],
    inputs=[model_folder],
    runconfig=run_config,
    allow_reuse=True,
)


# Construct the pipeline
train_pipeline = Pipeline(
    workspace=ws, steps=[preprocess_step, train_step, register_step]
)

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name="training-pipeline")
pipeline_run = experiment.submit(train_pipeline)
