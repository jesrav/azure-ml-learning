from tqdm import tqdm
from azureml.core import Workspace, Experiment
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib
import math

##################################################################
# Setup
##################################################################
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="diabetes-experiment")


##################################################################
# get data
##################################################################
# load the diabetes dataset
print("Loading diabetes data")
diabetes = pd.read_csv("data/diabetes.csv")
X, y = (
    diabetes[
        [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age",
        ]
    ].values,
    diabetes["Diabetic"].values,
)

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)


##################################################################
# Train multiple models, logging metrics of each model training in
# a separate run.
##################################################################
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for alpha in alphas:
    run = experiment.start_logging()
    run.log("alpha_value", alpha)

    model = Ridge(alpha=alpha)
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)
    rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    run.log("rmse", rmse)

    model_name = "model_alpha_" + str(alpha) + ".pkl"
    filename = "outputs/" + model_name

    joblib.dump(value=model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)
    run.complete()

##################################################################
# Get best model from tracked experiments
##################################################################
runs = {}
run_metrics = {}

# Create dictionaries containing the runs and the metrics for all
# runs containing the 'mse' metric
for r in tqdm(experiment.get_runs()):
    metrics = r.get_metrics()
    if "rmse" in metrics.keys():
        runs[r.id] = r
        run_metrics[r.id] = metrics

# Find the run with the best (lowest) mean squared error and display the id and metrics
best_run_id = min(run_metrics, key=lambda k: run_metrics[k]["rmse"])
best_run = runs[best_run_id]
print("Best run is:", best_run_id)
print("Metrics:", run_metrics[best_run_id])

# Tag the best run for identification later
best_run.tag("Best Run")

# View the files in the run
for f in best_run.get_file_names():
    print(f)

# Register the model with the workspace
model = best_run.register_model(
    model_name="best_model", model_path="model_alpha_0.1.pkl"
)
