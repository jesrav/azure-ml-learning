from pathlib import Path
from azureml.core import Workspace, Dataset

CSV_PATH = Path("data/diabetes.csv")

ws = Workspace.from_config()

blob_ds = ws.get_default_datastore()

csv_paths = [(blob_ds, str(CSV_PATH))]

# Create tabular data set with the diabetes data
diabetes_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)

# Register the dataset in the data
diabetes_ds = diabetes_ds.register(workspace=ws, name="diabetes")

# Delete the dataset(only a reference) and get it by name and version number
del diabetes_ds
diabetes_ds = Dataset.get_by_name(workspace=ws, name="diabetes", version=1)

# Get some data from the dataset
print(diabetes_ds.take(3).to_pandas_dataframe())
