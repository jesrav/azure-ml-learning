# Import libraries
from azureml.core import Run
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model_config import model


def create_age_group_variable(df):
    df = df.copy()
    df["age_group"] = np.where(df.Age > 30, "over30", "under30")
    return df


# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
diabetes = run.input_datasets["diabetes"].to_pandas_dataframe()

diabetes = create_age_group_variable(diabetes)

# Split data into training set and test set
train, test = train_test_split(diabetes, test_size=0.30, random_state=0)

# Train
model = model.fit(train, train.Diabetic)

# calculate accuracy
y_hat = model.predict(test)
acc = accuracy_score(y_hat.astype(int), test.Diabetic)
print("Accuracy:", acc)
run.log("Accuracy", np.float(acc))

# Save the trained model in the outputs folder
os.makedirs("outputs", exist_ok=True)
model.save(fname="outputs/diabetes_model.pkl")

run.complete()
