import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import os
from azureml.core import Run
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Get the experiment run context
run = Run.get_context()


def main(input_folder, output_folder):

    # Load data
    df = pd.read_pickle(Path(input_folder) / Path("preprocessed_diabetes.pkl"))

    # Separate features and labels
    X, y = (
        df[
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
        df["Diabetic"].values,
    )

    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )

    # Set regularization hyperparameter
    reg = 0.01

    # Train a logistic regression model
    print("Training a logistic regression model with regularization rate of", reg)
    run.log("Regularization Rate", np.float(reg))
    model = LogisticRegression(C=1 / reg, solver="liblinear").fit(X_train, y_train)

    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print("Accuracy:", acc)
    run.log("Accuracy", np.float(acc))

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print("AUC: " + str(auc))
    run.log("AUC", np.float(auc))

    # Save the trained model in the outputs folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = Path(output_folder) / Path("model.pkl")
    joblib.dump(value=model, filename=output_path)

    run.complete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        dest="input_folder",
        help="Folder for trained model artifact.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        default="diabetes_model",
        help="output folder",
    )
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
