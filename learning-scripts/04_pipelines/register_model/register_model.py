import argparse
from pathlib import Path

import joblib
from azureml.core import Model, Run


def main(model_folder):
    run = Run.get_context()

    # load the model
    print("Loading model from " + model_folder)
    model_file = Path("model_folder") / Path("model.pkl")
    model = joblib.load(model_file)

    Model.register(
        workspace=run.experiment.workspace,
        model_path=model_file,
        model_name="pipeline_diabetes_model",
        tags={"Training context": "Pipeline"},
    )

    run.complete()

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_folder",
            type=str,
            dest="model_folder",
            default="diabetes_model",
            help="model location",
        )
        args = parser.parse_args()
        main(args.model_folder)
