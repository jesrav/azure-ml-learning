import argparse
import os

from pathlib import Path

import pandas as pd
import numpy as np
from azureml.core import Run


def main(output_folder):

    # Get the experiment run context
    run = Run.get_context()

    # load the diabetes dataset
    df: pd.DataFrame = run.input_datasets["diabetes"].to_pandas_dataframe()

    # Add random noise
    df["random_noise"] = np.random.rand(df.shape[0])

    # Save the trained model
    os.makedirs(Path(output_folder), exist_ok=True)
    output_path = Path(output_folder) / Path("preprocessed_diabetes.pkl")
    df.to_pickle(output_path)

    run.complete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        default="preprocessed_diabetes",
        help="output folder",
    )
    args = parser.parse_args()
    main(args.output_folder)
