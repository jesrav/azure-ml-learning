import json
import joblib
import numpy as np
from azureml.core.model import Model
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.schema_decorators import input_schema, output_schema
import pandas as pd


def init():
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path("diabetes_model")
    model = joblib.load(model_path)


pandas_input_data = {
    "Pregnancies": ["2", "3"],
    "PlasmaGlucose": [180, 180],
    "DiastolicBloodPressure": [74, 77],
    "TricepsThickness": [24, 44],
    "SerumInsulin": [23.9091702, 44.33],
    "BMI": [1.488172308, 1.388172308],
    "DiabetesPedigree": 1.488172308,
    "Age": [33, 22],
}
sample_input = pd.DataFrame(data=pandas_input_data)
sample_output = np.array([0, 1])

# input_sample = np.array([[2, 180, 74, 24, 21, 23.9091702, 1.488172308, 22]])
# output_sample = np.array([1])


@input_schema("data", PandasParameterType(sample_input))
@output_schema(NumpyParameterType(sample_output))
# @input_schema('data', NumpyParameterType(input_sample))
# @output_schema(NumpyParameterType(output_sample))
def run(data):
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()
