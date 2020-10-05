import json
import pandas as pd
from azureml.core import Workspace

ws = Workspace.from_config()

service = ws.webservices["diabetes-service"]

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
input_dict = {"data": pd.DataFrame(data=pandas_input_data).to_dict(orient="records")}
# input_dict = {'data': [[2, 180, 74, 24, 21, 23.9091702, 1.488172308, 22]]}

input_json = json.dumps(input_dict)
predictions = service.run(input_data=input_json)

print(predictions[1])

print(service.get_logs())
