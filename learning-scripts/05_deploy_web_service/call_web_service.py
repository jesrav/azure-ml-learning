import json

from azureml.core import Workspace

ws = Workspace.from_config()

service = ws.webservices["diabetes-service"]

x_new = [[2, 180, 74, 24, 21, 23.9091702, 1.488172308, 22]]
print("Patient: {}".format(x_new[0]))

input_json = json.dumps({"data": x_new})

predictions = service.run(input_data=input_json)

print(predictions[0])

print(service.get_logs())
