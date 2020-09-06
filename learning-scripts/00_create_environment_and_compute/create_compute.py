from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config()

compute_name = "aml-cluster"

try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print("Found existing cluster.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS12_V2", max_nodes=4
    )
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
