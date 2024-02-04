
# returns stringified json of the workflow
from utils.constants import MLQueryObject
from utils.ml_processor.constants import MLModel


def get_model_workflow_from_query(model: MLModel, query_obj: MLQueryObject) -> str:
    # TODO: lg_complete
    pass

def predict_gpu_output(workflow: str) -> str:
    # TODO: lg_complete
    # 1. returns the output file location
    # 2. clears json files (if created)
    pass