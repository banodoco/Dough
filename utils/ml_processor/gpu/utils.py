from utils.constants import MLQueryObject
from utils.ml_processor.constants import ML_MODEL, MLModel
from utils.ml_processor.comfy_data_transform import ComfyDataTransform


class ComfyRunner:
    pass

def predict_gpu_output(workflow: str, output_node=None) -> str:
    # TODO: lg_complete
    comfy_runner = ComfyRunner()
    output = comfy_runner.predict(
        workflow=workflow,
        stop_server_after_completion=True,
        output_node=output_node
    )

    return output