from utils.constants import MLQueryObject
from utils.ml_processor.constants import ML_MODEL, MLModel
from utils.ml_processor.comfy_data_transform import ComfyDataTransform


# NOTE: only populating with models currently in use
MODEL_WORKFLOW_MAP = {
    ML_MODEL.sdxl: ComfyDataTransform.transform_sdxl_workflow,
    ML_MODEL.sdxl_controlnet: ComfyDataTransform.transform_sdxl_controlnet_workflow,
    ML_MODEL.sdxl_controlnet_openpose: ComfyDataTransform.transform_sdxl_controlnet_openpose_workflow,
    ML_MODEL.llama_2_7b: ComfyDataTransform.transform_llama_2_7b_workflow,
    ML_MODEL.sdxl_inpainting: ComfyDataTransform.transform_sdxl_inpainting_workflow,
}

# returns stringified json of the workflow
def get_model_workflow_from_query(model: MLModel, query_obj: MLQueryObject) -> str:
    if model not in MODEL_WORKFLOW_MAP:
        raise ValueError(f'Model {model} not supported for local inference')
    
    return MODEL_WORKFLOW_MAP[model](query_obj)

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