from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from utils.constants import MLQueryObject
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow, MLModel
import json

# TODO: lg_complete
# these methods return the workflow along with the output node class name
class ComfyDataTransform:
    @staticmethod
    def _get_workflow_json(model: ComfyWorkflow):
        MODEL_PATH_DICT = {
            ComfyWorkflow.SDXL: './comfy_workflows/sdxl_api.json',
            ComfyWorkflow.SDXL_CONTROLNET: './comfy_workflows/sdxl_controlnet_api.json',
            ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE: './comfy_workflows/sdxl_controlnet_openpose_api.json',
            ComfyWorkflow.LLAMA_2_7B: './comfy_workflows/llama_2_7b_api.json',
            ComfyWorkflow.SDXL_INPAINTING: './comfy_workflows/sdxl_inpainting_api.json',
            ComfyWorkflow.IP_ADAPTER_PLUS: './comfy_workflows/ipadapter_plus_api.json',
            ComfyWorkflow.IP_ADAPTER_FACE: './comfy_workflows/ipadapter_face_api.json',
            ComfyWorkflow.IP_ADAPTER_FACE_PLUS: './comfy_workflows/ipadapter_face_plus_api.json',
            ComfyWorkflow.STEERABLE_MOTION: './comfy_workflows/steerable_motion_api.json'
        }

        json_file_path = MODEL_PATH_DICT[model]
        with open(json_file_path) as f:
            json_data = json.load(f)
            return json_data

    @staticmethod
    def transform_sdxl_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_sdxl_controlnet_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_sdxl_controlnet_openpose_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_llama_2_7b_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_sdxl_inpainting_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_ipadaptor_plus_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_ipadaptor_face_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_ipadaptor_face_plus_workflow(query: MLQueryObject):
        pass

    @staticmethod
    def transform_steerable_motion_workflow(query: MLQueryObject):
        pass


# NOTE: only populating with models currently in use
MODEL_WORKFLOW_MAP = {
    ML_MODEL.sdxl: ComfyDataTransform.transform_sdxl_workflow,
    ML_MODEL.sdxl_controlnet: ComfyDataTransform.transform_sdxl_controlnet_workflow,
    ML_MODEL.sdxl_controlnet_openpose: ComfyDataTransform.transform_sdxl_controlnet_openpose_workflow,
    ML_MODEL.llama_2_7b: ComfyDataTransform.transform_llama_2_7b_workflow,
    ML_MODEL.sdxl_inpainting: ComfyDataTransform.transform_sdxl_inpainting_workflow,
    ML_MODEL.ip_adapter_plus: ComfyDataTransform.transform_ipadaptor_plus_workflow,
    ML_MODEL.ip_adapter_face: ComfyDataTransform.transform_ipadaptor_face_workflow,
    ML_MODEL.ip_adapter_face_plus: ComfyDataTransform.transform_ipadaptor_face_plus_workflow,
    ML_MODEL.ad_interpolation: ComfyDataTransform.transform_steerable_motion_workflow
}

# returns stringified json of the workflow
def get_model_workflow_from_query(model: MLModel, query_obj: MLQueryObject) -> str:
    if model not in MODEL_WORKFLOW_MAP:
        app_logger.log(LoggingType.ERROR, f"model {model.name} not supported for local inference")
        raise ValueError(f'Model {model.name} not supported for local inference')
    
    return MODEL_WORKFLOW_MAP[model](query_obj)