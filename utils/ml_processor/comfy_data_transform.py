from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from ui_components.methods.common_methods import get_filename_from_url
from ui_components.methods.file_methods import zip_images
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow, MLModel
import json

MODEL_PATH_DICT = {
    ComfyWorkflow.SDXL: {"workflow_path": './comfy_workflows/sdxl_workflow_api.json', "output_node_id": 19},
    ComfyWorkflow.SDXL_CONTROLNET: {"workflow_path": './comfy_workflows/sdxl_controlnet_workflow_api.json', "output_node_id": 9},
    ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE: {"workflow_path": './comfy_workflows/sdxl_openpose_workflow_api.json', "output_node_id": 9},
    ComfyWorkflow.LLAMA_2_7B: {"workflow_path": './comfy_workflows/llama_workflow_api.json', "output_node_id": 14},
    ComfyWorkflow.SDXL_INPAINTING: {"workflow_path": './comfy_workflows/sdxl_inpainting_workflow_api.json', "output_node_id": 56},
    ComfyWorkflow.IP_ADAPTER_PLUS: {"workflow_path": './comfy_workflows/ipadapter_plus_api.json', "output_node_id": 29},
    ComfyWorkflow.IP_ADAPTER_FACE: {"workflow_path": './comfy_workflows/ipadapter_face_api.json', "output_node_id": 29},
    ComfyWorkflow.IP_ADAPTER_FACE_PLUS: {"workflow_path": './comfy_workflows/ipadapter_face_plus_api.json', "output_node_id": 29},
    ComfyWorkflow.STEERABLE_MOTION: {"workflow_path": './comfy_workflows/steerable_motion_api.json', "output_node_id": 281}
}

# these methods return the workflow along with the output node class name
class ComfyDataTransform:
    @staticmethod
    def get_workflow_json(model: ComfyWorkflow):
        json_file_path = MODEL_PATH_DICT[model]["workflow_path"]
        with open(json_file_path) as f:
            json_data = json.load(f)
            return json_data

    @staticmethod
    def transform_sdxl_workflow(query: MLQueryObject):
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL)
        
        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale

        # updating params
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["6"]["inputs"]["text"] = workflow["15"]["inputs"]["text"] = positive_prompt
        workflow["7"]["inputs"]["text"] = workflow["16"]["inputs"]["text"] = negative_prompt
        workflow["10"]["inputs"]["steps"], workflow["11"]["inputs"]["cfg"] = steps, cfg

        return json.dumps(workflow)


    @staticmethod
    def transform_sdxl_controlnet_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_CONTROLNET)

        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        low_threshold, high_threshold = query.low_threshold, query.high_threshold
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["17"]["width"], workflow["17"]["height"] = width, height
        workflow["6"]["inputs"]["text"], workflow["7"]["inputs"]["text"] = positive_prompt, negative_prompt
        workflow["12"]["inputs"]["low_threshold"], workflow["12"]["inputs"]["high_threshold"] = low_threshold, high_threshold
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["13"]["inputs"]["image"] = image_name

        return json.dumps(workflow)


    @staticmethod
    def transform_sdxl_controlnet_openpose_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE)

        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["11"]["width"], workflow["11"]["height"] = width, height
        workflow["6"]["inputs"]["text"], workflow["7"]["inputs"]["text"] = positive_prompt, negative_prompt
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["12"]["inputs"]["image"] = image_name

        return json.dumps(workflow)

    @staticmethod
    def transform_llama_2_7b_workflow(query: MLQueryObject):
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.LLAMA_2_7B)

        # workflow params
        input_text = query.prompt
        temperature = query.data.get("temperature", 0.8)

        # updating params
        workflow["15"]["inputs"]["prompt"] = input_text
        workflow["15"]["inputs"]["temperature"] = temperature

        return json.dumps(workflow)

    @staticmethod
    def transform_sdxl_inpainting_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_INPAINTING)

        # workflow params
        # node 'get_img_size' automatically fetches the size
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["20"]["inputs"]["image"] = image_name
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["34"]["inputs"]["text_g"] = workflow["34"]["inputs"]["text_l"] = positive_prompt
        workflow["37"]["inputs"]["text_g"] = workflow["37"]["inputs"]["text_l"] = negative_prompt

        return json.dumps(workflow)


    @staticmethod
    def transform_ipadaptor_plus_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_PLUS)

        # workflow params
        height, width = query.height, query.width
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        image_2 = data_repo.get_file_from_uuid(query.data.get("file_image_2_uuid", None))
        image_name_2 = image_2.filename if image_2 else None

        # updating params
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["28"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["24"]["inputs"]["image"] = image_name_2 # insight face image

        return json.dumps(workflow)


    @staticmethod
    def transform_ipadaptor_face_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE)

        # workflow params
        height, width = query.height, query.width
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        image_2 = data_repo.get_file_from_uuid(query.data.get("file_image_2_uuid", None))
        image_name_2 = image_2.filename if image_2 else None

        # updating params
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["28"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["24"]["inputs"]["image"] = image_name_2 # insight face image

        return json.dumps(workflow)


    @staticmethod
    def transform_ipadaptor_face_plus_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE)

        # workflow params
        height, width = query.height, query.width
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        image_2 = data_repo.get_file_from_uuid(query.data.get("file_image_2_uuid", None))
        image_name_2 = image_2.filename if image_2 else None

        # updating params
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["28"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["24"]["inputs"]["image"] = image_name_2 # insight face image

        return json.dumps(workflow)

    @staticmethod
    def transform_steerable_motion_workflow(query: MLQueryObject):
        workflow = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE)

        # workflow params
        steps, cfg = query.num_inference_steps, query.guidance_scale
        # all the images will directly be sent in the input files

        # updating params
        workflow["207"]["inputs"]["steps"], workflow["207"]["inputs"]["cfg"] = steps, cfg

        return json.dumps(workflow)


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

# returns the zip file which can be passed to the comfy_runner replicate endpoint
def get_file_zip(query_obj: MLQueryObject) -> str:
    data_repo = DataRepo()
    file_uuid_list = []

    if query_obj.image_uuid:
        file_uuid_list.append(query_obj.image_uuid)
    
    for k, v in query_obj.data.items():
        if k.startswith("file_"):
            file_uuid_list.append(v)

    file_list = data_repo.get_image_list_from_uuid_list(file_uuid_list)
    return zip_images([f.location for f in file_list], 'videos/temp/input_images.zip')