import os
import random
import tempfile
import uuid
from shared.constants import InternalFileType
from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from ui_components.methods.common_methods import combine_mask_and_input_image, random_seed
from ui_components.methods.file_methods import save_or_host_file, zip_images
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow, MLModel
import json


MODEL_PATH_DICT = {
    ComfyWorkflow.SDXL: {"workflow_path": 'comfy_workflows/sdxl_workflow_api.json', "output_node_id": 19},
    ComfyWorkflow.SDXL_IMG2IMG: {"workflow_path": 'comfy_workflows/sdxl_img2img_workflow_api.json', "output_node_id": 31},
    ComfyWorkflow.SDXL_CONTROLNET: {"workflow_path": 'comfy_workflows/sdxl_controlnet_workflow_api.json', "output_node_id": 9},
    ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE: {"workflow_path": 'comfy_workflows/sdxl_openpose_workflow_api.json', "output_node_id": 9},
    ComfyWorkflow.LLAMA_2_7B: {"workflow_path": 'comfy_workflows/llama_workflow_api.json', "output_node_id": 14},
    ComfyWorkflow.SDXL_INPAINTING: {"workflow_path": 'comfy_workflows/sdxl_inpainting_workflow_api.json', "output_node_id": 56},
    ComfyWorkflow.IP_ADAPTER_PLUS: {"workflow_path": 'comfy_workflows/ipadapter_plus_api.json', "output_node_id": 29},
    ComfyWorkflow.IP_ADAPTER_FACE: {"workflow_path": 'comfy_workflows/ipadapter_face_api.json', "output_node_id": 29},
    ComfyWorkflow.IP_ADAPTER_FACE_PLUS: {"workflow_path": 'comfy_workflows/ipadapter_face_plus_api.json', "output_node_id": 29},
    ComfyWorkflow.STEERABLE_MOTION: {"workflow_path": 'comfy_workflows/steerable_motion_api.json', "output_node_id": 281}
}

# these methods return the workflow along with the output node class name
class ComfyDataTransform:
    @staticmethod
    def get_workflow_json(model: ComfyWorkflow):
        json_file_path = "./utils/ml_processor/" + MODEL_PATH_DICT[model]["workflow_path"]
        with open(json_file_path) as f:
            json_data = json.load(f)
            return json_data, [MODEL_PATH_DICT[model]['output_node_id']]

    @staticmethod
    def transform_sdxl_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL)
        
        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale

        # updating params
        seed = random_seed()
        workflow["10"]["inputs"]["noise_seed"] = seed
        workflow["10"]["inputs"]["noise_seed"] = seed
        workflow["5"]["width"], workflow["5"]["height"] = max(width, 1024), max(height, 1024)
        workflow["6"]["inputs"]["text"] = workflow["15"]["inputs"]["text"] = positive_prompt
        workflow["7"]["inputs"]["text"] = workflow["16"]["inputs"]["text"] = negative_prompt
        workflow["10"]["inputs"]["steps"], workflow["10"]["inputs"]["cfg"] = steps, cfg
        workflow["11"]["inputs"]["steps"], workflow["11"]["inputs"]["cfg"] = steps, cfg

        return json.dumps(workflow), output_node_ids, [], []
    
    @staticmethod
    def transform_sdxl_img2img_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_IMG2IMG)

        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = 20, 7      # hardcoding values
        strength = round(query.strength / 100, 1)
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["37:0"]["inputs"]["image"] = image_name
        workflow["42:0"]["inputs"]["text"] = positive_prompt
        workflow["42:1"]["inputs"]["text"] = negative_prompt
        workflow["42:2"]["inputs"]["steps"] = steps
        workflow["42:2"]["inputs"]["cfg"] = cfg
        workflow["42:2"]["inputs"]["denoise"] = 1 - strength
        workflow["42:2"]["inputs"]["seed"] = random_seed()

        return json.dumps(workflow), output_node_ids, [], []
    
    @staticmethod
    def transform_sdxl_controlnet_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_CONTROLNET)

        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        low_threshold, high_threshold = query.low_threshold, query.high_threshold
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["17"]["width"], workflow["17"]["height"] = width, height
        workflow["6"]["inputs"]["text"], workflow["7"]["inputs"]["text"] = positive_prompt, negative_prompt
        workflow["12"]["inputs"]["low_threshold"], workflow["12"]["inputs"]["high_threshold"] = low_threshold, high_threshold
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["13"]["inputs"]["image"] = image_name

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_sdxl_controlnet_openpose_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE)

        # workflow params
        height, width = query.height, query.width
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["11"]["width"], workflow["11"]["height"] = width, height
        workflow["6"]["inputs"]["text"], workflow["7"]["inputs"]["text"] = positive_prompt, negative_prompt
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["12"]["inputs"]["image"] = image_name

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_llama_2_7b_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.LLAMA_2_7B)

        # workflow params
        input_text = query.prompt
        temperature = query.data.get("temperature", 0.8)

        # updating params
        workflow["15"]["inputs"]["prompt"] = input_text
        workflow["15"]["inputs"]["temperature"] = temperature

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_sdxl_inpainting_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_INPAINTING)

        # workflow params
        # node 'get_img_size' automatically fetches the size
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        input_image = query.data.get('data', {}).get('input_image', None)
        mask = query.data.get('data', {}).get('mask', None)
        timing = data_repo.get_timing_from_uuid(query.timing_uuid)

        # inpainting workflows takes in an image and inpaints the transparent area
        combined_img = combine_mask_and_input_image(mask, input_image)
        filename = str(uuid.uuid4()) + ".png"
        hosted_url = save_or_host_file(combined_img, "videos/temp/" + filename)

        file_data = {
            "name": filename,
            "type": InternalFileType.IMAGE.value,
            "project_id": timing.shot.project.uuid
        }

        if hosted_url:
            file_data.update({'hosted_url': hosted_url})
        else:
            file_data.update({'local_path': "videos/temp/" + filename})
        file = data_repo.create_file(**file_data)

        # adding the combined image in query (and removing io buffers)
        query.data = {
            "data": {
                "file_combined_img": file.uuid
            }
        }

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["20"]["inputs"]["image"] = filename
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["34"]["inputs"]["text_g"] = workflow["34"]["inputs"]["text_l"] = positive_prompt
        workflow["37"]["inputs"]["text_g"] = workflow["37"]["inputs"]["text_l"] = negative_prompt

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadaptor_plus_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_PLUS)

        # workflow params
        height, width = query.height, query.width
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        # workflow["24"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["28"]["inputs"]["image"] = image_name  # dummy image
        workflow["6"]["inputs"]["text"] = query.prompt
        workflow["7"]["inputs"]["text"] = query.negative_prompt
        workflow["27"]["inputs"]["weight"] = query.strength
        
        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadaptor_face_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE)

        # workflow params
        height, width = query.height, query.width
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        strength = query.strength

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["24"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["6"]["inputs"]["text"] = query.prompt
        workflow["7"]["inputs"]["text"] = query.negative_prompt                
        workflow["36"]["inputs"]["weight"] = query.strength
        workflow["36"]["inputs"]["weight_v2"] = query.strength

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadaptor_face_plus_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE_PLUS)

        # workflow params
        height, width = query.height, query.width
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        image_2 = data_repo.get_file_from_uuid(query.data.get('data', {}).get("file_image_2_uuid", None))
        image_name_2 = image_2.filename if image_2 else None

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["24"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["28"]["inputs"]["image"] = image_name_2 # insight face image
        workflow["6"]["inputs"]["text"] = query.prompt
        workflow["7"]["inputs"]["text"] = query.negative_prompt
        workflow["29"]["inputs"]["weight"] = query.strength[0]
        workflow["27"]["inputs"]["weight"] = query.strength[1]

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_steerable_motion_workflow(query: MLQueryObject):

        def update_json_with_loras(json_data, loras):
            start_id = 536
            new_ids = []

            # Add LoRAs
            for lora in loras:
                new_id = str(start_id)
                json_data[new_id] = {
                    "inputs": {
                        "lora_name": lora["filename"],
                        "strength": lora["lora_strength"],
                    },
                    "class_type": "ADE_AnimateDiffLoRALoader",
                    "_meta": {
                        "title": "Load AnimateDiff LoRA 🎭🅐🅓"
                    }
                }
                
                if new_ids:
                    json_data[new_ids[-1]]['inputs']['prev_motion_lora'] = [new_id, 0]
                    
                new_ids.append(new_id)
                start_id += 1

            # Update node 545 if needed and if there are new items
            if "545" in json_data and len(new_ids):
                if "motion_lora" not in json_data["545"]["inputs"]:
                    # If "motion_lora" is not present, add it with the specified values
                    json_data["545"]["inputs"]["motion_lora"] = ["536", 0]
                else:
                    # If "motion_lora" is already present, just update the first value
                    json_data["545"]["inputs"]["motion_lora"][0] = "536"

            return json_data
    
        sm_data = query.data.get('data', {})
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.STEERABLE_MOTION)
        workflow = update_json_with_loras(workflow, sm_data.get('lora_data'))

        print(sm_data)
        workflow['464']['inputs']['height'] = sm_data.get('height')
        workflow['464']['inputs']['width'] = sm_data.get('width')
        
        workflow['461']['inputs']['ckpt_name'] = sm_data.get('ckpt')
        
        workflow['558']['inputs']['buffer'] = sm_data.get('buffer')
        workflow['548']['inputs']['text'] = sm_data.get('motion_scales')
        # workflow['548']['inputs']['text'] = sm_data.get('motion_scales')
        workflow['281']['inputs']['format'] = sm_data.get('output_format')
        workflow['541']['inputs']['pre_text'] = sm_data.get('prompt')
        workflow['543']['inputs']['pre_text'] = sm_data.get('negative_prompt')
        workflow['559']['inputs']['multiplier'] = sm_data.get('stmfnet_multiplier')
        workflow['558']['inputs']['relative_ipadapter_strength'] = sm_data.get('relative_ipadapter_strength')
        workflow['558']['inputs']['relative_cn_strength'] = sm_data.get('relative_cn_strength')        
        workflow['558']['inputs']['type_of_strength_distribution'] = sm_data.get('type_of_strength_distribution')
        workflow['558']['inputs']['linear_strength_value'] = sm_data.get('linear_strength_value')
        
        workflow['558']['inputs']['dynamic_strength_values'] = str(sm_data.get('dynamic_strength_values'))[1:-1]  
        workflow['558']['inputs']['linear_frame_distribution_value'] = sm_data.get('linear_frame_distribution_value')                
        workflow['558']['inputs']['dynamic_frame_distribution_values'] = ', '.join(str(int(value)) for value in sm_data.get('dynamic_frame_distribution_values'))        
        workflow['558']['inputs']['type_of_frame_distribution'] = sm_data.get('type_of_frame_distribution')
        workflow['558']['inputs']['type_of_key_frame_influence'] = sm_data.get('type_of_key_frame_influence')
        workflow['558']['inputs']['linear_key_frame_influence_value'] = sm_data.get('linear_key_frame_influence_value')
        
        # print(dynamic_key_frame_influence_values)
        workflow['558']['inputs']['dynamic_key_frame_influence_values'] = str(sm_data.get('dynamic_key_frame_influence_values'))[1:-1]
        workflow['558']['inputs']['ipadapter_noise'] = sm_data.get('ipadapter_noise')
        workflow['342']['inputs']['context_length'] = sm_data.get('context_length')
        workflow['342']['inputs']['context_stride'] = sm_data.get('context_stride')
        workflow['342']['inputs']['context_overlap'] = sm_data.get('context_overlap')
        workflow['468']['inputs']['end_percent'] = sm_data.get('multipled_base_end_percent')
        workflow['470']['inputs']['strength_model'] = sm_data.get('multipled_base_adapter_strength')
        workflow["207"]["inputs"]["noise_seed"] = random_seed()
        workflow["541"]["inputs"]["text"] = sm_data.get('individual_prompts')
        
        # make max_frames an int
        workflow["541"]["inputs"]["max_frames"] = int(float(sm_data.get('max_frames')))
        workflow["543"]["inputs"]["max_frames"] = int(float(sm_data.get('max_frames')))
        workflow["543"]["inputs"]["text"] = sm_data.get('individual_negative_prompts')

        # download the json file as text.json
        # with open("text.json", "w") as f:
        #     f.write(json.dumps(workflow))

        ignore_list = sm_data.get("lora_data", [])
        return json.dumps(workflow), output_node_ids, [], ignore_list


# NOTE: only populating with models currently in use
MODEL_WORKFLOW_MAP = {
    ML_MODEL.sdxl.workflow_name: ComfyDataTransform.transform_sdxl_workflow,
    ML_MODEL.sdxl_controlnet.workflow_name: ComfyDataTransform.transform_sdxl_controlnet_workflow,
    ML_MODEL.sdxl_controlnet_openpose.workflow_name: ComfyDataTransform.transform_sdxl_controlnet_openpose_workflow,
    ML_MODEL.llama_2_7b.workflow_name: ComfyDataTransform.transform_llama_2_7b_workflow,
    ML_MODEL.sdxl_inpainting.workflow_name: ComfyDataTransform.transform_sdxl_inpainting_workflow,
    ML_MODEL.ipadapter_plus.workflow_name: ComfyDataTransform.transform_ipadaptor_plus_workflow,
    ML_MODEL.ipadapter_face.workflow_name: ComfyDataTransform.transform_ipadaptor_face_workflow,
    ML_MODEL.ipadapter_face_plus.workflow_name: ComfyDataTransform.transform_ipadaptor_face_plus_workflow,
    ML_MODEL.ad_interpolation.workflow_name: ComfyDataTransform.transform_steerable_motion_workflow,
    ML_MODEL.sdxl_img2img.workflow_name: ComfyDataTransform.transform_sdxl_img2img_workflow
}

# returns stringified json of the workflow
def get_model_workflow_from_query(model: MLModel, query_obj: MLQueryObject) -> str:
    if model.workflow_name not in MODEL_WORKFLOW_MAP:
        app_logger.log(LoggingType.ERROR, f"model {model.workflow_name} not supported for local inference")
        raise ValueError(f'Model {model.workflow_name} not supported for local inference')
    
    return MODEL_WORKFLOW_MAP[model.workflow_name](query_obj)

def get_workflow_json_url(workflow_json):
    from utils.ml_processor.ml_interface import get_ml_client
    ml_client = get_ml_client()
    temp_fd, temp_json_path = tempfile.mkstemp(suffix='.json')
    
    with open(temp_json_path, 'w') as temp_json_file:
        temp_json_file.write(workflow_json)
    
    return ml_client.upload_training_data(temp_json_path, delete_after_upload=True)


def get_file_list_from_query_obj(query_obj: MLQueryObject):
    file_uuid_list = []

    if query_obj.image_uuid:
        file_uuid_list.append(query_obj.image_uuid)
    
    if query_obj.mask_uuid:
        file_uuid_list.append(query_obj.mask_uuid)
    
    for k, v in query_obj.data.get('data', {}).items():
        if k.startswith("file_"):
            file_uuid_list.append(v)
    
    return file_uuid_list

# returns the zip file which can be passed to the comfy_runner replicate endpoint
def get_file_zip_url(file_uuid_list, index_files=False) -> str:
    from utils.ml_processor.ml_interface import get_ml_client

    data_repo = DataRepo()
    ml_client = get_ml_client()
    
    file_list = data_repo.get_image_list_from_uuid_list(file_uuid_list)
    filename_list = [f.filename for f in file_list] if not index_files else []  # file names would be indexed like 1.png, 2.png ...
    zip_path = zip_images([f.location for f in file_list], 'videos/temp/input_images.zip', filename_list)

    return ml_client.upload_training_data(zip_path, delete_after_upload=True)