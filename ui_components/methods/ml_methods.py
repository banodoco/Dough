
import os
import tempfile
import streamlit as st
import replicate
from typing import List
from PIL import Image
import uuid
import urllib
from backend.models import InternalFileObject
from shared.constants import GPU_INFERENCE_ENABLED, QUEUE_INFERENCE_QUERIES, SERVER, AIModelCategory, InferenceType, InternalFileType, ServerType
from ui_components.constants import MASK_IMG_LOCAL_PATH, TEMP_MASK_FILE
from ui_components.methods.common_methods import combine_mask_and_input_image, process_inference_output
from ui_components.methods.file_methods import save_or_host_file
from ui_components.models import InternalAIModelObject, InternalFrameTimingObject, InternalSettingObject
from utils.constants import ImageStage, MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL, MLModel


# NOTE: code not is use
# def trigger_restyling_process(timing_uuid, update_inference_settings, \
#                               transformation_stage, promote_new_generation, **kwargs):
#     data_repo = DataRepo()
    
#     timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
#     project_settings: InternalSettingObject = data_repo.get_project_setting(timing.shot.project.uuid)
    
#     source_image = timing.source_image if transformation_stage == ImageStage.SOURCE_IMAGE.value else \
#                             timing.primary_image

#     query_obj = MLQueryObject(
#         timing_uuid, 
#         image_uuid=source_image.uuid if 'add_image_in_params' in kwargs and kwargs['add_image_in_params'] else None, 
#         width=project_settings.width, 
#         height=project_settings.height, 
#         **kwargs
#     )

#     prompt = query_obj.prompt
#     if update_inference_settings is True:
#         prompt = prompt.replace(",", ".")
#         prompt = prompt.replace("\n", "")

#         project_settings.batch_prompt = prompt
#         project_settings.batch_strength = query_obj.strength
#         project_settings.batch_negative_prompt = query_obj.negative_prompt
#         project_settings.batch_guidance_scale = query_obj.guidance_scale
#         project_settings.batch_seed = query_obj.seed
#         project_settings.batch_num_inference_steps = query_obj.num_inference_steps
#         # project_settings.batch_custom_models = query_obj.data.get('custom_models', []),
#         project_settings.batch_adapter_type = query_obj.adapter_type
#         # project_settings.batch_add_image_in_params = st.session_state['add_image_in_params'],

#     # query_obj.prompt = dynamic_prompting(prompt, source_image)
#     output, log = restyle_images(query_obj, QUEUE_INFERENCE_QUERIES)

#     inference_data = {
#         "inference_type": InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value,
#         "output": output,
#         "log_uuid": log.uuid,
#         "timing_uuid": timing_uuid,
#         "promote_new_generation": promote_new_generation,
#     }
#     process_inference_output(**inference_data)

# def restyle_images(query_obj: MLQueryObject, queue_inference=False) -> InternalFileObject:
#     data_repo = DataRepo()
#     ml_client = get_ml_client()
#     db_model  = data_repo.get_ai_model_from_uuid(query_obj.model_uuid)

#     if db_model.category == AIModelCategory.LORA.value:
#         model = ML_MODEL.clones_lora_training_2
#         output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

#     elif db_model.category == AIModelCategory.CONTROLNET.value:
#         adapter_type = query_obj.adapter_type
#         if adapter_type == "normal":
#             model = ML_MODEL.jagilley_controlnet_normal
#         elif adapter_type == "canny":
#             model = ML_MODEL.jagilley_controlnet_canny
#         elif adapter_type == "hed":
#             model = ML_MODEL.jagilley_controlnet_hed
#         elif adapter_type == "scribble":
#             model = ML_MODEL.jagilley_controlnet_scribble 
#         elif adapter_type == "seg":
#             model = ML_MODEL.jagilley_controlnet_seg
#         elif adapter_type == "hough":
#             model = ML_MODEL.jagilley_controlnet_hough
#         elif adapter_type == "depth2img":
#             model = ML_MODEL.jagilley_controlnet_depth2img
#         elif adapter_type == "pose":
#             model = ML_MODEL.jagilley_controlnet_pose
#         output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

#     elif db_model.category == AIModelCategory.DREAMBOOTH.value:
#         output, log = prompt_model_dreambooth(query_obj,  queue_inference=queue_inference)

#     else:
#         model = ML_MODEL.get_model_by_db_obj(db_model)   # TODO: remove this dependency
#         output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

#     return output, log

# def prompt_model_dreambooth(query_obj: MLQueryObject, queue_inference=False):
#     data_repo = DataRepo()
#     ml_client = get_ml_client()

#     model_uuid = query_obj.data.get('dreambooth_model_uuid', None)
#     if not model_uuid:
#         st.error('No dreambooth model selected')
#         return
    
#     dreambooth_model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(model_uuid)
    
#     model_name = dreambooth_model.name
#     model_id = dreambooth_model.replicate_url

#     if not dreambooth_model.version:
#         version = ml_client.get_model_version_from_id(model_id)
#         data_repo.update_ai_model(uuid=dreambooth_model.uuid, version=version)
#     else:
#         version = dreambooth_model.version

#     app_setting = data_repo.get_app_setting_from_uuid()
#     model = MLModel(f"{app_setting.replicate_username}/{model_name}", version)
#     output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

#     return output, log

# NOTE: code not is use
# def prompt_clip_interrogator(input_image, which_model, best_or_fast):
#     if which_model == "Stable Diffusion 1.5":
#         which_model = "ViT-L-14/openai"
#     elif which_model == "Stable Diffusion 2":
#         which_model = "ViT-H-14/laion2b_s32b_b79k"

#     if not input_image.startswith("http"):
#         input_image = open(input_image, "rb")

#     ml_client = get_ml_client()
#     output, _ = ml_client.predict_model_output(
#         ML_MODEL.clip_interrogator, image=input_image, clip_model_name=which_model, mode=best_or_fast)

#     return output

# NOTE: code not is use
# def prompt_model_blip2(input_image, query):
#     if not input_image.startswith("http"):
#         input_image = open(input_image, "rb")

#     ml_client = get_ml_client()
#     output, _ = ml_client.predict_model_output(
#         ML_MODEL.salesforce_blip_2, image=input_image, question=query)

#     return output

# NOTE: code not is use
# def facial_expression_recognition(input_image):
#     input_image = input_image.location
#     if not input_image.startswith("http"):
#         input_image = open(input_image, "rb")

#     ml_client = get_ml_client()
#     output, _ = ml_client.predict_model_output(
#         ML_MODEL.phamquiluan_face_recognition, input_path=input_image)

#     emo_label = output[0]["emo_label"]
#     if emo_label == "disgust":
#         emo_label = "disgusted"
#     elif emo_label == "fear":
#         emo_label = "fearful"
#     elif emo_label == "surprised":
#         emo_label = "surprised"
#     emo_proba = output[0]["emo_proba"]
#     if emo_proba > 0.95:
#         emotion = (f"very {emo_label} expression")
#     elif emo_proba > 0.85:
#         emotion = (f"{emo_label} expression")
#     elif emo_proba > 0.75:
#         emotion = (f"somewhat {emo_label} expression")
#     elif emo_proba > 0.65:
#         emotion = (f"slightly {emo_label} expression")
#     elif emo_proba > 0.55:
#         emotion = (f"{emo_label} expression")
#     else:
#         emotion = (f"neutral expression")
#     return emotion

def inpainting(input_image: str, prompt, negative_prompt, width, height, shot_uuid, project_uuid) -> InternalFileObject:
    data_repo = DataRepo()
    # timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

  
    project = data_repo.get_project_from_uuid(project_uuid)
    mask = st.session_state['mask_to_use']

    if not mask.startswith("http"):
        mask = open(mask, "rb")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    query_obj = MLQueryObject(
        timing_uuid=None,
        model_uuid=None,
        guidance_scale=7.5,
        seed=-1,
        num_inference_steps=25,            
        strength=0.7,
        adapter_type=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        
        low_threshold=100,  # update these default values
        high_threshold=200,
        image_uuid=None,
        mask_uuid=None,
        data={
            "input_image": st.session_state['editing_image'],
            "mask": st.session_state['mask_to_use'] ,
            "shot_uuid": shot_uuid,
        }
    )

    ml_client = get_ml_client()
    output, log = ml_client.predict_model_output_standardized(
        ML_MODEL.sdxl_inpainting, 
        query_obj,
        QUEUE_INFERENCE_QUERIES
    )

    return output, log

# NOTE: code not is use
# def remove_background(input_image):
#     if not input_image.startswith("http"):
#         input_image = open(input_image, "rb")

#     ml_client = get_ml_client()
#     output, _ = ml_client.predict_model_output(
#         ML_MODEL.pollination_modnet, image=input_image)
#     return output

# NOTE: code not is use
# def create_depth_mask_image(input_image, layer, timing_uuid):
#     from ui_components.methods.common_methods import create_or_update_mask
    
#     if not input_image.startswith("http"):
#         input_image = open(input_image, "rb")

#     ml_client = get_ml_client()
#     output, log = ml_client.predict_model_output(
#         ML_MODEL.cjwbw_midas, image=input_image, model_type="dpt_beit_large_512")
#     try:
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
#         with urllib.request.urlopen(output) as response, open(temp_file.name, 'wb') as out_file:
#             out_file.write(response.read())
#     except Exception as e:
#         print(e)

#     depth_map = Image.open(temp_file.name)
#     os.remove(temp_file.name)
#     depth_map = depth_map.convert("L")  # Convert to grayscale image
#     pixels = depth_map.load()
#     mask = Image.new("L", depth_map.size)
#     mask_pixels = mask.load()

#     fg_mask = Image.new("L", depth_map.size) if "Foreground" in layer else None
#     mg_mask = Image.new(
#         "L", depth_map.size) if "Middleground" in layer else None
#     bg_mask = Image.new("L", depth_map.size) if "Background" in layer else None

#     fg_pixels = fg_mask.load() if fg_mask else None
#     mg_pixels = mg_mask.load() if mg_mask else None
#     bg_pixels = bg_mask.load() if bg_mask else None

#     for i in range(depth_map.size[0]):
#         for j in range(depth_map.size[1]):
#             depth_value = pixels[i, j]

#             if fg_pixels:
#                 fg_pixels[i, j] = 0 if depth_value > 200 else 255
#             if mg_pixels:
#                 mg_pixels[i, j] = 0 if depth_value <= 200 and depth_value > 50 else 255
#             if bg_pixels:
#                 bg_pixels[i, j] = 0 if depth_value <= 50 else 255

#             mask_pixels[i, j] = 255
#             if fg_pixels:
#                 mask_pixels[i, j] &= fg_pixels[i, j]
#             if mg_pixels:
#                 mask_pixels[i, j] &= mg_pixels[i, j]
#             if bg_pixels:
#                 mask_pixels[i, j] &= bg_pixels[i, j]

#     return create_or_update_mask(timing_uuid, mask)

# NOTE: code not is use
# def dynamic_prompting(prompt, source_image):
#     if "[expression]" in prompt:
#         prompt_expression = facial_expression_recognition(source_image)
#         prompt = prompt.replace("[expression]", prompt_expression)

#     if "[location]" in prompt:
#         prompt_location = prompt_model_blip2(
#             source_image, "What's surrounding the character?")
#         prompt = prompt.replace("[location]", prompt_location)

#     if "[mouth]" in prompt:
#         prompt_mouth = prompt_model_blip2(
#             source_image, "is their mouth open or closed?")
#         prompt = prompt.replace("[mouth]", "mouth is " + str(prompt_mouth))

#     if "[looking]" in prompt:
#         prompt_looking = prompt_model_blip2(
#             source_image, "the person is looking")
#         prompt = prompt.replace("[looking]", "looking " + str(prompt_looking))

#     return prompt

def query_llama2(prompt, temperature):
    ml_client = get_ml_client()
    input={
            "debug": False,
            "top_k": 250,
            "top_p": 0.95,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": 30,
            "min_new_tokens": -1,
            "stop_sequences": "\n"
        }
    
    output, log = ml_client.predict_model_output(ML_MODEL.llama_2_7b, **input)
    result = ""
    for item in output:
        result += item
    return result