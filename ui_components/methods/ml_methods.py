
import os
import tempfile
import streamlit as st
import replicate
from typing import List
from PIL import Image
import uuid
import urllib
from backend.models import InternalFileObject
from shared.constants import QUEUE_INFERENCE_QUERIES, SERVER, AIModelCategory, InferenceType, InternalFileType, ServerType
from ui_components.constants import MASK_IMG_LOCAL_PATH, TEMP_MASK_FILE
from ui_components.methods.common_methods import process_inference_output
from ui_components.models import InternalAIModelObject, InternalFrameTimingObject, InternalSettingObject
from utils.constants import ImageStage, MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import REPLICATE_MODEL, ReplicateModel


def trigger_restyling_process(timing_uuid, update_inference_settings, \
                              transformation_stage, promote_new_generation, **kwargs):
    data_repo = DataRepo()
    
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(timing.shot.project.uuid)
    
    source_image = timing.source_image if transformation_stage == ImageStage.SOURCE_IMAGE.value else \
                            timing.primary_image

    query_obj = MLQueryObject(
        timing_uuid, 
        image_uuid=source_image.uuid if 'add_image_in_params' in kwargs and kwargs['add_image_in_params'] else None, 
        width=project_settings.width, 
        height=project_settings.height, 
        **kwargs
    )

    prompt = query_obj.prompt
    if update_inference_settings is True:
        prompt = prompt.replace(",", ".")
        prompt = prompt.replace("\n", "")
        data_repo.update_project_setting(
            timing.shot.project.uuid,
            default_prompt=prompt,
            default_strength=query_obj.strength,
            default_model_id=query_obj.model_uuid,
            default_negative_prompt=query_obj.negative_prompt,
            default_guidance_scale=query_obj.guidance_scale,
            default_seed=query_obj.seed,
            default_num_inference_steps=query_obj.num_inference_steps,
            default_which_stage_to_run_on=transformation_stage,
            default_custom_models=query_obj.data.get('custom_models', []),
            default_adapter_type=query_obj.adapter_type,
            default_low_threshold=query_obj.low_threshold,
            default_high_threshold=query_obj.high_threshold,
            add_image_in_params=st.session_state['add_image_in_params'],
        )

    query_obj.prompt = dynamic_prompting(prompt, source_image)
    output, log = restyle_images(query_obj, QUEUE_INFERENCE_QUERIES)

    inference_data = {
        "inference_type": InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value,
        "output": output,
        "log_uuid": log.uuid,
        "timing_uuid": timing_uuid,
        "promote_new_generation": promote_new_generation,
    }
    process_inference_output(**inference_data)
    

def restyle_images(query_obj: MLQueryObject, queue_inference=False) -> InternalFileObject:
    data_repo = DataRepo()
    ml_client = get_ml_client()
    db_model  = data_repo.get_ai_model_from_uuid(query_obj.model_uuid)

    if db_model.category == AIModelCategory.LORA.value:
        model = REPLICATE_MODEL.clones_lora_training_2
        output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

    elif db_model.category == AIModelCategory.CONTROLNET.value:
        adapter_type = query_obj.adapter_type
        if adapter_type == "normal":
            model = REPLICATE_MODEL.jagilley_controlnet_normal
        elif adapter_type == "canny":
            model = REPLICATE_MODEL.jagilley_controlnet_canny
        elif adapter_type == "hed":
            model = REPLICATE_MODEL.jagilley_controlnet_hed
        elif adapter_type == "scribble":
            model = REPLICATE_MODEL.jagilley_controlnet_scribble 
        elif adapter_type == "seg":
            model = REPLICATE_MODEL.jagilley_controlnet_seg
        elif adapter_type == "hough":
            model = REPLICATE_MODEL.jagilley_controlnet_hough
        elif adapter_type == "depth2img":
            model = REPLICATE_MODEL.jagilley_controlnet_depth2img
        elif adapter_type == "pose":
            model = REPLICATE_MODEL.jagilley_controlnet_pose
        output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

    elif db_model.category == AIModelCategory.DREAMBOOTH.value:
        output, log = prompt_model_dreambooth(query_obj,  queue_inference=queue_inference)

    else:
        model = REPLICATE_MODEL.get_model_by_db_obj(db_model)   # TODO: remove this dependency
        output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

    return output, log

def prompt_model_dreambooth(query_obj: MLQueryObject, queue_inference=False):
    data_repo = DataRepo()
    ml_client = get_ml_client()

    model_uuid = query_obj.data.get('dreambooth_model_uuid', None)
    if not model_uuid:
        st.error('No dreambooth model selected')
        return
    
    dreambooth_model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(model_uuid)
    
    model_name = dreambooth_model.name
    model_id = dreambooth_model.replicate_url

    if not dreambooth_model.version:
        version = ml_client.get_model_version_from_id(model_id)
        data_repo.update_ai_model(uuid=dreambooth_model.uuid, version=version)
    else:
        version = dreambooth_model.version

    app_setting = data_repo.get_app_setting_from_uuid()
    model = ReplicateModel(f"{app_setting.replicate_username}/{model_name}", version)
    output, log = ml_client.predict_model_output_standardized(model, query_obj, queue_inference=queue_inference)

    return output, log


def prompt_clip_interrogator(input_image, which_model, best_or_fast):
    if which_model == "Stable Diffusion 1.5":
        which_model = "ViT-L-14/openai"
    elif which_model == "Stable Diffusion 2":
        which_model = "ViT-H-14/laion2b_s32b_b79k"

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output, _ = ml_client.predict_model_output(
        REPLICATE_MODEL.clip_interrogator, image=input_image, clip_model_name=which_model, mode=best_or_fast)

    return output

def prompt_model_blip2(input_image, query):
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output, _ = ml_client.predict_model_output(
        REPLICATE_MODEL.salesforce_blip_2, image=input_image, question=query)

    return output

def facial_expression_recognition(input_image):
    input_image = input_image.location
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output, _ = ml_client.predict_model_output(
        REPLICATE_MODEL.phamquiluan_face_recognition, input_path=input_image)

    emo_label = output[0]["emo_label"]
    if emo_label == "disgust":
        emo_label = "disgusted"
    elif emo_label == "fear":
        emo_label = "fearful"
    elif emo_label == "surprised":
        emo_label = "surprised"
    emo_proba = output[0]["emo_proba"]
    if emo_proba > 0.95:
        emotion = (f"very {emo_label} expression")
    elif emo_proba > 0.85:
        emotion = (f"{emo_label} expression")
    elif emo_proba > 0.75:
        emotion = (f"somewhat {emo_label} expression")
    elif emo_proba > 0.65:
        emotion = (f"slightly {emo_label} expression")
    elif emo_proba > 0.55:
        emotion = (f"{emo_label} expression")
    else:
        emotion = (f"neutral expression")
    return emotion

def inpainting(input_image: str, prompt, negative_prompt, timing_uuid, mask_in_project=False) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    if mask_in_project == False:
        mask = timing.mask.location
    else:
        mask = timing.shot.project.get_temp_mask_file(TEMP_MASK_FILE).location

    if not mask.startswith("http"):
        mask = open(mask, "rb")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output, log = ml_client.predict_model_output(
        REPLICATE_MODEL.sdxl_inpainting, 
        mask=mask, 
        image=input_image, 
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=25, 
        strength=1.0,
        queue_inference=True
    )

    return output, log

def remove_background(input_image):
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output, _ = ml_client.predict_model_output(
        REPLICATE_MODEL.pollination_modnet, image=input_image)
    return output

def create_depth_mask_image(input_image, layer, timing_uuid):
    from ui_components.methods.common_methods import create_or_update_mask
    
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output, log = ml_client.predict_model_output(
        REPLICATE_MODEL.cjwbw_midas, image=input_image, model_type="dpt_beit_large_512")
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
        with urllib.request.urlopen(output) as response, open(temp_file.name, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(e)

    depth_map = Image.open(temp_file.name)
    os.remove(temp_file.name)
    depth_map = depth_map.convert("L")  # Convert to grayscale image
    pixels = depth_map.load()
    mask = Image.new("L", depth_map.size)
    mask_pixels = mask.load()

    fg_mask = Image.new("L", depth_map.size) if "Foreground" in layer else None
    mg_mask = Image.new(
        "L", depth_map.size) if "Middleground" in layer else None
    bg_mask = Image.new("L", depth_map.size) if "Background" in layer else None

    fg_pixels = fg_mask.load() if fg_mask else None
    mg_pixels = mg_mask.load() if mg_mask else None
    bg_pixels = bg_mask.load() if bg_mask else None

    for i in range(depth_map.size[0]):
        for j in range(depth_map.size[1]):
            depth_value = pixels[i, j]

            if fg_pixels:
                fg_pixels[i, j] = 0 if depth_value > 200 else 255
            if mg_pixels:
                mg_pixels[i, j] = 0 if depth_value <= 200 and depth_value > 50 else 255
            if bg_pixels:
                bg_pixels[i, j] = 0 if depth_value <= 50 else 255

            mask_pixels[i, j] = 255
            if fg_pixels:
                mask_pixels[i, j] &= fg_pixels[i, j]
            if mg_pixels:
                mask_pixels[i, j] &= mg_pixels[i, j]
            if bg_pixels:
                mask_pixels[i, j] &= bg_pixels[i, j]

    return create_or_update_mask(timing_uuid, mask)

def dynamic_prompting(prompt, source_image):
    if "[expression]" in prompt:
        prompt_expression = facial_expression_recognition(source_image)
        prompt = prompt.replace("[expression]", prompt_expression)

    if "[location]" in prompt:
        prompt_location = prompt_model_blip2(
            source_image, "What's surrounding the character?")
        prompt = prompt.replace("[location]", prompt_location)

    if "[mouth]" in prompt:
        prompt_mouth = prompt_model_blip2(
            source_image, "is their mouth open or closed?")
        prompt = prompt.replace("[mouth]", "mouth is " + str(prompt_mouth))

    if "[looking]" in prompt:
        prompt_looking = prompt_model_blip2(
            source_image, "the person is looking")
        prompt = prompt.replace("[looking]", "looking " + str(prompt_looking))

    return prompt

def query_llama2(user_instructions, system_instructions):
    ml_client = get_ml_client()
    input={
            "debug": False,
            "top_k": 250,
            "top_p": 0.95,
            "prompt": system_instructions + "\n" + user_instructions + "|",
            "temperature": 0.73,
            "max_new_tokens": 30,
            "min_new_tokens": -1,
            "stop_sequences": "\n"
        }
    
    output, log = ml_client.predict_model_output(REPLICATE_MODEL.llama_2_7b, **input)
    result = ""
    for item in output:
        result += item
    return result