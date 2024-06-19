import time
import streamlit as st
from backend.models import InternalFileObject
from shared.constants import QUEUE_INFERENCE_QUERIES, InferenceType

from ui_components.methods.common_methods import process_inference_output
from ui_components.widgets.base_theme import BaseTheme as theme
from utils.common_utils import padded_integer
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL


# NOTE: don't update max_step, its logic is hardcoded at the moment
def train_motion_lora(
    input_video: InternalFileObject, lora_prompt: str, lora_name: str, width, height, ckpt, max_step=500
):
    query_obj = MLQueryObject(
        timing_uuid=None,
        model_uuid=None,
        guidance_scale=7.5,
        seed=-1,
        num_inference_steps=25,
        strength=0.7,
        adapter_type=None,
        prompt=lora_prompt,
        negative_prompt="",
        width=width,
        height=height,
        low_threshold=100,
        high_threshold=200,
        image_uuid=None,
        mask_uuid=None,
        data={"file_video": input_video.uuid, "max_step": max_step, "lora_name": lora_name, "ckpt": ckpt},
    )

    ml_client = get_ml_client()
    output, log = ml_client.predict_model_output_standardized(
        ML_MODEL.motion_lora_trainer, query_obj, QUEUE_INFERENCE_QUERIES
    )

    if log:
        inference_data = {
            "inference_type": InferenceType.MOTION_LORA_TRAINING.value,
            "output": output,
            "log_uuid": log.uuid,
            "settings": {},
        }

        process_inference_output(**inference_data)


def inpainting(
    input_image: str, prompt, negative_prompt, width, height, shot_uuid, project_uuid
) -> InternalFileObject:
    mask = st.session_state["mask_to_use"]

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
            "input_image": st.session_state["editing_image"],
            "mask": st.session_state["mask_to_use"],
            "shot_uuid": shot_uuid,
        },
    )

    ml_client = get_ml_client()
    output, log = ml_client.predict_model_output_standardized(
        ML_MODEL.sdxl_inpainting, query_obj, QUEUE_INFERENCE_QUERIES
    )

    return output, log


def query_llama2(prompt, temperature):
    ml_client = get_ml_client()
    input = {
        "debug": False,
        "top_k": 250,
        "top_p": 0.95,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": 30,
        "min_new_tokens": -1,
        "stop_sequences": "\n",
    }

    output, log = ml_client.predict_model_output(ML_MODEL.llama_2_7b, **input)
    result = ""
    for item in output:
        result += item
    return result


def generate_sm_video(shot_uuid, settings={}, variant_count=1, backlog=False, img_list=[]):
    from ui_components.methods.common_methods import process_inference_output

    data_repo = DataRepo()
    ml_client = get_ml_client()

    if not (img_list and len(img_list)):
        timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
        img_list = [t.primary_image for t in timing_list]
    settings.update(file_uuid_list=[t.uuid for t in img_list])

    # res is an array of tuples (video_bytes, log) as there can be multiple videos queued
    res = []
    sm_query = create_sm_query_obj(settings)
    for query in [sm_query] * variant_count:
        out = ml_client.predict_model_output_standardized(
            ML_MODEL.ad_interpolation,
            query,
            QUEUE_INFERENCE_QUERIES,
            backlog,
        )
        res.append(out)

    if res:
        for output, log in res:
            inference_data = {
                "inference_type": InferenceType.FRAME_INTERPOLATION.value,
                "output": output,
                "log_uuid": log.uuid,
                "settings": settings,
                "shot_uuid": str(shot_uuid),
                "inference_tag": settings.get("inference_type", ""),
            }

            process_inference_output(**inference_data)
    else:
        theme.error_msg("Failed to create interpolated clip")


def create_sm_query_obj(settings):
    sm_data = settings
    settings.update(
        {
            "queue_inference": True,
            "amount_of_motion": 1.25,
            "high_detail_mode": settings.get("high_detail_mode", False),
            "filename_prefix": settings.get("filename_prefix", None),
        }
    )

    # adding the input images
    file_data = {}
    for idx, img_uuid in enumerate(settings["file_uuid_list"]):
        file_data[f"file_image_{padded_integer(idx+1)}" + "_uuid"] = {"uuid": img_uuid, "dest": "input/"}

    # adding structure control img
    if "structure_control_image_uuid" in settings and settings["structure_control_image_uuid"] is not None:
        sm_data[f"file_structure_control_img_uuid"] = settings["structure_control_image_uuid"]

    """
    sm_data = {
        **kwargs, # height, width, prompt, filename_prefix etc..
        file_uuid_list, # this is also used in loading images from this variant
        shot_data: {
            motion_data: {
                timing_data: {...}      # distance to next frame, frame strength etc..
                main_setting_data: {...}    # lora used, positive and negative prompt etc..    
            }
        }
    }
    NOTE: rn the sm_data basically has two copies of the input param, one as kwargs and the other as
    shot_data, kwargs - used for plugging into workflow, shot_data - used to plug in the UI
    """
    ml_query_object = MLQueryObject(
        prompt="SM",  # hackish fix
        timing_uuid=None,
        model_uuid=None,
        guidance_scale=None,
        seed=None,
        num_inference_steps=None,
        strength=None,
        adapter_type=None,
        negative_prompt="",
        height=512,
        width=512,
        low_threshold=100,
        high_threshold=200,
        mask_uuid=None,
        data=sm_data,
        file_data=file_data,
    )

    return ml_query_object
