from pathlib import Path
import os
import csv
import subprocess
import psutil
import streamlit as st
import json
from shared.constants import SERVER, ServerType
from ui_components.models import InternalUserObject
from utils.cache.cache import StCache
from utils.data_repo.data_repo import DataRepo

def copy_sample_assets(project_uuid):
    import shutil

    # copy sample video
    source = "sample_assets/sample_videos/sample.mp4"
    dest = "videos/" + project_uuid + "/assets/resources/input_videos/sample.mp4"
    shutil.copyfile(source, dest)

def create_working_assets(project_uuid):
    if SERVER != ServerType.DEVELOPMENT.value:
        return

    new_project = True
    if os.path.exists("videos/"+project_uuid):
        new_project = False

    directory_list = [
        # project specific files
        "videos/" + project_uuid,
        "videos/" + project_uuid + "/temp",
        "videos/" + project_uuid + "/assets",
        "videos/" + project_uuid + "/assets/frames",
        "videos/" + project_uuid + "/assets/frames/0_extracted",
        "videos/" + project_uuid + "/assets/frames/1_selected",
        "videos/" + project_uuid + "/assets/frames/2_character_pipeline_completed",
        "videos/" + project_uuid + "/assets/frames/3_backdrop_pipeline_completed",
        "videos/" + project_uuid + "/assets/resources",
        "videos/" + project_uuid + "/assets/resources/backgrounds",
        "videos/" + project_uuid + "/assets/resources/masks",
        "videos/" + project_uuid + "/assets/resources/audio",
        "videos/" + project_uuid + "/assets/resources/input_videos",
        "videos/" + project_uuid + "/assets/resources/prompt_images",
        "videos/" + project_uuid + "/assets/videos",
        "videos/" + project_uuid + "/assets/videos/0_raw",
        "videos/" + project_uuid + "/assets/videos/1_final",
        "videos/" + project_uuid + "/assets/videos/2_completed",
        # app data
        "inference_log",
        # temp folder
        "videos/temp",
        "videos/temp/assets/videos/0_raw/"
    ]
    
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # copying sample assets for new project
    if new_project:
        copy_sample_assets(project_uuid)

def truncate_decimal(num: float, n: int = 2) -> float:
    return int(num * 10 ** n) / 10 ** n


def get_current_user() -> InternalUserObject:
    data_repo = DataRepo()
    user = data_repo.get_first_active_user()
    return user

def user_credits_available():
    current_user = get_current_user()
    return True if (current_user and current_user.total_credits > 0) else False

def get_current_user_uuid():
    current_user = get_current_user()
    if current_user:
        return current_user.uuid
    else: 
        return None


def reset_project_state():
    keys_to_delete = [
        "page",
        "current_frame_uuid",
        "which_number_for_starting_image",
        "rotated_image",
        "current_frame_index",
        "prev_frame_index",
        "zoom_level_input",
        "rotation_angle_input",
        "x_shift",
        "y_shift",
        "working_image",
        "degrees_rotated_to",
        "degree",
        "edited_image",
        "index_of_type_of_mask_selection",
        "type_of_mask_replacement",
        "which_layer",
        "which_layer_index",
        "drawing_input",
        "image_created",
        "precision_cropping_inpainted_image_uuid",
        "frame_styling_view_type",
        "transformation_stage",
        "custom_pipeline",
        "index_of_last_custom_pipeline",
        "index_of_controlnet_adapter_type",
        "lora_model_1",
        "lora_model_2",
        "lora_model_3",
        "index_of_lora_model_1",
        "index_of_lora_model_2",
        "index_of_lora_model_3",
        "custom_models",
        "adapter_type",
        "low_threshold",
        "high_threshold",
        "model",
        "prompt",
        "strength",
        "guidance_scale",
        "seed",
        "num_inference_steps",
        "dreambooth_model_uuid",
        "seed",
        "promote_new_generation",
        "use_new_settings",
    ]

    for k in keys_to_delete:
        if k in st.session_state:
            del st.session_state[k]

    # numbered keys
    numbered_keys_to_delete = [
        'animation_style_index_',
        'animation_style_'
    ]

    # TODO: remove hardcoded 20, find a better way to clear numbered state
    for i in range(20):
        for k in numbered_keys_to_delete:
            key = k + str(i)
            if key in st.session_state:
                del st.session_state[key]


    # reset cache
    StCache.clear_entire_cache()


def reset_styling_settings(timing_uuid):
    keys_to_delete = [
        f"index_of_which_stage_to_run_on_{timing_uuid}",
        "index_of_default_model",
        "index_of_controlnet_adapter_type",
        "index_of_dreambooth_model",
        f'prompt_value_{timing_uuid}',
        "negative_prompt_value",
    ]

    for k in keys_to_delete:
        if k in st.session_state:
            del st.session_state[k]


def is_process_active(custom_process_name):
    # this caching assumes that the runner won't interupt or break once started
    if custom_process_name + "_process_state" in st.session_state and st.session_state[custom_process_name + "_process_state"]:
        return True

    try:
        ps_output = subprocess.check_output(["ps", "aux"]).decode("utf-8")
        if custom_process_name in ps_output:
            st.session_state[custom_process_name + "_process_state"] = True
            return True
    except subprocess.CalledProcessError:
        return False

    return False