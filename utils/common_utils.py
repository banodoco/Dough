from pathlib import Path
import os
import csv
import subprocess
import time
import psutil
import socket
import streamlit as st
import json
import platform
from shared.constants import SERVER, ServerType
from ui_components.models import InternalUserObject
from utils.cache.cache import CacheKey, StCache
from utils.data_repo.data_repo import DataRepo
from ui_components.constants import DefaultProjectSettingParams

def set_default_values(shot_uuid):
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    if "page" not in st.session_state:
        st.session_state['page'] = "Timeline"

    if "strength" not in st.session_state:
        st.session_state['strength'] = DefaultProjectSettingParams.batch_strength
        st.session_state['prompt_value'] = DefaultProjectSettingParams.batch_prompt
        st.session_state['model'] = None
        st.session_state['negative_prompt_value'] = DefaultProjectSettingParams.batch_negative_prompt
        st.session_state['guidance_scale'] = DefaultProjectSettingParams.batch_guidance_scale
        st.session_state['seed'] = DefaultProjectSettingParams.batch_seed
        st.session_state['num_inference_steps'] = DefaultProjectSettingParams.batch_num_inference_steps
        st.session_state['transformation_stage'] = DefaultProjectSettingParams.batch_transformation_stage
        
    if "current_frame_uuid" not in st.session_state and len(timing_list) > 0:
        timing = timing_list[0]
        st.session_state['current_frame_uuid'] = timing.uuid
        st.session_state['current_frame_index'] = timing.aux_frame_index + 1
    
    if 'frame_styling_view_type' not in st.session_state:
        st.session_state['frame_styling_view_type'] = "Generate"
        st.session_state['frame_styling_view_type_index'] = 0

    if "explorer_view" not in st.session_state:
        st.session_state['explorer_view'] = "Explorations"
        st.session_state['explorer_view_index'] = 0

    if "shot_view" not in st.session_state:
        st.session_state['shot_view'] = "Animate Frames"
        st.session_state['shot_view_index'] = 0
    
    if "styling_view" not in st.session_state:
        st.session_state['styling_view'] = "Generate"
        st.session_state['styling_view_index'] = 0


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


def get_current_user(invalidate_cache=False) -> InternalUserObject:
    data_repo = DataRepo()
    user = data_repo.get_first_active_user(invalidate_cache=invalidate_cache)
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
        "shot_uuid",
        "maintain_state"
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
        f"frame_styling_stage_index_{timing_uuid}",
        "index_of_default_model",
        "index_of_controlnet_adapter_type",
        "index_of_dreambooth_model",
        f'prompt_value_{timing_uuid}',
        "negative_prompt_value",
    ]

    for k in keys_to_delete:
        if k in st.session_state:
            del st.session_state[k]


def is_process_active(custom_process_name, custom_process_port):
    # This caching assumes that the runner won't interrupt or break once started
    cache_key = custom_process_name + "_process_state"
    if cache_key in st.session_state and st.session_state[cache_key]:
        return True

    res = False
    try:
        if platform.system() == "Windows":
            try:
                client_socket = socket.create_connection(("localhost", custom_process_port))
                client_socket.close()
                res = True
            except ConnectionRefusedError:
                res = False
        else:
            # Use 'ps' for Unix/Linux
            ps_output = subprocess.check_output(["ps", "aux"]).decode("utf-8")
            res = True if custom_process_name in ps_output else False
            
        if res:
            st.session_state[cache_key] = True
        return res
    except subprocess.CalledProcessError:
        return False

    # If the process is not found or an error occurs, assume it's not active
    return False


def acquire_lock(key):
    data_repo = DataRepo()
    retries = 0
    while retries < 1:
        lock_status = data_repo.acquire_lock(key)
        if lock_status:
            return lock_status
        retries += 1
        time.sleep(0.2)
    return False

def release_lock(key):
    data_repo = DataRepo()
    data_repo.release_lock(key)
    return True


def refresh_app(maintain_state=False):
    # st.session_state['maintain_state'] = maintain_state
    st.rerun()


def padded_integer(integer, pad_length=4):
    padded_string = str(integer).zfill(pad_length)
    return padded_string
