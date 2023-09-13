from pathlib import Path
import os
import csv
import streamlit as st
import json
from shared.constants import SERVER, ServerType
from ui_components.models import InternalUserObject
from utils.constants import LOGGED_USER
from utils.data_repo.data_repo import DataRepo

# creates a file path if it's not already present
def create_file_path(path):
    if not path:
        return
    
    file = Path(path)
    if not file.is_file():
        last_slash_index = path.rfind('/')
        if last_slash_index != -1:
                directory_path = path[:last_slash_index]
                file_name = path[last_slash_index + 1:]
                
                # creating directory if not present
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
        else:
            directory_path = './'
            file_name = path

        # creating file
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'w') as f:
            pass
        
        # adding columns/rows in the file
        if file_name == 'timings.csv':
            data = [
                ['frame_time', 'frame_number', 'primary_image', 'alternative_images', 'custom_pipeline', 'negative_prompt', 'guidance_scale', 'seed', 'num_inference_steps',
                      'model_id', 'strength', 'notes', 'source_image', 'custom_models', 'adapter_type', 'clip_duration', 'interpolated_video', 'timed_clip', 'prompt', 'mask'],
            ]
        elif file_name == 'settings.csv':
            data = [
                ['key', 'value'],
                ['last_prompt', ''],
                ['default_model', 'controlnet'],
                ['last_strength', '0.5'],
                ['last_custom_pipeline', 'None'],
                ['audio', ''],
                ['input_type', 'Video'],
                ['input_video', ''],
                ['extraction_type', 'Regular intervals'],
                ['width', '704'],
                ['height', '512'],
                ['last_negative_prompt', '"nudity,  boobs, breasts, naked, nsfw"'],
                ['last_guidance_scale', '7.5'],
                ['last_seed', '0'],
                ['last_num_inference_steps', '100'],
                ['last_which_stage_to_run_on', 'Current Main Variants'],
                ['last_custom_models', '[]'],
                ['last_adapter_type', 'normal']
            ]
        elif file_name == 'app_settings.csv':
            data = [
                ['key', 'value'],
                ['replicate_com_api_key', ''],
                ['aws_access_key_id', ''],
                ['aws_secret_access_key', ''],
                ['previous_project', ''],
                ['replicate_username', ''],
                ['welcome_state', '0']
            ]
        elif file_name == 'log.csv':
            data = [
                ['model_name', 'model_version', 'total_inference_time', 'input_params', 'created_on'],
            ]

        
        if len(data):
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(data)

def copy_sample_assets(project_name):
    import shutil

    # copy sample video
    source = "sample_assets/sample_videos/sample.mp4"
    dest = "videos/" + project_name + "/assets/resources/input_videos/sample.mp4"
    shutil.copyfile(source, dest)

def create_working_assets(project_name):
    if SERVER != ServerType.DEVELOPMENT.value:
        return

    new_project = True
    if os.path.exists("videos/"+project_name):
        new_project = False

    directory_list = [
        # project specific files
        "videos/" + project_name,
        "videos/" + project_name + "/assets",
        "videos/" + project_name + "/assets/frames",
        "videos/" + project_name + "/assets/frames/0_extracted",
        "videos/" + project_name + "/assets/frames/1_selected",
        "videos/" + project_name + "/assets/frames/2_character_pipeline_completed",
        "videos/" + project_name + "/assets/frames/3_backdrop_pipeline_completed",
        "videos/" + project_name + "/assets/resources",
        "videos/" + project_name + "/assets/resources/backgrounds",
        "videos/" + project_name + "/assets/resources/masks",
        "videos/" + project_name + "/assets/resources/audio",
        "videos/" + project_name + "/assets/resources/input_videos",
        "videos/" + project_name + "/assets/resources/prompt_images",
        "videos/" + project_name + "/assets/videos",
        "videos/" + project_name + "/assets/videos/0_raw",
        "videos/" + project_name + "/assets/videos/1_final",
        "videos/" + project_name + "/assets/videos/2_completed",
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
        copy_sample_assets(project_name)

    csv_file_list = [
        f'videos/{project_name}/settings.csv',
        f'videos/{project_name}/timings.csv',
        'inference_log/log.csv'
    ]

    for csv_file in csv_file_list:
        create_file_path(csv_file)

# fresh_fetch - bypasses the cache
def get_current_user(fresh_fetch=False) -> InternalUserObject:
    # changing the code to operate on streamlit state rather than local file
    if not LOGGED_USER in st.session_state or fresh_fetch:
        data_repo = DataRepo()
        user = data_repo.get_first_active_user()
        st.session_state[LOGGED_USER] = user.to_json() if user else None
    
    return json.loads(st.session_state[LOGGED_USER]) if LOGGED_USER in st.session_state else None

def user_credits_available():
    current_user = get_current_user(fresh_fetch=True)
    return True if (current_user and current_user['total_credits'] > 0) else False

def get_current_user_uuid():
    current_user = get_current_user()
    if current_user and 'uuid' in current_user:
        return current_user['uuid']
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

