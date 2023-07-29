from io import BytesIO
from pathlib import Path
import os
import csv
from typing import Union
import uuid
import streamlit as st
import json
from shared.constants import SERVER, InternalFileType, ServerType
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from PIL import Image
import numpy as np

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
    source = "sample_assets/input_videos/sample.mp4"
    dest = "videos/" + project_name + "/assets/resources/input_videos/sample.mp4"
    shutil.copyfile(source, dest)

    # copy selected frames
    select_samples_path = 'sample_assets/frames/selected_sample'
    file_list = os.listdir(select_samples_path)
    file_paths = []
    for item in file_list:
        item_path = os.path.join(select_samples_path, item)
        if os.path.isfile(item_path):
            file_paths.append(item_path)
    
    for idx in range(len(file_list)):
        source = file_paths[idx]
        dest = f"videos/{project_name}/assets/frames/1_selected/{file_list[idx]}"
        shutil.copyfile(source, dest)
    
    # copy timings file
    source = "sample_assets/frames/meta_data/timings.csv"
    dest = f"videos/{project_name}/timings.csv"
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

def get_current_user():
    logger = AppLogger()
    # changing the code to operate on streamlit state rather than local file
    if not LOGGED_USER in st.session_state:
        data_repo = DataRepo()
        user = data_repo.get_first_active_user()
        st.session_state[LOGGED_USER] = user.to_json() if user else None
    
    return json.loads(st.session_state[LOGGED_USER]) if LOGGED_USER in st.session_state else None

def get_current_user_uuid():
    current_user = get_current_user()
    if current_user and 'uuid' in current_user:
        return current_user['uuid']
    else: 
        return None

# depending on the environment it will either save or host the PIL image object
def save_or_host_file(file, path):
    uploaded_url = None
    mime_type = file.type
    if SERVER != ServerType.DEVELOPMENT.value:
        image_bytes = BytesIO()
        file.save(image_bytes, format=mime_type.split('/')[1])
        image_bytes.seek(0)

        data_repo = DataRepo()
        uploaded_url = data_repo.upload_file(image_bytes)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file.save(path)

    return uploaded_url

def add_temp_file_to_project(project_uuid, key, hosted_url):
    data_repo = DataRepo()

    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid,
        'hosted_url': hosted_url
    }

    temp_file = data_repo.create_file(**file_data)
    project = data_repo.get_project_from_uuid(project_uuid)
    temp_file_list = project.project_temp_file_list
    temp_file_list.update({key: temp_file.uuid})
    temp_file_list = json.dumps(temp_file_list)
    project_data = {
        'uuid': project_uuid,
        'temp_file_list': temp_file_list
    }
    data_repo.update_project(**project_data)