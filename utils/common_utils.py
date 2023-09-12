from io import BytesIO
import io
from pathlib import Path
import os
import csv
import tempfile
from typing import Union
from urllib.parse import urlparse
import uuid
import requests
import streamlit as st
import json
from shared.constants import SERVER, InternalFileType, ServerType
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from PIL import Image
import numpy as np
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

    # copy selected frames
    # select_samples_path = 'sample_assets/sample_images'
    # file_list = os.listdir(select_samples_path)
    # file_paths = []
    # for item in file_list:
    #     item_path = os.path.join(select_samples_path, item)
    #     if os.path.isfile(item_path):
    #         file_paths.append(item_path)
    
    # for idx in range(len(file_list)):
    #     source = file_paths[idx]
    #     dest = f"videos/{project_name}/assets/frames/1_selected/{file_list[idx]}"
    #     shutil.copyfile(source, dest)
    
    # copy timings file
    # source = "sample_assets/frames/meta_data/timings.csv"
    # dest = f"videos/{project_name}/timings.csv"
    # shutil.copyfile(source, dest)

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

def zoom_and_crop(file, width, height):
    # scaling
    s_x = width / file.width
    s_y = height / file.height
    scale = max(s_x, s_y)
    new_width = int(file.width * scale)
    new_height = int(file.height * scale)
    file = file.resize((new_width, new_height))

    # cropping
    left = (file.width - width) // 2
    top = (file.height - height) // 2
    right = (file.width + width) // 2
    bottom = (file.height + height) // 2
    file = file.crop((left, top, right, bottom))

    return file


# depending on the environment it will either save or host the PIL image object
def save_or_host_file(file, path, mime_type='image/png'):
    data_repo = DataRepo()
    # TODO: fix session state management, remove direct access out side the main code
    project_setting = data_repo.get_project_setting(st.session_state['project_uuid'])
    if project_setting:
        file = zoom_and_crop(file, project_setting.width, project_setting.height)
    else:
        # new project
        file = zoom_and_crop(file, 512, 512)

    uploaded_url = None
    if SERVER != ServerType.DEVELOPMENT.value:
        image_bytes = BytesIO()
        file.save(image_bytes, format=mime_type.split('/')[1])
        image_bytes.seek(0)

        uploaded_url = data_repo.upload_file(image_bytes, '.png')
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file.save(path)

    return uploaded_url

def save_or_host_file_bytes(video_bytes, path, ext=".mp4"):
    uploaded_url = None
    if SERVER != ServerType.DEVELOPMENT.value:
        data_repo = DataRepo()
        uploaded_url = data_repo.upload_file(video_bytes, ext)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(video_bytes)
    
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


def generate_temp_file(url, ext=".mp4"):
    response = requests.get(url)
    if not response.ok:
        raise ValueError(f"Could not download video from URL: {url}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='wb')
    temp_file.write(response.content)
    temp_file.close()

    return temp_file

def generate_pil_image(img: Union[Image.Image, str, np.ndarray, io.BytesIO]):
    # Check if img is a PIL image
    if isinstance(img, Image.Image):
        pass

    # Check if img is a URL
    elif isinstance(img, str) and bool(urlparse(img).netloc):
        response = requests.get(img)
        img = Image.open(BytesIO(response.content))

    # Check if img is a local file
    elif isinstance(img, str):
        img = Image.open(img)

    # Check if img is a numpy ndarray
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Check if img is a BytesIO stream
    elif isinstance(img, io.BytesIO):
        img = Image.open(img)

    else:
        raise ValueError(
            "Invalid image input. Must be a PIL image, a URL string, a local file path string or a numpy ndarray.")

    return img

def generate_temp_file_from_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file

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


def convert_bytes_to_file(file_location_to_save, mime_type, file_bytes, project_uuid):
    data_repo = DataRepo()

    hosted_url = save_or_host_file_bytes(file_bytes, file_location_to_save, mime_type=mime_type)
    file_data = {
        "name": str(uuid.uuid4()),
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid.uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_location_to_save})

    file = data_repo.create_file(**file_data)

    return file