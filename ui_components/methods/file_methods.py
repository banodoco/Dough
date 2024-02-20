import base64
from io import BytesIO
import io
import json
import os
import mimetypes
import random
import shutil
import string
import tempfile
from typing import Union
from urllib.parse import urlparse
import zipfile
from PIL import Image
import numpy as np
import uuid
from dotenv import set_key, get_key
import requests
import streamlit as st
from shared.constants import SERVER, InternalFileType, ServerType
from ui_components.models import InternalFileObject
from utils.data_repo.data_repo import DataRepo


# depending on the environment it will either save or host the PIL image object
def save_or_host_file(file, path, mime_type='image/png', dim=None):
    data_repo = DataRepo()
    # TODO: fix session state management, remove direct access out side the main code
    if dim:
        width, height = dim[0], dim[1]
    elif 'project_uuid' in st.session_state and st.session_state['project_uuid']:
        project_setting = data_repo.get_project_setting(st.session_state['project_uuid'])
        width, height = project_setting.width, project_setting.height

    if width and height:
        file = zoom_and_crop(file, width, height)
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

def zoom_and_crop(file, width, height):
    if file.width == width and file.height == height:
        return file
    
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

# resizes file dimensions to current project_settings
def normalize_size_internal_file_obj(file_obj: InternalFileObject, **kwargs):
    if not file_obj or file_obj.type != InternalFileType.IMAGE.value or not file_obj.project:
        return file_obj
    
    data_repo = DataRepo()

    if 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        project_setting = data_repo.get_project_setting(file_obj.project.uuid)
        dim = (project_setting.width, project_setting.height)

    create_new_file = True if 'create_new_file' in kwargs \
        and kwargs['create_new_file'] else False

    if create_new_file:
        file_obj = create_duplicate_file(file_obj)
    
    pil_file = generate_pil_image(file_obj.location)
    uploaded_url = save_or_host_file(pil_file, file_obj.location, mime_type='image/png', dim=dim)
    if uploaded_url:
        data_repo = DataRepo()
        data_repo.update_file(file_obj.uuid, hosted_url=uploaded_url)
    
    return file_obj

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

def add_temp_file_to_project(project_uuid, key, file_path):
    data_repo = DataRepo()

    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if file_path.startswith('http'):
        file_data.update({'hosted_url': file_path})
    else:
        file_data.update({'local_path': file_path})

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

def convert_bytes_to_file(file_location_to_save, mime_type, file_bytes, project_uuid, inference_log_id=None, filename=None, tag="") -> InternalFileObject:
    data_repo = DataRepo()

    hosted_url = save_or_host_file_bytes(file_bytes, file_location_to_save, "." + mime_type.split("/")[1])
    file_data = {
        "name": str(uuid.uuid4()) + "." + mime_type.split("/")[1] if not filename else filename,
        "type": InternalFileType.VIDEO.value if 'video' in mime_type else (InternalFileType.AUDIO.value if 'audio' in mime_type else InternalFileType.IMAGE.value),
        "project_id": project_uuid,
        "tag": tag
    }

    if inference_log_id:
        file_data.update({'inference_log_id': str(inference_log_id)})

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_location_to_save})

    file = data_repo.create_file(**file_data)

    return file

def convert_file_to_base64(fh: io.IOBase) -> str:
    fh.seek(0)

    b = fh.read()
    if isinstance(b, str):
        b = b.encode("utf-8")
    encoded_body = base64.b64encode(b)
    if getattr(fh, "name", None):
        mime_type = mimetypes.guess_type(fh.name)[0]  # type: ignore
    else:
        mime_type = "application/octet-stream"
    s = encoded_body.decode("utf-8")
    return f"data:{mime_type};base64,{s}"

def resize_io_buffers(io_buffer, target_width, target_height, format="PNG"):
    input_image = Image.open(io_buffer)
    input_image = input_image.resize((target_width, target_height), Image.ANTIALIAS)
    output_image_buffer = io.BytesIO()
    input_image.save(output_image_buffer, format='PNG')
    return output_image_buffer

ENV_FILE_PATH = '.env'
def save_to_env(key, value):
    set_key(dotenv_path=ENV_FILE_PATH, key_to_set=key, value_to_set=value)

def load_from_env(key):
    val = get_key(dotenv_path=ENV_FILE_PATH, key_to_get=key)
    return val

import zipfile
import os
import requests
from PIL import Image
from io import BytesIO

def zip_images(image_locations, zip_filename='images.zip', filename_list=[]):
    # Calculate the number of digits needed for padding
    num_digits = len(str(len(image_locations) - 1))

    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for idx, image_location in enumerate(image_locations):
            # Pad the index with zeros
            padded_idx = str(idx).zfill(num_digits)
            if filename_list and len(filename_list) > idx:
                image_name = filename_list[idx]
            else:
                image_name = f"{padded_idx}.png"

            if image_location.startswith('http'):
                response = requests.get(image_location)
                image_data = response.content

                # Open the image for inspection and possible conversion
                with Image.open(BytesIO(image_data)) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save the potentially converted image to a byte stream
                    with BytesIO() as output:
                        img.save(output, format='PNG')
                        zip_file.writestr(image_name, output.getvalue())
            else:
                # For local files, open, possibly convert, and then add to zip
                with Image.open(image_location) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img.save(image_name, format='PNG')
                    zip_file.write(image_name, image_name)
                    os.remove(image_name)  # Clean up the temporary file

    return zip_filename

def create_duplicate_file(file: InternalFileObject, project_uuid=None) -> InternalFileObject:
    data_repo = DataRepo()

    unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    file_data = {
        "name": unique_id + '_' + file.name,
        "type": file.type,
    }

    if file.hosted_url:
        file_data.update({'hosted_url': file.hosted_url})
    
    if file.local_path:
        file_data.update({'local_path': file.local_path})

    if file.project:
        file_data.update({'project_id': file.project.uuid})
    elif project_uuid:
        file_data.update({'project_id': project_uuid})

    if file.tag:
        file_data.update({'tag': file.tag})

    if file.inference_log:
        file_data.update({'inference_log_id': str(file.inference_log.uuid)})

    new_file = data_repo.create_file(**file_data)
    return new_file


def copy_local_file(filepath, destination_directory, new_name):
    try:
        os.makedirs(destination_directory, exist_ok=True)
        new_filepath = os.path.join(destination_directory, new_name)
        shutil.copy2(filepath, new_filepath)
    except Exception as e:
        print("error occured: ", e)