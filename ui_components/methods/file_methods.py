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
import time
from typing import Union
from urllib.parse import urlparse
import zipfile
from PIL import Image
import cv2
import numpy as np
import uuid
from moviepy.editor import VideoFileClip
from dotenv import set_key, get_key
import requests
from shared.constants import SERVER, InternalFileTag, InternalFileType, ServerType
from ui_components.models import InternalFileObject
from utils.data_repo.data_repo import DataRepo


# depending on the environment it will either save or host the PIL image object
def save_or_host_file(file, path, mime_type="image/png", dim=None):
    import streamlit as st
    data_repo = DataRepo()
    uploaded_url = None
    file_type, file_ext = mime_type.split("/")
    try:
        if file_type == "image":
            # TODO: fix session state management, remove direct access outside the main code
            if dim:
                width, height = dim[0], dim[1]
            elif "project_uuid" in st.session_state and st.session_state["project_uuid"]:
                project_setting = data_repo.get_project_setting(st.session_state["project_uuid"])
                width, height = project_setting.width, project_setting.height
            else:
                # Default dimensions for new project
                width, height = 512, 512
            # Apply zoom and crop based on determined dimensions
            file = zoom_and_crop(file, width, height)

        if SERVER != ServerType.DEVELOPMENT.value:
            file_bytes = BytesIO()
            file.save(file_bytes, format=file_ext)
            file_bytes.seek(0)

            uploaded_url = data_repo.upload_file(file_bytes, "." + file_ext)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if file_type == "image":
                file.save(path)
            else:
                with open(path, "wb") as f:
                    f.write(file.read())

    except Exception as e:
        # Log the error. You can replace 'print' with logging to a file or external logging service.
        print(f"Error saving or hosting file: {e}")
        # Optionally, re-raise the exception if you want the calling code to handle it
        # raise e

    return uploaded_url


def zoom_and_crop(file, width, height):
            
    if file.width == width and file.height == height:
        return file
    
    # Calculate the scaling factors
    scale_w = width / file.width
    scale_h = height / file.height
    
    # Use the larger scaling factor to ensure the image fills the target dimensions
    scale = max(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(file.width * scale)
    new_height = int(file.height * scale)
    
    # Resize the image
    resized_image = file.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate coordinates for cropping
    left = (resized_image.width - width) / 2
    top = (resized_image.height - height) / 2
    right = (resized_image.width + width) / 2
    bottom = (resized_image.height + height) / 2
    
    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image


# resizes file dimensions to current project_settings
def normalize_size_internal_file_obj(file_obj: InternalFileObject, **kwargs):
    if not file_obj or file_obj.type != InternalFileType.IMAGE.value or not file_obj.project:
        return file_obj

    data_repo = DataRepo()

    if "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        project_setting = data_repo.get_project_setting(file_obj.project.uuid)
        dim = (project_setting.width, project_setting.height)

    create_new_file = True if "create_new_file" in kwargs and kwargs["create_new_file"] else False

    if create_new_file:
        file_obj = create_duplicate_file(file_obj)

    pil_file = generate_pil_image(file_obj.location)
    uploaded_url = save_or_host_file(pil_file, file_obj.location, mime_type="image/png", dim=dim)
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
        with open(path, "wb") as f:
            f.write(video_bytes)

    return uploaded_url


def add_temp_file_to_project(project_uuid, key, file_path):
    data_repo = DataRepo()

    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid,
    }

    if file_path.startswith("http"):
        file_data.update({"hosted_url": file_path})
    else:
        file_data.update({"local_path": file_path})

    temp_file = data_repo.create_file(**file_data)
    project = data_repo.get_project_from_uuid(project_uuid)
    temp_file_list = project.project_temp_file_list
    temp_file_list.update({key: temp_file.uuid})
    temp_file_list = json.dumps(temp_file_list)
    project_data = {"uuid": project_uuid, "temp_file_list": temp_file_list}
    data_repo.update_project(**project_data)


def generate_temp_file(url, ext=".mp4"):
    response = requests.get(url)
    if not response.ok:
        raise ValueError(f"Could not download video from URL: {url}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode="wb")
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
            "Invalid image input. Must be a PIL image, a URL string, a local file path string or a numpy ndarray."
        )

    return img


def generate_temp_file_from_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file


def convert_bytes_to_file(
    file_location_to_save, mime_type, file_bytes, project_uuid, inference_log_id=None, filename=None, tag=""
) -> InternalFileObject:
    data_repo = DataRepo()

    hosted_url = save_or_host_file_bytes(file_bytes, file_location_to_save, "." + mime_type.split("/")[1])
    file_data = {
        "name": str(uuid.uuid4()) + "." + mime_type.split("/")[1] if not filename else filename,
        "type": (
            InternalFileType.VIDEO.value
            if "video" in mime_type
            else (InternalFileType.AUDIO.value if "audio" in mime_type else InternalFileType.IMAGE.value)
        ),
        "project_id": project_uuid,
        "tag": tag,
    }

    if inference_log_id:
        file_data.update({"inference_log_id": str(inference_log_id)})

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": file_location_to_save})

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
    input_image.save(output_image_buffer, format="PNG")
    return output_image_buffer


ENV_FILE_PATH = ".env"


def save_to_env(key, value):
    set_key(dotenv_path=ENV_FILE_PATH, key_to_set=key, value_to_set=value)


def load_from_env(key):
    try:
        val = get_key(dotenv_path=ENV_FILE_PATH, key_to_get=key)
        return val
    except Exception as e:
        return None


def delete_from_env(key_to_delete):
    with open(ENV_FILE_PATH, "r") as f:
        lines = f.readlines()

    with open(ENV_FILE_PATH, "w") as f:
        for line in lines:
            if not line.startswith(f"{key_to_delete}="):
                f.write(line)


def zip_images(image_locations, zip_filename="images.zip", filename_list=[]):
    # Calculate the number of digits needed for padding
    num_digits = len(str(len(image_locations) - 1))

    with zipfile.ZipFile(zip_filename, "w") as zip_file:
        for idx, image_location in enumerate(image_locations):
            # Pad the index with zeros
            padded_idx = str(idx).zfill(num_digits)
            if filename_list and len(filename_list) > idx:
                image_name = filename_list[idx]
            else:
                image_name = f"{padded_idx}.png"

            if image_location.startswith("http"):
                response = requests.get(image_location)
                image_data = response.content

                # Open the image for inspection and possible conversion
                with Image.open(BytesIO(image_data)) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Save the potentially converted image to a byte stream
                    with BytesIO() as output:
                        img.save(output, format="PNG")
                        zip_file.writestr(image_name, output.getvalue())
            else:
                # For local files, open, possibly convert, and then add to zip
                with Image.open(image_location) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    img.save(image_name, format="PNG")
                    zip_file.write(image_name, image_name)
                    os.remove(image_name)  # Clean up the temporary file

    return zip_filename


def create_duplicate_file(file: InternalFileObject, project_uuid=None) -> InternalFileObject:
    """
    this creates a duplicate InternalFileobject in the db, the actual file on the disk/or url
    remains the same
    """
    data_repo = DataRepo()

    unique_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    file_data = {
        "name": unique_id + "_" + file.name,
        "type": file.type,
    }

    if file.hosted_url:
        file_data.update({"hosted_url": file.hosted_url})

    if file.local_path:
        file_data.update({"local_path": file.local_path})

    if file.project:
        file_data.update({"project_id": file.project.uuid})
    elif project_uuid:
        file_data.update({"project_id": project_uuid})

    if file.tag:
        file_data.update({"tag": file.tag})

    if file.inference_log:
        file_data.update({"inference_log_id": str(file.inference_log.uuid)})

    new_file = data_repo.create_file(**file_data)
    return new_file


def copy_local_file(filepath, destination_directory, new_name):
    try:
        os.makedirs(destination_directory, exist_ok=True)
        new_filepath = os.path.join(destination_directory, new_name)
        shutil.copy2(filepath, new_filepath)
    except Exception as e:
        pass


def determine_dimensions_for_sdxl(width, height):
    if width == height:
        # Square aspect ratio
        return 1024, 1024
    elif width > height:
        # Landscape orientation
        aspect_ratio = width / height
        # Select the size based on the aspect ratio thresholds for landscape orientations
        if aspect_ratio >= 1.6:
            return 1536, 640
        elif aspect_ratio >= 1.4:
            return 1344, 768
        elif aspect_ratio >= 1.3:
            return 1216, 832
        else:
            return 1152, 896
    else:
        # Portrait orientation
        aspect_ratio = height / width
        # Select the size based on the aspect ratio thresholds for portrait orientations
        if aspect_ratio >= 1.6:
            return 640, 1536
        elif aspect_ratio >= 1.4:
            return 768, 1344
        elif aspect_ratio >= 1.3:
            return 832, 1216
        else:
            return 896, 1152


def list_files_in_folder(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            files.append(file)
    return files


def get_file_bytes_and_extension(path_or_url):
    try:
        if urlparse(path_or_url).scheme:
            # URL
            response = requests.get(path_or_url)
            response.raise_for_status()  # non-2xx responses
            file_bytes = response.content
            parsed_url = urlparse(path_or_url)
            filename, file_extension = os.path.splitext(parsed_url.path)
            file_extension = file_extension.lstrip(".")
        else:
            # Local file path
            with open(path_or_url, "rb") as file:
                file_bytes = file.read()
            filename, file_extension = os.path.splitext(path_or_url)
            file_extension = file_extension.lstrip(".")

        return file_bytes, file_extension
    except Exception as e:
        print("Error:", e)
        return None, None


# adds a white border around the polygon to minimize irregularities
def detect_and_draw_contour(image):
    # Convert PIL Image to OpenCV format (BGR)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Convert image to grayscale
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary mask
    th, im_th = cv2.threshold(g, 220, 250, cv2.THRESH_BINARY_INV)

    # Morphological opening to separate each component and have delimited edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    im_th2 = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)

    # Connected Components segmentation
    maxLabels, labels = cv2.connectedComponents(im_th2)

    # Iterate through each component and find its contour
    for label in range(1, maxLabels):
        # Create a mask for the current component
        mask = np.uint8(labels == label) * 255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate the contours with straight lines
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Draw contours on the original image with white color
        cv2.drawContours(img_bgr, [approx], -1, (255, 255, 255), 2)

    # Convert the modified image back to PIL format (RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(img_rgb)

    return output_image


def get_file_size(file_path):
    file_size = 0
    if file_path.startswith("http://") or file_path.startswith("https://"):
        response = requests.head(file_path)
        if response.status_code == 200:
            file_size = int(response.headers.get("content-length", 0))
        else:
            print("Failed to fetch file from URL:", file_path)
    else:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        else:
            print("File does not exist:", file_path)

    return int(file_size / (1024 * 1024))


def get_media_dimensions(media_path):
    try:
        if media_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_clip = VideoFileClip(media_path)
            width = video_clip.size[0]
            height = video_clip.size[1]
            return width, height
        else:
            with Image.open(media_path) as img:
                width, height = img.size
                return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None


# fetches all the files (including subfolders) in a directory
def get_files_in_a_directory(directory, ext_list=[]):
    res = []

    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if ext_list and len(ext_list) and file.split(".")[-1] in ext_list:
                    res.append(file)  # (os.path.join(root, file))

    return res


def add_file_to_shortlist(file_uuid, project_uuid=None):
    data_repo = DataRepo()
    file: InternalFileObject = data_repo.get_file_from_uuid(file_uuid)

    project_uuid = project_uuid or file.project.uuid
    duplicate_file = create_duplicate_file(file, project_uuid)
    data_repo.update_file(
        duplicate_file.uuid,
        tag=InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value,
    )


def compress_image(file_path, max_size_kb=1024, min_quality=60, max_quality=95, step=5):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with Image.open(file_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_size_kb = os.path.getsize(file_path) / 1024
        if original_size_kb <= max_size_kb:
            print(f"File is already smaller than {max_size_kb}KB. No compression needed.")
            return file_path
        
        low, high = min_quality, max_quality
        best_quality = max_quality
        best_buffer = None
        
        while low <= high:
            mid = (low + high) // 2
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=mid, optimize=True)
            size = buffer.tell() / 1024  # Size in KB
            
            if size <= max_size_kb:
                best_quality = mid
                best_buffer = buffer
                low = mid + step
            else:
                high = mid - step
        
        if best_buffer is None:
            print("Could not compress the image to the desired size. Saving with minimum quality.")
            best_quality = min_quality
            best_buffer = io.BytesIO()
            img.save(best_buffer, format='JPEG', quality=best_quality, optimize=True)
        
        best_buffer.seek(0)
        with open(file_path, 'wb') as f:
            f.write(best_buffer.getvalue())
        
        final_size_kb = os.path.getsize(file_path) / 1024
        
        return file_path