import io
from typing import List
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter, ImageChops
from moviepy.editor import *
from requests_toolbelt.multipart.encoder import MultipartEncoder
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import csv
import pandas as pd
import replicate
import urllib
import requests as r
import imageio
import ffmpeg
import string
import math
import json
import tempfile
import boto3
import time
import zipfile
from math import cos, sin, ceil, radians, gcd
import random
import uuid
from io import BytesIO
from st_clickable_images import clickable_images
import ast
import numpy as np
from shared.constants import REPLICATE_USER, SERVER, AIModelType, InternalFileTag, InternalFileType, ServerType
from pydub import AudioSegment
import shutil
from moviepy.editor import concatenate_videoclips, TextClip, VideoFileClip, vfx
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from backend.models import InternalFileObject
from shared.file_upload.s3 import upload_file
from shared.utils import is_online_file_path
from ui_components.constants import CROPPED_IMG_LOCAL_PATH, MASK_IMG_LOCAL_PATH, SECOND_MASK_FILE, SECOND_MASK_FILE_PATH, TEMP_MASK_FILE, VideoQuality, WorkflowStageType
from ui_components.models import InternalAIModelObject, InternalAppSettingObject, InternalBackupObject, InternalFrameTimingObject, InternalProjectObject, InternalSettingObject
from utils.common_utils import add_temp_file_to_project, generate_temp_file, get_current_user_uuid, save_or_host_file, save_or_host_file_bytes
from utils.data_repo.data_repo import DataRepo
from shared.constants import InternalResponse, AnimationStyleType
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import DEFAULT_LORA_MODEL_URL, REPLICATE_MODEL
from ui_components.models import InternalFileObject
import utils.local_storage.local_storage as local_storage


from utils import st_memory
from urllib.parse import urlparse

from typing import Union
from moviepy.video.fx.all import speedx
import moviepy.editor
from streamlit_cropper import st_cropper
from htbuilder import details, div, p, styles, summary
from streamlit_image_comparison import image_comparison


def clone_styling_settings(source_frame_number, target_frame_uuid):
    data_repo = DataRepo()
    target_timing = data_repo.get_timing_from_uuid(target_frame_uuid)
    timing_details = data_repo.get_timing_list_from_project(
        target_timing.project.uuid)

    data_repo.update_specific_timing(
        target_frame_uuid, custom_pipeline=timing_details[source_frame_number].custom_pipeline)
    data_repo.update_specific_timing(
        target_frame_uuid, negative_prompt=timing_details[source_frame_number].negative_prompt)
    data_repo.update_specific_timing(
        target_frame_uuid, guidance_scale=timing_details[source_frame_number].guidance_scale)
    data_repo.update_specific_timing(
        target_frame_uuid, seed=timing_details[source_frame_number].seed)
    data_repo.update_specific_timing(
        target_frame_uuid, num_inteference_steps=timing_details[source_frame_number].num_inteference_steps)
    data_repo.update_specific_timing(
        target_frame_uuid, transformation_stage=timing_details[source_frame_number].transformation_stage)
    if timing_details[source_frame_number].model:
        data_repo.update_specific_timing(
            target_frame_uuid, model_id=timing_details[source_frame_number].model.uuid)
    data_repo.update_specific_timing(
        target_frame_uuid, strength=timing_details[source_frame_number].strength)
    data_repo.update_specific_timing(
        target_frame_uuid, custom_models=timing_details[source_frame_number].custom_model_id_list)
    data_repo.update_specific_timing(
        target_frame_uuid, adapter_type=timing_details[source_frame_number].adapter_type)
    data_repo.update_specific_timing(
        target_frame_uuid, low_threshold=timing_details[source_frame_number].low_threshold)
    data_repo.update_specific_timing(
        target_frame_uuid, high_threshold=timing_details[source_frame_number].high_threshold)
    data_repo.update_specific_timing(
        target_frame_uuid, prompt=timing_details[source_frame_number].prompt)


def prompt_finder_element(project_uuid):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("What image would you like to find the prompt for?", type=[
                                     'png', 'jpg', 'jpeg'], key="prompt_file")
    which_model = st.radio("Which model would you like to get a prompt for?", ["Stable Diffusion 1.5", "Stable Diffusion 2"], key="which_model",
                           help="This is to know which model we should optimize the prompt for. 1.5 is usually best if you're in doubt", horizontal=True)
    best_or_fast = st.radio("Would you like to optimize for best quality or fastest speed?", [
                            "Best", "Fast"], key="best_or_fast", help="This is to know whether we should optimize for best quality or fastest speed. Best quality is usually best if you're in doubt", horizontal=True).lower()
    if st.button("Get prompts"):
        if not uploaded_file:
            return
        
        uploaded_file_path = f"videos/{project_uuid}/assets/resources/prompt_images/{uploaded_file.name}"
        img = Image.open(uploaded_file)
        hosted_url = save_or_host_file(img, uploaded_file_path)
        uploaded_file_path = hosted_url or uploaded_file_path

        prompt = prompt_clip_interrogator(uploaded_file_path, which_model, best_or_fast)
        
        # if not os.path.exists(f"videos/{project_uuid}/prompts.csv"):
        #     with open(f"videos/{project_uuid}/prompts.csv", "w") as f:
        #         f.write("prompt,example_image,which_model\n")
        # add the prompt to prompts.csv
        # with open(f"videos/{project_uuid}/prompts.csv", "a") as f:
        #     f.write(
        #         f'"{prompt}",videos/{project_uuid}/assets/resources/prompt_images/{uploaded_file.name},{which_model}\n')
        
        st.session_state["last_generated_prompt"] = prompt

        st.success("Prompt added successfully!")
        time.sleep(1)
        uploaded_file = ""
        st.experimental_rerun()
    
    if 'last_generated_prompt' in st.session_state and st.session_state['last_generated_prompt']:
        st.write("Generated prompt - ", st.session_state['last_generated_prompt'])
    
    # list all the prompts in prompts.csv
    # if os.path.exists(f"videos/{project_uuid}/prompts.csv"):

    #     df = pd.read_csv(f"videos/{project_uuid}/prompts.csv", na_filter=False)
    #     prompts = df.to_dict('records')

    #     prompts.reverse()

    #     col1, col2 = st.columns([1.5, 1])
    #     with col1:
    #         st.markdown("### Prompt")
    #     with col2:
    #         st.markdown("### Example Image")
    #     with open(f"videos/{project_uuid}/prompts.csv", "r") as f:
    #         for i in prompts:
    #             index_of_current_item = prompts.index(i)
    #             col1, col2 = st.columns([1.5, 1])
    #             with col1:
    #                 st.write(prompts[index_of_current_item]["prompt"])
    #             with col2:
    #                 st.image(prompts[index_of_current_item]
    #                          ["example_image"], use_column_width=True)
    #             st.markdown("***")


# TODO: image format is assumed to be PNG, change this later
def save_new_image(img: Union[Image.Image, str, np.ndarray, io.BytesIO], project_uuid) -> InternalFileObject:

    
    # Check if img is a PIL image
    if isinstance(img, Image.Image):
        pass

    # Check if img is a URL
    elif isinstance(img, str) and bool(urlparse(img).netloc):
        response = r.get(img)
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
    
    file_name = str(uuid.uuid4()) + ".png"
    file_path = os.path.join("videos/temp", file_name)

    hosted_url = save_or_host_file(img, file_path)

    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_path})

    data_repo = DataRepo()
    new_image = data_repo.create_file(**file_data)
    return new_image



def save_uploaded_image(uploaded_file, project_uuid, frame_uuid, save_type):
    data_repo = DataRepo()

    print(uploaded_file)
    
    # Check if uploaded_file is already an opened PIL Image
    if isinstance(uploaded_file, Image.Image):
        image_to_save = uploaded_file
    else:
        try:
            image_to_save = Image.open(uploaded_file)
        except Exception as e:
            print(f"Failed to open image due to: {str(e)}")
            return None

    try:
        # Save new image
        saved_image = save_new_image(image_to_save, project_uuid)

        # Update records based on save_type
        if save_type == "source":
            data_repo.update_specific_timing(frame_uuid, source_image_id=saved_image.uuid)
        elif save_type == "styled":
            number_of_image_variants = add_image_variant(saved_image.uuid, frame_uuid)
            promote_image_variant(frame_uuid, number_of_image_variants - 1)

        return saved_image
    except Exception as e:
        print(f"Failed to save image file due to: {str(e)}")
        return None


def create_individual_clip(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if timing.animation_style == "":
        project_setting = data_repo.get_project_setting(
            timing.project.uuid)
        animation_style = project_setting.default_animation_style
    else:
        animation_style = timing.animation_style

    if animation_style == AnimationStyleType.INTERPOLATION.value:
        output_video = prompt_interpolation_model(timing_uuid)

    elif animation_style == AnimationStyleType.DIRECT_MORPHING.value:
        output_video = create_video_without_interpolation(timing_uuid)

    return output_video


'''
returns a video generated through interpolating frames between the current frame
and the next frame
'''


def prompt_interpolation_model(timing_uuid) -> InternalFileType:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    img1 = timing.primary_image_location
    next_timing = data_repo.get_next_timing(timing_uuid)
    img2 = next_timing.primary_image_location

    if not img1.startswith("http"):
        img1 = open(img1, "rb")

    if not img2.startswith("http"):
        img2 = open(img2, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.google_frame_interpolation, frame1=img1, frame2=img2,
                                            times_to_interpolate=timing.interpolation_steps)
    file_name = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=16)) + ".mp4"

    video_location = "videos/" + timing.project.uuid + \
        "/assets/videos/0_raw/" + str(file_name)
    try:
        urllib.request.urlretrieve(output, video_location)
    except Exception as e:
        print(e)

    video_file = data_repo.create_file(name=file_name, type=InternalFileType.VIDEO.value,
                                       hosted_url=output, local_path=video_location, project_id=timing.project.uuid,
                                       tag=InternalFileTag.GENERATED_VIDEO.value)

    return video_file


def create_video_without_interpolation(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    image_path_or_url = timing.primary_image_location

    video_location = "videos/" + timing.project.uuid + "/assets/videos/0_raw/" + \
                     ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k=16)) + ".mp4"

    os.makedirs(os.path.dirname(video_location), exist_ok=True)

    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        response = r.get(image_path_or_url)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path_or_url)

    if image is None:
        raise ValueError(
            "Could not read the image. Please provide a valid image path or URL.")

    height, width, _ = image.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = int(1 / 0.1)
    video_writer = cv2.VideoWriter(
        video_location, fourcc, fps, (width, height))

    for _ in range(fps):
        video_writer.write(image)

    video_writer.release()

    unique_file_name = str(uuid.uuid4())
    file_data = {
        "name": unique_file_name,
        "type": InternalFileType.VIDEO.value,
        "local_path": video_location,
        "tag": InternalFileTag.GENERATED_VIDEO.value
    }

    video_file: InternalFileObject = data_repo.create_file(**file_data)

    return video_file


def get_pillow_image(image_location):
    if image_location.startswith("http://") or image_location.startswith("https://"):
        response = r.get(image_location)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_location)

    return image


def create_alpha_mask(size, edge_blur_radius):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)

    width, height = size
    draw.rectangle((edge_blur_radius, edge_blur_radius, width -
                   edge_blur_radius, height - edge_blur_radius), fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_blur_radius))
    return mask

# returns a PIL Image object
def zoom_image(image, zoom_factor, fill_with=None):
    blur_radius = 5
    edge_blur_radius = 15

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0")

    # Check if the provided location is a URL
    # if location.startswith('http') or location.startswith('https'):
    #     response = r.get(location)
    #     image = Image.open(BytesIO(response.content))
    # else:
    #     if not os.path.exists(location):
    #         raise FileNotFoundError(f"File not found: {location}")
    #     image = Image.open(location)

    # Calculate new dimensions based on zoom factor
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    if zoom_factor < 1:
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        if fill_with == "Blur":
            blurred_image = image.filter(
                ImageFilter.GaussianBlur(radius=blur_radius))

            # Resize the blurred image to match the original dimensions
            blurred_background = blurred_image.resize(
                (width, height), Image.ANTIALIAS)

            # Create an alpha mask for blending
            alpha_mask = create_alpha_mask(
                resized_image.size, edge_blur_radius)

            # Blend the resized image with the blurred background using the alpha mask
            blended_image = Image.composite(resized_image, blurred_background.crop(
                (0, 0, new_width, new_height)), alpha_mask)

            # Calculate the position to paste the blended image at the center of the blurred background
            paste_left = (blurred_background.width - blended_image.width) // 2
            paste_top = (blurred_background.height - blended_image.height) // 2

            # Create a new blank image with the size of the blurred background
            final_image = Image.new('RGBA', blurred_background.size)

            # Paste the blurred background onto the final image
            final_image.paste(blurred_background, (0, 0))

            # Paste the blended image onto the final image using the alpha mask
            final_image.paste(blended_image, (paste_left,
                              paste_top), mask=alpha_mask)

            return final_image

        elif fill_with == "Inpainting":
            print("Coming soon")
            return resized_image

        elif fill_with is None:
            # Create an empty background with the original dimensions
            background = Image.new('RGBA', (width, height))

            # Calculate the position to paste the resized image at the center of the background
            paste_left = (background.width - resized_image.width) // 2
            paste_top = (background.height - resized_image.height) // 2

            # Paste the resized image onto the background
            background.paste(resized_image, (paste_left, paste_top))

            return background

        else:
            raise ValueError(
                "Invalid fill_with value. Accepted values are 'Blur', 'Inpainting', and None.")

    else:
        # If zooming in, proceed as before
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        left = (resized_image.width - width) / 2
        top = (resized_image.height - height) / 2
        right = (resized_image.width + width) / 2
        bottom = (resized_image.height + height) / 2

        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image

# image here is a PIL object


def apply_image_transformations(image, zoom_level, rotation_angle, x_shift, y_shift):
    width, height = image.size

    # Calculate the diagonal for the rotation
    diagonal = math.ceil(math.sqrt(width**2 + height**2))

    # Create a new image with black background for rotation
    rotation_bg = Image.new("RGB", (diagonal, diagonal), "black")
    rotation_offset = ((diagonal - width) // 2, (diagonal - height) // 2)
    rotation_bg.paste(image, rotation_offset)

    # Rotation
    rotated_image = rotation_bg.rotate(rotation_angle)

    # Shift
    # Create a new image with black background
    shift_bg = Image.new("RGB", (diagonal, diagonal), "black")
    shift_bg.paste(rotated_image, (x_shift, y_shift))

    # Zoom
    zoomed_width = int(diagonal * (zoom_level / 100))
    zoomed_height = int(diagonal * (zoom_level / 100))
    zoomed_image = shift_bg.resize((zoomed_width, zoomed_height))

    # Crop the zoomed image back to original size
    crop_x1 = (zoomed_width - width) // 2
    crop_y1 = (zoomed_height - height) // 2
    crop_x2 = crop_x1 + width
    crop_y2 = crop_y1 + height
    cropped_image = zoomed_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return cropped_image


def fetch_image_by_stage(project_uuid, stage):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    if stage == WorkflowStageType.SOURCE.value:
        return timing_details[st.session_state['current_frame_index'] - 1].source_image
    elif stage == WorkflowStageType.STYLED.value:
        return timing_details[st.session_state['current_frame_index'] - 1].primary_image
    else:
        return None


def zoom_inputs(project_settings, position='in-frame', horizontal=False):
    
    if horizontal:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1 = col2 = col3 = col4 = st

    zoom_level_input = col1.number_input(
        "Zoom Level (%)", min_value=10, max_value=1000, step=10, key=f"zoom_level_input_key_{position}", value=st.session_state.get('zoom_level_input', 100))
    
    rotation_angle_input = col2.number_input(
        "Rotation Angle", min_value=-360, max_value=360, step=5, key=f"rotation_angle_input_key_{position}", value=st.session_state.get('rotation_angle_input', 0))
    
    x_shift = col3.number_input(
        "Shift Left/Right", min_value=-1000, max_value=1000, step=5, key=f"x_shift_key_{position}", value=st.session_state.get('x_shift', 0))
    
    y_shift = col4.number_input(
        "Shift Up/Down", min_value=-1000, max_value=1000, step=5, key=f"y_shift_key_{position}", value=st.session_state.get('y_shift', 0))

    # Assign values to st.session_state
    st.session_state['zoom_level_input'] = zoom_level_input
    st.session_state['rotation_angle_input'] = rotation_angle_input
    st.session_state['x_shift'] = x_shift
    st.session_state['y_shift'] = y_shift








def save_zoomed_image(image, timing_uuid, stage, promote=False):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.project.uuid

    file_name = str(uuid.uuid4()) + ".png"

    if stage == WorkflowStageType.SOURCE.value:
        save_location = f"videos/{project_uuid}/assets/frames/1_selected/{file_name}"
        hosted_url = save_or_host_file(image, save_location)
        file_data = {
            "name": file_name,
            "type": InternalFileType.IMAGE.value,
            "project_id": project_uuid
        }

        if hosted_url:
            file_data.update({'hosted_url': hosted_url})
        else:
            file_data.update({'local_path': save_location})

        source_image: InternalFileObject = data_repo.create_file(**file_data)
        data_repo.update_specific_timing(
            st.session_state['current_frame_uuid'], source_image_id=source_image.uuid)
    elif stage == WorkflowStageType.STYLED.value:
        save_location = f"videos/{project_uuid}/assets/frames/2_character_pipeline_completed/{file_name}"
        hosted_url = save_or_host_file(image, save_location)
        file_data = {
            "name": file_name,
            "type": InternalFileType.IMAGE.value,
            "project_id": project_uuid
        }

        if hosted_url:
            file_data.update({'hosted_url': hosted_url})
        else:
            file_data.update({'local_path': save_location})
            
        styled_image: InternalFileObject = data_repo.create_file(**file_data)

        number_of_image_variants = add_image_variant(
            styled_image.uuid, timing_uuid)
        if promote:
            promote_image_variant(timing_uuid, number_of_image_variants - 1)

    project_update_data = {
        "zoom_level": st.session_state['zoom_level_input'],
        "rotation_angle_value": st.session_state['rotation_angle_input'],
        "x_shift": st.session_state['x_shift'],
        "y_shift": st.session_state['y_shift']
    }

    data_repo.update_project_setting(project_uuid, **project_update_data)

    # TODO: CORRECT-CODE - make a proper column for zoom details
    timing_update_data = {
        "zoom_details": f"{st.session_state['zoom_level_input']},{st.session_state['rotation_angle_input']},{st.session_state['x_shift']},{st.session_state['y_shift']}",

    }
    data_repo.update_specific_timing(timing_uuid, **timing_update_data)

def reset_zoom_element():
    st.session_state['zoom_level_input_key'] = 100
    st.session_state['rotation_angle_input_key'] = 0
    st.session_state['x_shift_key'] = 0
    st.session_state['y_shift_key'] = 0
    st.session_state['zoom_level_input'] = 100
    st.session_state['rotation_angle_input'] = 0
    st.session_state['x_shift'] = 0
    st.session_state['y_shift'] = 0
    st.experimental_rerun()



def precision_cropping_element(stage, project_uuid):
    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        project_uuid)

    
    input_image = fetch_image_by_stage(project_uuid, stage)

    # TODO: CORRECT-CODE check if this code works
    if not input_image:
        st.error("Please select a source image before cropping")
        return
    else:
        input_image = get_pillow_image(input_image.location)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Precision Cropping:")

        if st.button("Reset Cropping"):
            reset_zoom_element()
        
        
        zoom_inputs(project_settings)
        st.caption("Input Image:")
        st.image(input_image, caption="Input Image", width=300)

    with col2:

        st.caption("Output Image:")
        output_image = apply_image_transformations(
            input_image, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'])
        st.image(output_image, use_column_width=True)

        if st.button("Save Image"):
            save_zoomed_image(output_image, st.session_state['current_frame_uuid'], stage, promote=True)
            st.success("Image saved successfully!")
            time.sleep(1)
            st.experimental_rerun()

        inpaint_in_black_space_element(
            output_image, project_settings.project.uuid, stage)


def manual_cropping_element(stage, timing_uuid):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.project.uuid

    if not timing.source_image:
        st.error("Please select a source image before cropping")
        return
    else:
        if stage == WorkflowStageType.SOURCE.value:
            input_image = timing.source_image.location
        elif stage == WorkflowStageType.STYLED.value:
            input_image = timing.primary_image_location

        if 'current_working_image_number' not in st.session_state:
            st.session_state['current_working_image_number'] = st.session_state['current_frame_index']

        def get_working_image():
            st.session_state['working_image'] = get_pillow_image(input_image)
            st.session_state['working_image'] = ImageOps.expand(
                st.session_state['working_image'], border=200, fill="black")
            st.session_state['current_working_image_number'] = st.session_state['current_frame_index']

        if 'working_image' not in st.session_state or st.session_state['current_working_image_number'] != st.session_state['current_frame_index']:
            get_working_image()

        options1, options2, option3, option4 = st.columns([3, 1, 1, 1])
        with options1:
            sub_options_1, sub_options_2 = st.columns(2)
            if 'degrees_rotated_to' not in st.session_state:
                st.session_state['degrees_rotated_to'] = 0
            with sub_options_1:
                st.session_state['degree'] = st.slider(
                    "Rotate Image", -180, 180, value=st.session_state['degrees_rotated_to'])
                if st.session_state['degrees_rotated_to'] != st.session_state['degree']:
                    get_working_image()
                    st.session_state['working_image'] = st.session_state['working_image'].rotate(
                        -st.session_state['degree'], resample=Image.BICUBIC, expand=True)
                    st.session_state['degrees_rotated_to'] = st.session_state['degree']
                    st.experimental_rerun()

            with sub_options_2:
                st.write("")
                st.write("")
                if st.button("Reset image"):
                    st.session_state['degree'] = 0
                    get_working_image()
                    st.session_state['degrees_rotated_to'] = 0
                    st.experimental_rerun()

        

        project_settings: InternalProjectObject = data_repo.get_project_setting(
            timing.project.uuid)

        width = project_settings.width
        height = project_settings.height

        gcd_value = gcd(width, height)
        aspect_ratio_width = int(width // gcd_value)
        aspect_ratio_height = int(height // gcd_value)
        aspect_ratio = (aspect_ratio_width, aspect_ratio_height)

        img1, img2 = st.columns([3, 1.5])

        with img1:
            # use PIL to add 50 pixels of blackspace to the width and height of the image
            cropped_img = st_cropper(
                st.session_state['working_image'], realtime_update=True, box_color="#0000FF", aspect_ratio=aspect_ratio)

        with img2:
            st.image(cropped_img, caption="Cropped Image",
                     use_column_width=True, width=200)

            cropbtn1, cropbtn2 = st.columns(2)
            with cropbtn1:
                if st.button("Save Cropped Image"):
                    if stage == WorkflowStageType.SOURCE.value:
                        # resize the image to the original width and height
                        cropped_img = cropped_img.resize(
                            (width, height), Image.ANTIALIAS)
                        # generate a random filename and save it to /temp
                        file_path = f"videos/temp/{uuid.uuid4()}.png"
                        hosted_url = save_or_host_file(cropped_img, file_path)
                        
                        file_data = {
                            "name": str(uuid.uuid4()),
                            "type": InternalFileType.IMAGE.value,
                            "project_id": project_uuid
                        }

                        if hosted_url:
                            file_data.update({'hosted_url': hosted_url})
                        else:
                            file_data.update({'local_path': file_path})
                        cropped_image: InternalFileObject = data_repo.create_file(**file_data)

                        st.success("Cropped Image Saved Successfully")
                        data_repo.update_specific_timing(
                            st.session_state['current_frame_uuid'], source_image_id=cropped_image.uuid)
                        time.sleep(1)
                    st.experimental_rerun()
            with cropbtn2:
                st.warning("Warning: This will overwrite the original image")

            inpaint_in_black_space_element(
                cropped_img, timing.project.uuid, stage=stage)


def ai_frame_editing_element(timing_uuid, stage=WorkflowStageType.SOURCE.value):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    if len(timing_details) == 0:
        st.info("You need to add  key frames first in the Key Frame Selection section.")
    else:
        main_col_1, main_col_2 = st.columns([1, 2])

        with main_col_1:
            st.write("")

        # initiative value
        if "current_frame_uuid" not in st.session_state:
            st.session_state['current_frame_uuid'] = timing_details[0].uuid

        def reset_new_image():
            st.session_state['edited_image'] = ""

        if "edited_image" not in st.session_state:
            st.session_state.edited_image = ""

        if stage == WorkflowStageType.STYLED.value and len(timing.alternative_images_list) == 0:
            st.info("You need to add a style first in the Style Selection section.")
        else:
            if stage == WorkflowStageType.SOURCE.value:
                editing_image = timing.source_image.location
            elif stage == WorkflowStageType.STYLED.value:
                variants = timing.alternative_images_list
                editing_image = timing.primary_image_location

            width = int(project_settings.width)
            height = int(project_settings.height)

            if editing_image == "":
                st.error(
                    f"You don't have a {stage} image yet so you can't edit it.")
            else:
                with main_col_1:
                    if 'index_of_type_of_mask_selection' not in st.session_state:
                        st.session_state['index_of_type_of_mask_selection'] = 0
                    mask_selection_options = ["Manual Background Selection", "Automated Background Selection",
                                              "Automated Layer Selection", "Re-Use Previous Mask", "Invert Previous Mask"]
                    type_of_mask_selection = st.radio("How would you like to select what to edit?", mask_selection_options,
                                                      horizontal=True, index=st.session_state['index_of_type_of_mask_selection'])
                    if st.session_state['index_of_type_of_mask_selection'] != mask_selection_options.index(type_of_mask_selection):
                        st.session_state['index_of_type_of_mask_selection'] = mask_selection_options.index(
                            type_of_mask_selection)
                        st.experimental_rerun()

                    if "which_layer" not in st.session_state:
                        st.session_state['which_layer'] = "Background"
                        st.session_state['which_layer_index'] = 0

                    if type_of_mask_selection == "Automated Layer Selection":
                        layers = ["Background", "Middleground", "Foreground"]
                        st.session_state['which_layer'] = st.multiselect(
                            "Which layers would you like to replace?", layers)

                if type_of_mask_selection == "Manual Background Selection":
                    if st.session_state['edited_image'] == "":
                        with main_col_1:
                            if editing_image.startswith("http"):
                                canvas_image = r.get(editing_image)
                                canvas_image = Image.open(
                                    BytesIO(canvas_image.content))
                            else:
                                canvas_image = Image.open(editing_image)
                            if 'drawing_input' not in st.session_state:
                                st.session_state['drawing_input'] = 'Magic shapes 🪄'
                            col1, col2 = st.columns([6, 3])

                            with col1:
                                st.session_state['drawing_input'] = st.radio(
                                    "Drawing tool:",
                                    ("Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Make squares □", "Draw lines ✏️"), horizontal=True,
                                )

                            if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                                drawing_mode = "transform"
                                st.info(
                                    "To delete something, just move it outside of the image! 🥴")
                            elif st.session_state['drawing_input'] == "Make shapes 🪄":
                                drawing_mode = "polygon"
                                st.info("To end a shape, right click!")
                            elif st.session_state['drawing_input'] == "Draw lines ✏️":
                                drawing_mode = "freedraw"
                                st.info("To draw, draw! ")
                            elif st.session_state['drawing_input'] == "Make squares □":
                                drawing_mode = "rect"

                            with col2:
                                if drawing_mode == "freedraw":
                                    stroke_width = st.slider(
                                        "Stroke width: ", 1, 25, 12)
                                else:
                                    stroke_width = 3

                        with main_col_2:

                            realtime_update = True

                            canvas_result = st_canvas(
                                fill_color="rgba(0, 0, 0)",
                                stroke_width=stroke_width,
                                stroke_color="rgba(0, 0, 0)",
                                background_color="rgb(255, 255, 255)",
                                background_image=canvas_image,
                                update_streamlit=realtime_update,
                                height=height,
                                width=width,
                                drawing_mode=drawing_mode,
                                display_toolbar=True,
                                key="full_app",
                            )

                            if 'image_created' not in st.session_state:
                                st.session_state['image_created'] = 'no'

                            if canvas_result.image_data is not None:
                                img_data = canvas_result.image_data
                                im = Image.fromarray(
                                    img_data.astype("uint8"), mode="RGBA")
                                create_or_update_mask(
                                    st.session_state['current_frame_uuid'], im)
                    else:
                        image_file = data_repo.get_file_from_uuid(st.session_state['edited_image'])
                        image_comparison(
                            img1=editing_image,
                            img2=image_file.location, starting_position=5, label1="Original", label2="Edited")
                        if st.button("Reset Canvas"):
                            st.session_state['edited_image'] = ""
                            st.experimental_rerun()

                elif type_of_mask_selection == "Automated Background Selection" or type_of_mask_selection == "Automated Layer Selection" or type_of_mask_selection == "Re-Use Previous Mask" or type_of_mask_selection == "Invert Previous Mask":
                    with main_col_1:
                        if type_of_mask_selection in ["Re-Use Previous Mask", "Invert Previous Mask"]:
                            if not timing_details[st.session_state['current_frame_index'] - 1].mask:
                                st.info(
                                    "You don't have a previous mask to re-use.")
                            else:
                                mask1, mask2 = st.columns([2, 1])
                                with mask1:
                                    if type_of_mask_selection == "Re-Use Previous Mask":
                                        st.info(
                                            "This will update the **black pixels** in the mask with the pixels from the image you are editing.")
                                    elif type_of_mask_selection == "Invert Previous Mask":
                                        st.info(
                                            "This will update the **white pixels** in the mask with the pixels from the image you are editing.")
                                    st.image(
                                        timing_details[st.session_state['current_frame_index'] - 1].mask.location, use_column_width=True)

                    with main_col_2:
                        if st.session_state['edited_image'] == "":
                            st.image(editing_image, use_column_width=True)
                        else:
                            image_file = data_repo.get_file_from_uuid(st.session_state['edited_image'])
                            image_comparison(
                                img1=editing_image,
                                img2=image_file.location, starting_position=5, label1="Original", label2="Edited")
                            if st.button("Reset Canvas"):
                                st.session_state['edited_image'] = ""
                                st.experimental_rerun()

                with main_col_1:

                    if "type_of_mask_replacement" not in st.session_state:
                        st.session_state["type_of_mask_replacement"] = "Replace With Image"
                        st.session_state["index_of_type_of_mask_replacement"] = 0

                    types_of_mask_replacement = [
                        "Inpainting", "Replace With Image"]
                    st.session_state["type_of_mask_replacement"] = st.radio(
                        "Select type of edit", types_of_mask_replacement, horizontal=True, index=st.session_state["index_of_type_of_mask_replacement"])

                    if st.session_state["index_of_type_of_mask_replacement"] != types_of_mask_replacement.index(st.session_state["type_of_mask_replacement"]):
                        st.session_state["index_of_type_of_mask_replacement"] = types_of_mask_replacement.index(
                            st.session_state["type_of_mask_replacement"])
                        st.experimental_rerun()

                    if st.session_state["type_of_mask_replacement"] == "Replace With Image":
                        prompt = ""
                        negative_prompt = ""
                        background_list = [f for f in os.listdir(
                            f'videos/{timing.project.uuid}/assets/resources/backgrounds') if f.endswith('.png')]
                        background_list = [f for f in os.listdir(
                            f'videos/{timing.project.uuid}/assets/resources/backgrounds') if f.endswith('.png')]
                        sources_of_images = ["Uploaded", "From Other Frame"]
                        if 'index_of_source_of_image' not in st.session_state:
                            st.session_state['index_of_source_of_image'] = 0
                        source_of_image = st.radio("Select type of image", sources_of_images,
                                                   horizontal=True, index=st.session_state['index_of_source_of_image'])

                        if st.session_state['index_of_source_of_image'] != sources_of_images.index(source_of_image):
                            st.session_state['index_of_source_of_image'] = sources_of_images.index(
                                source_of_image)
                            st.experimental_rerun()

                        if source_of_image == "Uploaded":
                            btn1, btn2 = st.columns([1, 1])
                            with btn1:
                                uploaded_files = st.file_uploader(
                                    "Add more background images here", accept_multiple_files=True)
                                if st.button("Upload Backgrounds"):
                                    for uploaded_file in uploaded_files:
                                        with open(os.path.join(f"videos/{timing.project.uuid}/assets/resources/backgrounds", uploaded_file.name), "wb") as f:
                                            f.write(uploaded_file.getbuffer())
                                            st.success(
                                                "Your backgrounds are uploaded file - they should appear in the dropdown.")
                                            background_list.append(
                                                uploaded_file.name)
                                            time.sleep(1.5)
                                            st.experimental_rerun()
                            with btn2:
                                background_selection = st.selectbox(
                                    "Range background", background_list)
                                background_image = f'videos/{timing.project.uuid}/assets/resources/backgrounds/{background_selection}'
                                if background_list != []:
                                    st.image(f"{background_image}",
                                             use_column_width=True)
                        elif source_of_image == "From Other Frame":
                            btn1, btn2 = st.columns([1, 1])
                            with btn1:
                                which_stage_to_use = st.radio(
                                    "Select stage to use:", WorkflowStageType.value_list())
                                which_image_to_use = st.number_input(
                                    "Select image to use:", min_value=0, max_value=len(timing_details)-1, value=0)
                                if which_stage_to_use == WorkflowStageType.SOURCE.value:
                                    background_image = timing_details[which_image_to_use].source_image.location

                                elif which_stage_to_use == WorkflowStageType.STYLED.value:
                                    background_image = timing_details[which_image_to_use].primary_image_location
                            with btn2:
                                st.image(background_image,
                                         use_column_width=True)

                    elif st.session_state["type_of_mask_replacement"] == "Inpainting":
                        btn1, btn2 = st.columns([1, 1])
                        with btn1:
                            prompt = st.text_area("Prompt:", help="Describe the whole image, but focus on the details you want changed!",
                                                  value=project_settings.default_prompt)
                        with btn2:
                            negative_prompt = st.text_area(
                                "Negative Prompt:", help="Enter any things you want to make the model avoid!", value=project_settings.default_negative_prompt)

                    edit1, edit2 = st.columns(2)

                    with edit1:
                        if st.button(f'Run Edit On Current Image'):
                            if st.session_state["type_of_mask_replacement"] == "Inpainting":
                                edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"],
                                                                  "", editing_image, prompt, negative_prompt, width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                                st.session_state['edited_image'] = edited_image.uuid
                            elif st.session_state["type_of_mask_replacement"] == "Replace With Image":
                                edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"],
                                                                  background_image, editing_image, "", "", width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                                st.session_state['edited_image'] = edited_image.uuid
                            st.experimental_rerun()

                    with edit2:
                        if st.session_state['edited_image'] != "":
                            if st.button("Promote Last Edit", type="primary"):
                                if stage == WorkflowStageType.SOURCE.value:
                                    data_repo.update_specific_timing(
                                        st.session_state['current_frame_uuid'], source_image_id=st.session_state['edited_image'])
                                elif stage == WorkflowStageType.STYLED.value:
                                    number_of_image_variants = add_image_variant(
                                        st.session_state['edited_image'], st.session_state['current_frame_uuid'])
                                    promote_image_variant(
                                        st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                                st.session_state['edited_image'] = ""
                                st.experimental_rerun()
                        else:
                            if st.button("Run Edit & Promote"):
                                if st.session_state["type_of_mask_replacement"] == "Inpainting":
                                    edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"],
                                                                      "", editing_image, prompt, negative_prompt, width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                                    st.session_state['edited_image'] = edited_image.uuid
                                elif st.session_state["type_of_mask_replacement"] == "Replace With Image":
                                    edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"],
                                                                      background_image, editing_image, "", "", width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                                    st.session_state['edited_image'] = edited_image.uuid

                                if stage == WorkflowStageType.SOURCE.value:
                                    data_repo.update_specific_timing(
                                        st.session_state['current_frame_uuid'], source_image_id=st.session_state['edited_image'])
                                elif stage == WorkflowStageType.STYLED.value:
                                    number_of_image_variants = add_image_variant(
                                        edited_image.uuid, st.session_state['current_frame_uuid'])
                                    promote_image_variant(
                                        st.session_state['current_frame_uuid'], number_of_image_variants - 1)

                                st.session_state['edited_image'] = ""
                                st.success("Image promoted!")
                                st.experimental_rerun()


def save_image_by_stage(status):
    st.write("")

# cropped_img here is a PIL image object
def inpaint_in_black_space_element(cropped_img, project_uuid, stage=WorkflowStageType.SOURCE.value):
    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        project_uuid)

    st.markdown("##### Inpaint in black space:")

    inpaint_prompt = st.text_area(
        "Prompt", value=project_settings.default_prompt)
    inpaint_negative_prompt = st.text_input(
        "Negative Prompt", value='edge,branches, frame, fractals, text' + project_settings.default_negative_prompt)
    if 'precision_cropping_inpainted_image_uuid' not in st.session_state:
        st.session_state['precision_cropping_inpainted_image_uuid'] = ""

    if st.button("Inpaint"):
        width = int(project_settings.width)
        height = int(project_settings.height)

        saved_cropped_img = cropped_img.resize(
            (width, height), Image.ANTIALIAS)
        
        hosted_cropped_img_path = save_or_host_file(saved_cropped_img, CROPPED_IMG_LOCAL_PATH)

        # Convert image to grayscale
        # Create a new image with the same size as the cropped image
        mask = Image.new('RGB', cropped_img.size)

        # Get the width and height of the image
        width, height = cropped_img.size

        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                pixel = cropped_img.getpixel((x, y))

                # If the image is RGB, unpack the pixel into r, g, and b
                if cropped_img.mode == 'RGB':
                    r, g, b = pixel
                # If the image is RGBA, unpack the pixel into r, g, b, and a
                elif cropped_img.mode == 'RGBA':
                    r, g, b, a = pixel
                # If the image is grayscale ('L' for luminosity), there's only one channel
                elif cropped_img.mode == 'L':
                    brightness = pixel
                else:
                    raise ValueError(
                        f'Unsupported image mode: {cropped_img.mode}')

                # If the pixel is black, set it and its adjacent pixels to black in the new image
                if r == 0 and g == 0 and b == 0:
                    mask.putpixel((x, y), (0, 0, 0))  # Black
                    # Adjust these values to change the range of adjacent pixels
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            # Check that the pixel is within the image boundaries
                            if 0 <= x + i < width and 0 <= y + j < height:
                                mask.putpixel((x + i, y + j),
                                              (0, 0, 0))  # Black
                # Otherwise, make the pixel white in the new image
                else:
                    mask.putpixel((x, y), (255, 255, 255))  # White
        # Save the mask image
        hosted_url = save_or_host_file(mask, MASK_IMG_LOCAL_PATH)
        if hosted_url:
            add_temp_file_to_project(project_uuid, TEMP_MASK_FILE, hosted_url)

        cropped_img_path = hosted_cropped_img_path if hosted_cropped_img_path else CROPPED_IMG_LOCAL_PATH
        inpainted_file = inpainting(cropped_img_path, inpaint_prompt,
                                    inpaint_negative_prompt, st.session_state['current_frame_uuid'], True, pass_mask=True)

        st.session_state['precision_cropping_inpainted_image_uuid'] = inpainted_file.uuid

    if st.session_state['precision_cropping_inpainted_image_uuid']:
        img_file = data_repo.get_file_from_uuid(
            st.session_state['precision_cropping_inpainted_image_uuid'])
        st.image(img_file.location, caption="Inpainted Image",
                 use_column_width=True, width=200)

        if stage == WorkflowStageType.SOURCE.value:
            if st.button("Make Source Image"):
                data_repo.update_specific_timing(
                    st.session_state['current_frame_uuid'], source_image_id=img_file.uuid)
                st.session_state['precision_cropping_inpainted_image_uuid'] = ""
                st.experimental_rerun()

        elif stage == WorkflowStageType.STYLED.value:
            if st.button("Save + Promote Image"):
                timing_details = data_repo.get_timing_list_from_project(
                    project_uuid)
                number_of_image_variants = add_image_variant(
                    st.session_state['precision_cropping_inpainted_image_uuid'], st.session_state['current_frame_uuid'])
                promote_image_variant(
                    st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                st.session_state['precision_cropping_inpainted_image_uuid'] = ""
                st.experimental_rerun()


# returns a PIL image object
def rotate_image(location, degree):
    if location.startswith('http') or location.startswith('https'):
        response = r.get(location)
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(f"File not found: {location}")
        image = Image.open(location)

    # Rotate the image by the specified degree
    rotated_image = image.rotate(-degree, resample=Image.BICUBIC, expand=False)

    return rotated_image


# returns the timed_clip, which is the interpolated video with correct length
def create_or_get_single_preview_video(timing_uuid):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_details: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    if not timing.interpolated_clip:
        data_repo.update_specific_timing(timing_uuid, interpolation_steps=3)
        interpolated_video: InternalFileObject = prompt_interpolation_model(
            timing_uuid)
        data_repo.update_specific_timing(
            timing_uuid, interpolated_clip_id=interpolated_video.uuid)

    if not timing.timed_clip:
        timing = data_repo.get_timing_from_uuid(timing_uuid)
        
        temp_video_file = None
        if timing.interpolated_clip.hosted_url:
            temp_video_file = generate_temp_file(timing.interpolated_clip.hosted_url, '.mp4')

        file_path = temp_video_file.name if temp_video_file else timing.interpolated_clip.local_path
        clip = VideoFileClip(file_path)
            
        number_text = TextClip(str(timing.aux_frame_index),
                               fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(
            number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(
            ('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position(
            (number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)
        clip_with_number = CompositeVideoClip(
            [clip, number_background, number_text])

        # clip_with_number.write_videofile(timing.interpolated_clip.local_path)
        temp_output_file = generate_temp_file(timing.interpolated_clip.hosted_url, '.mp4')
        clip_with_number.write_videofile(filename=temp_output_file.name, codec='libx264', audio_codec='aac')

        video_bytes = None
        with open(temp_output_file.name, 'rb') as f:
            video_bytes = f.read()

        hosted_url = save_or_host_file_bytes(video_bytes, timing.interpolated_clip.local_path)

        if hosted_url:
            data_repo.update_file(timing.interpolated_clip.uuid, hosted_url=hosted_url)

        if temp_video_file:
            os.remove(temp_video_file.name)

        # timed_clip has the correct length (equal to the time difference between the current and the next frame)
        # which the interpolated video may or maynot have
        clip_duration = calculate_desired_duration_of_individual_clip(
            timing_uuid)
        data_repo.update_specific_timing(
            timing_uuid, clip_duration=clip_duration)

        # TODO: fix refetching of variables
        timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
            timing_uuid)
        output_video = update_speed_of_video_clip(
            timing.interpolated_clip, True, timing_uuid)
        data_repo.update_specific_timing(
            timing_uuid, timed_clip_id=output_video.uuid)

    # adding audio if the audio file is present
    if project_details.audio:
        audio_bytes = get_audio_bytes_for_slice(timing_uuid)
        add_audio_to_video_slice(timing.timed_clip, audio_bytes)

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    return timing.timed_clip


'''
preview_clips have frame numbers on them. Preview clip is generated from index-2 to index+2 frames
'''


def create_full_preview_video(timing_uuid, speed=1) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    index_of_item = timing.aux_frame_index

    num_timing_details = len(timing_details)
    clips = []

    temp_file_list = []

    for i in range(index_of_item - 2, index_of_item + 3):

        if i < 0 or i >= num_timing_details-1:
            continue

        primary_variant_location = timing_details[i].primary_image_location

        print(
            f"primary_variant_location for i={i}: {primary_variant_location}")

        if not primary_variant_location:
            break

        preview_video = create_or_get_single_preview_video(
            timing_details[i].uuid)

        clip = VideoFileClip(preview_video.location)

        number_text = TextClip(str(i), fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(
            number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(
            ('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position(
            (number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)

        clip_with_number = CompositeVideoClip(
            [clip, number_background, number_text])

        # remove existing preview video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
        temp_file_list.append(temp_file)
        clip_with_number.write_videofile(temp_file.name, codec='libx264', bitrate='3000k')
        video_bytes = None
        with open(temp_file.name, 'rb') as f:
            video_bytes = f.read()

        hosted_url = save_or_host_file_bytes(video_bytes, preview_video.local_path)
        if hosted_url:
            data_repo.update_file(preview_video.uuid, hosted_url=hosted_url)
        
        clips.append(preview_video)

        # if preview_video.local_path:
        #     os.remove(preview_video.local_path)

    print(clips)
    video_clips = []

    for v in clips:
        path = v.location
        if 'http' in path:
            temp_file = generate_temp_file(path)
            temp_file_list.append(temp_file)
            path = temp_file.name
        
        video_clips.append(VideoFileClip(path))

    # video_clips = [VideoFileClip(v.location) for v in clips]
    combined_clip = concatenate_videoclips(video_clips)
    output_filename = str(uuid.uuid4()) + ".mp4"
    video_location = f"videos/{timing.project.uuid}/assets/videos/1_final/{output_filename}"

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
    combined_clip = combined_clip.fx(vfx.speedx, speed)
    combined_clip.write_videofile(temp_output_file.name)

    video_bytes = None
    with open(temp_output_file.name, 'rb') as f:
        video_bytes = f.read()
    
    hosted_url = save_or_host_file_bytes(video_bytes, video_location)

    video_data = {
        "name": output_filename,
        "type": InternalFileType.VIDEO.value,
        "project_id": timing.project.uuid
    }
    if hosted_url:
        video_data.update({"hosted_url": hosted_url})
    else:
        video_data.update({"local_path": video_location})

    # if speed != 1.0:
    #     clip = VideoFileClip(video_location)
    #     output_clip = clip.fx(vfx.speedx, speed)
    #     os.remove(video_location)
    #     output_clip.write_videofile(
    #         video_location, codec="libx264", preset="fast")

    for file in temp_file_list:
        os.remove(file.name)

    video_file = data_repo.create_file(**video_data)

    return video_file


def back_and_forward_buttons():
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        st.session_state['current_frame_uuid'])
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    smallbutton0, smallbutton1, smallbutton2, smallbutton3, smallbutton4 = st.columns([
                                                                                      2, 2, 2, 2, 2])

    display_idx = st.session_state['current_frame_index']
    with smallbutton0:
        if display_idx > 2:
            if st.button(f"{display_idx-2} ⏮️", key=f"Previous Previous Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] - 2
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.experimental_rerun()
    with smallbutton1:
        # if it's not the first image
        if display_idx != 1:
            if st.button(f"{display_idx-1} ⏪", key=f"Previous Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] - 1
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.experimental_rerun()

    with smallbutton2:
        st.button(f"{display_idx} 📍", disabled=True)
    with smallbutton3:
        # if it's not the last image
        if display_idx != len(timing_details):
            if st.button(f"{display_idx+1} ⏩", key=f"Next Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 1
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.experimental_rerun()
    with smallbutton4:
        if display_idx <= len(timing_details)-2:
            if st.button(f"{display_idx+2} ⏭️", key=f"Next Next Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 2
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.experimental_rerun()

# TODO: CORRECT-CODE


def display_image(timing_uuid, stage=None, clickable=False):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_idx = timing.aux_frame_index + 1

    # if it's less than 0 or greater than the number in timing_details, show nothing
    if not timing:
        st.write("")

    else:
        if stage == WorkflowStageType.STYLED.value:
            image = timing.primary_image_location
        elif stage == WorkflowStageType.SOURCE.value:
            image = timing.source_image.location if timing.source_image else ""

        if image != "":
            if clickable is True:
                if 'counter' not in st.session_state:
                    st.session_state['counter'] = 0

                import base64

                if image.startswith("http"):
                    st.write("")
                else:
                    with open(image, "rb") as image:
                        st.write("")
                        encoded = base64.b64encode(image.read()).decode()
                        image = (f"data:image/jpeg;base64,{encoded}")

                st.session_state[f'{timing_idx}_{stage}_clicked'] = clickable_images([image], div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"}, img_style={
                    "max-width": "100%", "height": "auto", "cursor": "pointer"}, key=f"{timing_idx}_{stage}_image_{st.session_state['counter']}")

                if st.session_state[f'{timing_idx}_{stage}_clicked'] == 0:
                    timing_details = data_repo.get_timing_list_from_project(timing.project.uuid)
                    st.session_state['current_frame_uuid'] = timing_details[timing_idx].uuid
                    st.session_state['current_frame_index'] = timing_idx
                    st.session_state['prev_frame_index'] = timing_idx
                    # st.session_state['frame_styling_view_type_index'] = 0
                    st.session_state['frame_styling_view_type'] = "Individual View"
                    st.session_state['counter'] += 1
                    st.experimental_rerun()

            elif clickable is False:
                st.image(image, use_column_width=True)
        else:
            st.error(f"No {stage} image found for #{timing_idx}")


def carousal_of_images_element(project_uuid, stage=WorkflowStageType.STYLED.value):
    
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    header1, header2, header3, header4, header5 = st.columns([1, 1, 1, 1, 1])

    current_timing = data_repo.get_timing_from_uuid(
        st.session_state['current_frame_uuid'])
    
    with header1:
        if current_timing.aux_frame_index - 2 >=0:
            prev_2_timing = data_repo.get_timing_from_frame_number(project_uuid,
                current_timing.aux_frame_index - 2)

            if prev_2_timing:
                display_image(prev_2_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{prev_2_timing.aux_frame_index + 1}")

    with header2:
        if current_timing.aux_frame_index - 1 >= 0:
            prev_timing = data_repo.get_timing_from_frame_number(project_uuid,
                current_timing.aux_frame_index - 1)
            if prev_timing:
                display_image(prev_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{prev_timing.aux_frame_index + 1}")

    with header3:
        
        timing = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
        display_image(timing.uuid,
                      stage=stage, clickable=True)
        st.success(f"#{current_timing.aux_frame_index + 1}")
    with header4:
        if current_timing.aux_frame_index + 1 <= len(timing_details):
            next_timing = data_repo.get_timing_from_frame_number(project_uuid,
                current_timing.aux_frame_index + 1)
            if next_timing:
                display_image(next_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{next_timing.aux_frame_index + 1}")

    with header5:
        if current_timing.aux_frame_index + 2 <= len(timing_details):
            next_2_timing = data_repo.get_timing_from_frame_number(project_uuid,
                current_timing.aux_frame_index + 2)
            if next_2_timing:
                display_image(next_2_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{next_2_timing.aux_frame_index + 1}")
    # st.markdown("***")


# TODO: CORRECT-CODE
def styling_element(timing_uuid, view_type="Single"):
    

    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    stages = ["Source Image", "Main Variant"]

    if project_settings.default_stage != "":
        if 'index_of_which_stage_to_run_on' not in st.session_state:
            st.session_state['transformation_stage'] = project_settings.default_stage
            st.session_state['index_of_which_stage_to_run_on'] = stages.index(
                st.session_state['transformation_stage'])
    else:
        st.session_state['index_of_which_stage_to_run_on'] = 0

    if view_type == "Single":
        append_to_item_name = f"{st.session_state['current_frame_index']}"
    elif view_type == "List":
        append_to_item_name = "bulk"
        st.markdown("## Batch queries")

    stages = ["Source Image", "Main Variant"]

    # TODO: CORRECT-CODE transformation_stage add this in db
    if view_type == "Single":
        if timing.transformation_stage:
            if f'index_of_which_stage_to_run_on_{append_to_item_name}' not in st.session_state:
                st.session_state['transformation_stage'] = timing.transformation_stage
                st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(
                    st.session_state['transformation_stage'])
        else:
            st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = 0

    elif view_type == "List":
        if project_settings.default_stage != "":
            if f'index_of_which_stage_to_run_on_{append_to_item_name}' not in st.session_state:
                st.session_state['transformation_stage'] = project_settings.default_stage
                st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(
                    st.session_state['transformation_stage'])
        else:
            st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = 0

    stages1, stages2 = st.columns([1, 1])
    with stages1:
        st.session_state['transformation_stage'] = st.radio("What stage of images would you like to run styling on?", options=stages, horizontal=True, key=str(uuid.uuid4()),
                                                             index=st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'], help="Extracted frames means the original frames from the video.")
    with stages2:
        if st.session_state['transformation_stage'] == "Source Image":
            source_img = timing_details[st.session_state['current_frame_index'] - 1].source_image
            image = source_img.location if source_img else ""
        elif st.session_state['transformation_stage'] == "Main Variant":
            image = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
        if image != "":
            st.image(image, use_column_width=True,
                     caption=f"Image {st.session_state['current_frame_index']}")
        else:
            st.error(
                f"No {st.session_state['transformation_stage']} image found for this variant")

    if stages.index(st.session_state['transformation_stage']) != st.session_state['index_of_which_stage_to_run_on']:
        st.session_state['index_of_which_stage_to_run_on'] = stages.index(
            st.session_state['transformation_stage'])
        st.experimental_rerun()

    custom_pipelines = ["None", "Mystique"]
    if 'index_of_last_custom_pipeline' not in st.session_state:
        st.session_state['index_of_last_custom_pipeline'] = 0
    st.session_state['custom_pipeline'] = st.selectbox(
        f"Custom Pipeline:", custom_pipelines, index=st.session_state['index_of_last_custom_pipeline'])
    if custom_pipelines.index(st.session_state['custom_pipeline']) != st.session_state['index_of_last_custom_pipeline']:
        st.session_state['index_of_last_custom_pipeline'] = custom_pipelines.index(
            st.session_state['custom_pipeline'])
        st.experimental_rerun()

    if st.session_state['custom_pipeline'] == "Mystique":
        if st.session_state['index_of_default_model'] > 1:
            st.session_state['index_of_default_model'] = 0
            st.experimental_rerun()
        with st.expander("Mystique is a custom pipeline that uses a multiple models to generate a consistent character and style transformation."):
            st.markdown("## How to use the Mystique pipeline")
            st.markdown(
                "1. Create a fine-tined model in the Custom Model section of the app - we recommend Dreambooth for character transformations.")
            st.markdown(
                "2. It's best to include a detailed prompt. We recommend taking an example input image and running it through the Prompt Finder")
            st.markdown("3. Use [expression], [location], [mouth], and [looking] tags to vary the expression and location of the character dynamically if that changes throughout the clip. Varying this in the prompt will make the character look more natural - especially useful if the character is speaking.")
            st.markdown("4. In our experience, the best strength for coherent character transformations is 0.25-0.3 - any more than this and details like eye position change.")
        models = ["LoRA", "Dreambooth"]
        st.session_state['model'] = st.selectbox(
            f"Which type of model is trained on your character?", models, index=st.session_state['index_of_default_model'])
        if st.session_state['index_of_default_model'] != models.index(st.session_state['model']):
            st.session_state['index_of_default_model'] = models.index(
                st.session_state['model'])
            st.experimental_rerun()
    else:

        model_list = data_repo.get_all_ai_model_list(custom_trained=False)
        model_name_list = [m.name for m in model_list]

        # user_model_list = data_repo.get_all_ai_model_list(custom_trained=True)
        # user_model_name_list = [m.name for m in user_model_list]

        if not ('index_of_default_model' in st.session_state and st.session_state['index_of_default_model']):
            if project_settings.default_model:
                st.session_state['model'] = project_settings.default_model.uuid
                st.session_state['index_of_default_model'] = next((i for i, obj in enumerate(
                    model_list) if getattr(obj, 'uuid') == project_settings.default_model.uuid), 0)
                st.write(
                    f"Index of last model: {st.session_state['index_of_default_model']}")
            else:
                st.session_state['index_of_default_model'] = 0

        selected_model_name = st.selectbox(
            f"Which model would you like to use?", model_name_list, index=st.session_state['index_of_default_model'])
        st.session_state['model'] = next((obj.uuid for i, obj in enumerate(
            model_list) if getattr(obj, 'name') == selected_model_name), None)

        selected_model_index = next((i for i, obj in enumerate(
            model_list) if getattr(obj, 'name') == selected_model_name), None)
        if st.session_state['index_of_default_model'] != selected_model_index:
            st.session_state['index_of_default_model'] = selected_model_index
            # st.experimental_rerun()

    current_model_name = data_repo.get_ai_model_from_uuid(
        st.session_state['model']).name

    # NOTE: there is a check when creating custom models that no two model can have the same name
    if current_model_name == AIModelType.CONTROLNET.value:
        controlnet_adapter_types = [
            "scribble", "normal", "canny", "hed", "seg", "hough", "depth2img", "pose"]
        if 'index_of_controlnet_adapter_type' not in st.session_state:
            st.session_state['index_of_controlnet_adapter_type'] = 0
        st.session_state['adapter_type'] = st.selectbox(
            f"Adapter Type", controlnet_adapter_types, index=st.session_state['index_of_controlnet_adapter_type'])

        if st.session_state['index_of_controlnet_adapter_type'] != controlnet_adapter_types.index(st.session_state['adapter_type']):
            st.session_state['index_of_controlnet_adapter_type'] = controlnet_adapter_types.index(
                st.session_state['adapter_type'])
            st.experimental_rerun()
        st.session_state['custom_models'] = []

    elif current_model_name == AIModelType.LORA.value:
        if not ('index_of_lora_model_1' in st.session_state and st.session_state['index_of_lora_model_1']):
            st.session_state['index_of_lora_model_1'] = 0
            st.session_state['index_of_lora_model_2'] = 0
            st.session_state['index_of_lora_model_3'] = 0

        # df = pd.read_csv('models.csv')
        # filtered_df = df[df.iloc[:, 5] == 'LoRA']
        # lora_model_list = filtered_df.iloc[:, 0].tolist()
        lora_model_list = data_repo.get_all_ai_model_list(
            model_type_list=[AIModelType.LORA.value], custom_trained=True)
        null_model = InternalAIModelObject(
            None, "", None, None, None, None, None, None, None, None, None, None)
        lora_model_list.insert(0, null_model)
        lora_model_name_list = [m.name for m in lora_model_list]

        # TODO: remove this array from db table
        custom_models = []
        selected_lora_1_name = st.selectbox(
            f"LoRA Model 1", lora_model_name_list, index=st.session_state['index_of_lora_model_1'], key="lora_1")
        st.session_state['lora_model_1'] = next((obj.uuid for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_1_name), "")
        st.session_state['lora_model_1_url'] = next((obj.replicate_url for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_1_name), "")
        selected_lora_1_index = next((i for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_1_name), 0)

        if st.session_state['index_of_lora_model_1'] != selected_lora_1_index:
            st.session_state['index_of_lora_model_1'] = selected_lora_1_index

        if st.session_state['index_of_lora_model_1'] != 0:
            custom_models.append(st.session_state['lora_model_1'])

        selected_lora_2_name = st.selectbox(
            f"LoRA Model 2", lora_model_name_list, index=st.session_state['index_of_lora_model_2'], key="lora_2")
        st.session_state['lora_model_2'] = next((obj.uuid for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_2_name), "")
        st.session_state['lora_model_2_url'] = next((obj.replicate_url for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_2_name), "")
        selected_lora_2_index = next((i for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_2_name), 0)

        if st.session_state['index_of_lora_model_2'] != selected_lora_2_index:
            st.session_state['index_of_lora_model_2'] = selected_lora_2_index

        if st.session_state['index_of_lora_model_2'] != 0:
            custom_models.append(st.session_state['lora_model_2'])

        selected_lora_3_name = st.selectbox(
            f"LoRA Model 3", lora_model_name_list, index=st.session_state['index_of_lora_model_3'], key="lora_3")
        st.session_state['lora_model_3'] = next((obj.uuid for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_3_name), "")
        st.session_state['lora_model_3_url'] = next((obj.replicate_url for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_3_name), "")
        selected_lora_3_index = next((i for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_3_name), 0)

        if st.session_state['index_of_lora_model_3'] != selected_lora_3_index:
            st.session_state['index_of_lora_model_3'] = selected_lora_3_index

        if st.session_state['index_of_lora_model_3'] != 0:
            custom_models.append(st.session_state['lora_model_3'])

        st.session_state['custom_models'] = json.dumps(custom_models)

        st.info("You can reference each model in your prompt using the following keywords: <1>, <2>, <3> - for example '<1> in the style of <2>.")

        lora_adapter_types = ['sketch', 'seg', 'keypose', 'depth', None]
        if f"index_of_lora_adapter_type_{append_to_item_name}" not in st.session_state:
            st.session_state['index_of_lora_adapter_type'] = 0

        st.session_state['adapter_type'] = st.selectbox(
            f"Adapter Type:", lora_adapter_types, help="This is the method through the model will infer the shape of the object. ", index=st.session_state['index_of_lora_adapter_type'])

        if st.session_state['index_of_lora_adapter_type'] != lora_adapter_types.index(st.session_state['adapter_type']):
            st.session_state['index_of_lora_adapter_type'] = lora_adapter_types.index(
                st.session_state['adapter_type'])

    elif current_model_name == AIModelType.DREAMBOOTH.value:
        # df = pd.read_csv('models.csv')
        # filtered_df = df[df.iloc[:, 5] == 'Dreambooth']
        # dreambooth_model_list = filtered_df.iloc[:, 0].tolist()

        dreambooth_model_list = data_repo.get_all_ai_model_list(
            model_type_list=[AIModelType.DREAMBOOTH.value], custom_trained=True)
        dreambooth_model_name_list = [m.name for m in dreambooth_model_list]

        if not ('index_of_dreambooth_model' in st.session_state and st.session_state['index_of_dreambooth_model']):
            st.session_state['index_of_dreambooth_model'] = 0

        selected_dreambooth_model_name = st.selectbox(
            f"Dreambooth Model", dreambooth_model_name_list, index=st.session_state['index_of_dreambooth_model'])
        # st.session_state['custom_models'] = next((obj.uuid for i, obj in enumerate(
        #     dreambooth_model_list) if getattr(obj, 'name') == selected_dreambooth_model_name), "")
        selected_dreambooth_model_index = next((i for i, obj in enumerate(
            dreambooth_model_list) if getattr(obj, 'name') == selected_dreambooth_model_name), "")
        if st.session_state['index_of_dreambooth_model'] != selected_dreambooth_model_index:
            st.session_state['index_of_dreambooth_model'] = selected_dreambooth_model_index

        st.session_state['dreambooth_model_uuid'] = dreambooth_model_list[st.session_state['index_of_dreambooth_model']].uuid
    else:
        st.session_state['custom_models'] = []
        st.session_state['adapter_type'] = "N"

    if not ( 'adapter_type' in st.session_state and st.session_state['adapter_type']):
        st.session_state['adapter_type'] = 'N'

    if st.session_state['adapter_type'] == "canny":

        canny1, canny2 = st.columns(2)

        if view_type == "List":

            if project_settings.default_low_threshold != "":
                low_threshold_value = project_settings.default_low_threshold
            else:
                low_threshold_value = 50

            if project_settings.default_high_threshold != "":
                high_threshold_value = project_settings.default_high_threshold
            else:
                high_threshold_value = 150

        elif view_type == "Single":

            if timing.low_threshold != "":
                low_threshold_value = timing.low_threshold
            else:
                low_threshold_value = 50

            if timing.high_threshold != "":
                high_threshold_value = timing.high_threshold
            else:
                high_threshold_value = 150

        with canny1:
            st.session_state['low_threshold'] = st.slider(
                'Low Threshold', 0, 255, value=int(low_threshold_value))
        with canny2:
            st.session_state['high_threshold'] = st.slider(
                'High Threshold', 0, 255, value=int(high_threshold_value))
    else:
        st.session_state['low_threshold'] = 0
        st.session_state['high_threshold'] = 0

    if st.session_state['model'] == "StyleGAN-NADA":
        st.warning("StyleGAN-NADA is a custom model that uses StyleGAN to generate a consistent character and style transformation. It only works for square images.")
        st.session_state['prompt'] = st.selectbox("What style would you like to apply to the character?", ['base', 'mona_lisa', 'modigliani', 'cubism', 'elf', 'sketch_hq', 'thomas', 'thanos', 'simpson', 'witcher',
                                                  'edvard_munch', 'ukiyoe', 'botero', 'shrek', 'joker', 'pixar', 'zombie', 'werewolf', 'groot', 'ssj', 'rick_morty_cartoon', 'anime', 'white_walker', 'zuckerberg', 'disney_princess', 'all', 'list'])
        st.session_state['strength'] = 0.5
        st.session_state['guidance_scale'] = 7.5
        st.session_state['seed'] = int(0)
        st.session_state['num_inference_steps'] = int(50)

    else:
        if view_type == "List":
            if project_settings.default_prompt != "":
                st.session_state[f'prompt_value_{append_to_item_name}'] = project_settings.default_prompt
            else:
                st.session_state[f'prompt_value_{append_to_item_name}'] = ""

        elif view_type == "Single":
            if timing.prompt != "":
                st.session_state[f'prompt_value_{append_to_item_name}'] = timing.prompt
            else:
                st.session_state[f'prompt_value_{append_to_item_name}'] = ""

        st.session_state['prompt'] = st.text_area(
            f"Prompt", label_visibility="visible", value=st.session_state[f'prompt_value_{append_to_item_name}'], height=150)
        if st.session_state['prompt'] != st.session_state['prompt_value']:
            st.session_state['prompt_value'] = st.session_state['prompt']
            st.experimental_rerun()
        if view_type == "List":
            st.info(
                "You can include the following tags in the prompt to vary the prompt dynamically: [expression], [location], [mouth], and [looking]")
        if st.session_state['model'] == AIModelType.DREAMBOOTH.value:
            model_details: InternalAIModelObject = data_repo.get_ai_model_from_uuid(
                st.session_state['dreambooth_model_uuid'])
            st.info(
                f"Must include '{model_details.keyword}' to run this model")
            # TODO: CORRECT-CODE add controller_type to ai_model
            if model_details.controller_type != "":
                st.session_state['adapter_type'] = st.selectbox(
                    f"Would you like to use the {model_details.controller_type} controller?", ['Yes', 'No'])
            else:
                st.session_state['adapter_type'] = "No"

        else:
            if st.session_state['model'] == AIModelType.PIX_2_PIX.value:
                st.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")

        if view_type == "List":
            if project_settings.default_strength != "":
                st.session_state['strength'] = project_settings.default_strength
            else:
                st.session_state['strength'] = 0.5

        elif view_type == "Single":
            if timing.strength:
                st.session_state['strength'] = timing.strength
            else:
                st.session_state['strength'] = 0.5

        st.session_state['strength'] = st.slider(f"Strength", value=float(
            st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)

        if view_type == "List":
            if project_settings.default_guidance_scale != "":
                st.session_state['guidance_scale'] = project_settings.default_guidance_scale
            else:
                st.session_state['guidance_scale'] = 7.5
        elif view_type == "Single":
            if timing.guidance_scale != "":
                st.session_state['guidance_scale'] = timing.guidance_scale
            else:
                st.session_state['guidance_scale'] = 7.5

        st.session_state['negative_prompt'] = st.text_area(
            f"Negative prompt", value=st.session_state['negative_prompt_value'], label_visibility="visible")
        
        if st.session_state['negative_prompt'] != st.session_state['negative_prompt_value']:
            st.session_state['negative_prompt_value'] = st.session_state['negative_prompt']
            st.experimental_rerun()
        
        st.session_state['guidance_scale'] = st.number_input(
            f"Guidance scale", value=float(st.session_state['guidance_scale']))
        
        if view_type == "List":
            if project_settings.default_seed != "":
                st.session_state['seed'] = project_settings.default_seed
            else:
                st.session_state['seed'] = 0

        elif view_type == "Single":
            if timing.seed != "":
                st.session_state['seed'] = timing.seed
            else:
                st.session_state['seed'] = 0

        st.session_state['seed'] = st.number_input(
            f"Seed", value=int(st.session_state['seed']))
        
        if view_type == "List":
            if project_settings.default_num_inference_steps:
                st.session_state['num_inference_steps'] = project_settings.default_num_inference_steps
            else:
                st.session_state['num_inference_steps'] = 50
        elif view_type == "Single":
            if timing.num_inteference_steps:
                st.session_state['num_inference_steps'] = timing.num_inteference_steps
            else:
                st.session_state['num_inference_steps'] = 50
        st.session_state['num_inference_steps'] = st.number_input(
            f"Inference steps", value=int(st.session_state['num_inference_steps']))

    st.session_state["promote_new_generation"] = st.checkbox(
        "Promote new generation to main variant", key="promote_new_generation_to_main_variant_1")
    st.session_state["use_new_settings"] = True

    if view_type == "List":
        batch_run_range = st.slider(
            "Select range:", 1, 0, (0, len(timing_details)-1))
        first_batch_run_value = batch_run_range[0]
        last_batch_run_value = batch_run_range[1]

        st.write(batch_run_range)

        st.session_state["promote_new_generation"] = st.checkbox(
            "Promote new generation to main variant", key="promote_new_generation_to_main_variant")
        st.session_state["use_new_settings"] = st.checkbox(
            "Use new settings for batch query", key="keep_existing_settings", help="If unchecked, the new settings will be applied to the existing variants.")

        if 'restyle_button' not in st.session_state:
            st.session_state['restyle_button'] = ''
            st.session_state['item_to_restyle'] = ''

        btn1, btn2 = st.columns(2)

        with btn1:

            batch_number_of_variants = st.number_input(
                "How many variants?", value=1, min_value=1, max_value=10, step=1, key="number_of_variants")

        with btn2:

            st.write("")
            st.write("")
            if st.button(f'Batch restyle') or st.session_state['restyle_button'] == 'yes':

                if st.session_state['restyle_button'] == 'yes':
                    range_start = int(st.session_state['item_to_restyle'])
                    range_end = range_start + 1
                    st.session_state['restyle_button'] = ''
                    st.session_state['item_to_restyle'] = ''

                for i in range(first_batch_run_value, last_batch_run_value+1):
                    for _ in range(0, batch_number_of_variants):
                        trigger_restyling_process(timing_details[i].uuid, st.session_state['model'], st.session_state['prompt'], st.session_state['strength'], st.session_state['custom_pipeline'], st.session_state['negative_prompt'], st.session_state['guidance_scale'], st.session_state['seed'], st.session_state[
                                                  'num_inference_steps'], st.session_state['transformation_stage'], st.session_state["promote_new_generation"], st.session_state['custom_models'], st.session_state['adapter_type'], st.session_state["use_new_settings"], st.session_state['low_threshold'], st.session_state['high_threshold'])
                st.experimental_rerun()

        # if st.button(f'Jump to list view'):
        #    st.session_state['frame_styling_view_type'] = "List View"
        #    st.experimental_rerun()


def get_primary_variant_location(timing_details, which_image):

    if timing_details[which_image]["alternative_images"] == "":
        return ""
    else:
        variants = timing_details[which_image]["alternative_images"]
        current_variant = int(timing_details[which_image]["primary_image"])
        primary_variant_location = variants[current_variant]
        return primary_variant_location


def convert_to_minutes_and_seconds(frame_time):
    minutes = int(frame_time/60)
    seconds = frame_time - (minutes*60)
    seconds = round(seconds, 2)
    return f"{minutes} min, {seconds} secs"


def calculate_time_at_frame_number(input_video: InternalFileObject, frame_number):
    video = cv2.VideoCapture(input_video.local_path)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_percentage = float(frame_number / frame_count)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    time_at_frame = float(frame_percentage * length_of_video)
    return time_at_frame


def preview_frame(project_uuid, video, frame_num):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(
        project_uuid)
    cap = cv2.VideoCapture(
        f'videos/{project.uuid}/assets/resources/input_videos/{video.name}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame

# extract_frame_number is extracted from the input_video and added as source_image at frame_number
# (timing_uuid) in the timings table


def extract_frame(timing_uuid, input_video: InternalFileObject, extract_frame_number):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.project.uuid

    # TODO: standardize the input video path
    # input_video = "videos/" + \
    #     str(timing.project.name) + \
    #     "/assets/resources/input_videos/" + str(input_video)
    cv_video = cv2.VideoCapture(input_video.local_path)
    total_frames = cv_video.get(cv2.CAP_PROP_FRAME_COUNT)
    if extract_frame_number == total_frames:
        extract_frame_number = int(total_frames - 1)
    cv_video.set(cv2.CAP_PROP_POS_FRAMES, extract_frame_number)
    ret, frame = cv_video.read()

    frame_time = calculate_time_at_frame_number(
        input_video, float(extract_frame_number))

    data_repo.update_specific_timing(
        timing_uuid, frame_number=extract_frame_number)
    data_repo.update_specific_timing(timing_uuid, frame_time=frame_time)

    file_name = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=16)) + ".png"
    file_location = "videos/" + timing.project.uuid + \
        "/assets/frames/1_selected/" + str(file_name)
    
    # cv2.imwrite(file_location, frame)
    # img = Image.open("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    # img.save("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")

    pil_img = Image.fromarray(frame)
    hosted_url = save_or_host_file(pil_img, file_location)

    file_data = {
        "name": file_name,
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_location})

    final_image = data_repo.create_file(**file_data)
    data_repo.update_specific_timing(
        timing_uuid, source_image_id=final_image.uuid)

    return final_image


def calculate_frame_number_at_time(input_video, time_of_frame, project_uuid):
    data_repo = DataRepo()
    project = data_repo.get_project_from_uuid(project_uuid)

    time_of_frame = float(time_of_frame)
    input_video = "videos/" + \
        str(project.uuid) + "/assets/resources/input_videos/" + str(input_video)
    video = cv2.VideoCapture(input_video)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    percentage_of_video = float(time_of_frame / length_of_video)
    frame_number = int(percentage_of_video * frame_count)
    if frame_number == 0:
        frame_number = 1
    return frame_number


def move_frame(direction, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if direction == "Up":
        if timing.aux_frame_index == 0:
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index - 1)
        # prev_timing = data_repo.get_prev_timing(timing.uuid)
        # previous_primary_image = prev_timing.primary_image
        # previous_alternative_images = prev_timing.alternative_images
        # previous_source_image = prev_timing.source_image

        # data_repo.update_specific_timing(
        #     prev_timing.uuid, primary_image_id=current_primary_image.uuid)
        # print("current_alternative_images= ", current_alternative_images)
        # data_repo.update_specific_timing(
        #     prev_timing.uuid, alternative_images=str(current_alternative_images))
        # data_repo.update_specific_timing(
        #     prev_timing.uuid, source_image_id=current_source_image.uuid)
        # data_repo.update_specific_timing(
        #     prev_timing.uuid, interpolated_video=None)
        # data_repo.update_specific_timing(prev_timing.uuid, timed_clip_id=None)

        # data_repo.update_specific_timing(
        #     timing.uuid, primary_image_id=previous_primary_image.uuid)
        # data_repo.update_specific_timing(
        #     timing.uuid, alternative_images=str(previous_alternative_images))
        # data_repo.update_specific_timing(
        #     timing.uuid, source_image_id=previous_source_image.uuid)

    elif direction == "Down":
        timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)
        if timing.aux_frame_index == len(timing_list) - 1:
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index + 1)
        # next_primary_image = data_repo.get_next_timing(
        #     timing.uuid).primary_image
        # next_alternative_images = data_repo.get_next_timing(
        #     timing.uuid).alternative_images
        # next_source_image = data_repo.get_next_timing(timing.uuid).source_image

        # data_repo.update_specific_timing(
        #     data_repo.get_next_timing(timing.uuid).uuid, primary_image_id=current_primary_image.uuid)
        # data_repo.update_specific_timing(
        #     data_repo.get_next_timing(timing.uuid).uuid, alternative_images=str(current_alternative_images))
        # data_repo.update_specific_timing(
        #     data_repo.get_next_timing(timing.uuid).uuid, source_image_id=current_source_image.uuid)
        # data_repo.update_specific_timing(
        #     data_repo.get_next_timing(timing.uuid).uuid, interpolated_video=None)
        # data_repo.update_specific_timing(
        #     data_repo.get_next_timing(timing.uuid).uuid, timed_clip_id=None)

        # data_repo.update_specific_timing(
        #     timing.uuid, primary_image_id=next_primary_image.uuid)
        # data_repo.update_specific_timing(
        #     timing.uuid, alternative_images=str(next_alternative_images))
        # data_repo.update_specific_timing(
        #     timing.uuid, source_image_id=next_source_image.uuid)


# def get_timing_details(video_name):
#     file_path = "videos/" + str(video_name) + "/timings.csv"
#     csv_processor = CSVProcessor(file_path)
#     column_types = {
#         'frame_time': float,
#         'frame_number': int,
#         'primary_image': int,
#         'guidance_scale': float,
#         'seed': int,
#         'num_inference_steps': int,
#         'strength': float
#     }
#     df = csv_processor.get_df_data().astype(column_types, errors='ignore')

#     df['primary_image'] = pd.to_numeric(df['primary_image'], errors='coerce').round(
#     ).astype(pd.Int64Dtype(), errors='ignore')
#     df['seed'] = pd.to_numeric(df['seed'], errors='coerce').round().astype(
#         pd.Int64Dtype(), errors='ignore')
#     df['num_inference_steps'] = pd.to_numeric(
#         df['num_inference_steps'], errors='coerce').round().astype(pd.Int64Dtype(), errors='ignore')
#     # if source_image if empty, set to https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png

#     # Evaluate the alternative_images column and replace it with the evaluated list
#     df['alternative_images'] = df['alternative_images'].fillna(
#         '').apply(lambda x: ast.literal_eval(x[1:-1]) if x != '' else '')
#     return df.to_dict('records')

# delete keyframe at a particular index from timings.csv


def delete_frame(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    next_timing = data_repo.get_next_timing(timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)

    if next_timing:
        data_repo.update_specific_timing(
                next_timing.uuid, interpolated_video_id=None)

        data_repo.update_specific_timing(
                next_timing.uuid, timed_clip_id=None)

    data_repo.delete_timing_from_uuid(timing.uuid)
    
    if timing.aux_frame_index == len(timing_details) - 1:
        st.session_state['current_frame_index'] = max(1, st.session_state['current_frame_index'] - 1)
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid

    


# def batch_update_timing_values(timing_uuid, prompt, strength, model, custom_pipeline, negative_prompt, guidance_scale, seed, num_inference_steps, source_image, custom_models, adapter_type, low_threshold, high_threshold):

#     csv_processor = CSVProcessor(
#         "videos/" + str(project_name) + "/timings.csv")
#     df = csv_processor.get_df_data()

#     if model != "Dreambooth":
#         custom_models = f'"{custom_models}"'
#     df.iloc[index_of_current_item, [18, 10, 9, 4, 5, 6, 7, 8, 12, 13, 14, 24, 25]] = [prompt, float(strength), model, custom_pipeline, negative_prompt, float(
#         guidance_scale), int(seed), int(num_inference_steps), source_image, custom_models, adapter_type, int(float(low_threshold)), int(float(high_threshold))]

#     df["primary_image"] = pd.to_numeric(
#         df["primary_image"], downcast='integer', errors='coerce')
#     df["primary_image"].fillna(0, inplace=True)
#     df["primary_image"] = df["primary_image"].astype(int)

#     df["seed"] = pd.to_numeric(df["seed"], downcast='integer', errors='coerce')
#     df["seed"].fillna(0, inplace=True)
#     df["seed"] = df["seed"].astype(int)

#     df["num_inference_steps"] = pd.to_numeric(
#         df["num_inference_steps"], downcast='integer', errors='coerce')
#     df["num_inference_steps"].fillna(0, inplace=True)
#     df["num_inference_steps"] = df["num_inference_steps"].astype(int)

#     df["low_threshold"] = pd.to_numeric(
#         df["low_threshold"], downcast='integer', errors='coerce')
#     df["low_threshold"].fillna(0, inplace=True)
#     df["low_threshold"] = df["low_threshold"].astype(int)

#     df["high_threshold"] = pd.to_numeric(
#         df["high_threshold"], downcast='integer', errors='coerce')
#     df["high_threshold"].fillna(0, inplace=True)
#     df["high_threshold"] = df["high_threshold"].astype(int)

#     df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)

# TODO: redundant function, absorb this in another function
def batch_update_timing_values(
    timing_uuid,
    model_uuid,
    source_image_uuid,
    prompt,
    strength,
    custom_pipeline,
    negative_prompt,
    guidance_scale,
    seed,
    num_inference_steps,
    custom_models,
    adapter_type,
    low_threshold,
    high_threshold
):
    data_repo = DataRepo()

    data_repo.update_specific_timing(
        uuid=timing_uuid,
        model_id=model_uuid,
        source_image_id=source_image_uuid,
        prompt=prompt,
        strength=strength,
        custom_pipeline=custom_pipeline,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        seed=seed,
        num_inference_steps=num_inference_steps,
        custom_models=custom_models,
        adapter_type=adapter_type,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )


def dynamic_prompting(prompt, source_image, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

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

    data_repo.update_specific_timing(timing_uuid, prompt=prompt)


def trigger_restyling_process(
    timing_uuid,
    model_uuid,
    prompt,
    strength,
    custom_pipeline,
    negative_prompt,
    guidance_scale,
    seed,
    num_inference_steps,
    transformation_stage,
    promote_new_generation,
    custom_models,
    adapter_type,
    update_inference_settings,
    low_threshold,
    high_threshold
):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    # timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
    #     timing.project.uuid)
    # project_setting: InternalSettingObject = data_repo.get_project_setting(
    #     timing.project.uuid)

    # TODO: add proper form validations throughout the code
    if not prompt:
        st.error("Please enter a prompt")
        return

    if update_inference_settings is True:
        prompt = prompt.replace(",", ".")
        prompt = prompt.replace("\n", "")
        data_repo.update_project_setting(
            timing.project.uuid,
            default_prompt=prompt,
            default_strength=strength,
            default_model_id=model_uuid,
            default_custom_pipeline=custom_pipeline,
            default_negative_prompt=negative_prompt,
            default_guidance_scale=guidance_scale,
            default_seed=seed,
            default_num_inference_steps=num_inference_steps,
            default_which_stage_to_run_on=transformation_stage,
            default_custom_models=custom_models,
            default_adapter_type=adapter_type
        )

        if low_threshold != "":
            data_repo.update_project_setting(
                timing.project.uuid, default_low_threshold=low_threshold)
        if high_threshold != "":
            data_repo.update_project_setting(
                timing.project.uuid, default_high_threshold=high_threshold)

        if timing.source_image == "":
            source_image = ""
        else:
            source_image = timing.source_image

        batch_update_timing_values(
            timing_uuid,
            model_uuid,
            timing.source_image.uuid,
            prompt,
            strength,
            custom_pipeline,
            negative_prompt,
            guidance_scale,
            seed,
            num_inference_steps,
            custom_models,
            adapter_type,
            low_threshold,
            high_threshold
        )
        dynamic_prompting(prompt, source_image, timing_uuid)

    timing = data_repo.get_timing_from_uuid(timing_uuid)
    if transformation_stage == "Source Image":
        source_image = timing.source_image
    else:
        variants: List[InternalFileObject] = timing.alternative_images_list
        number_of_variants = len(variants)
        primary_image = timing.primary_image
        source_image = primary_image.location

    if st.session_state['custom_pipeline'] == "Mystique":
        output_file = custom_pipeline_mystique(timing_uuid, source_image)
    else:
        output_file = restyle_images(timing_uuid, source_image)

    if output_file != None:
        add_image_variant(output_file.uuid, timing_uuid)

        if promote_new_generation == True:
            timing = data_repo.get_timing_from_uuid(timing_uuid)
            variants = timing.alternative_images_list
            number_of_variants = len(variants)
            if number_of_variants == 1:
                print("No new generation to promote")
            else:
                promote_image_variant(timing_uuid, number_of_variants - 1)
    else:
        print("No new generation to promote")


def promote_image_variant(timing_uuid, variant_to_promote_frame_number: str):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    variant_to_promote = timing.alternative_images_list[variant_to_promote_frame_number]
    data_repo.update_specific_timing(
        timing_uuid, primary_image_id=variant_to_promote.uuid)

    prev_timing = data_repo.get_prev_timing(timing_uuid)
    if prev_timing:
        data_repo.update_specific_timing(
            prev_timing.uuid, interpolated_clip_id=None)
        data_repo.update_specific_timing(
            timing_uuid, interpolated_clip_id=None)

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    frame_idx = timing.aux_frame_index

    # DOUBT: setting last interpolated_video to empty?
    if frame_idx < len(timing_details):
        data_repo.update_specific_timing(
            timing.uuid, interpolated_clip_id=None)

    if frame_idx > 1:
        data_repo.update_specific_timing(
            data_repo.get_prev_timing(timing_uuid).uuid, timed_clip_id=None)

    data_repo.update_specific_timing(timing_uuid, timed_clip_id=None)

    if frame_idx < len(timing_details):
        data_repo.update_specific_timing(timing.uuid, timed_clip_id=None)


def extract_canny_lines(image_path_or_url, project_uuid, low_threshold=50, high_threshold=150) -> InternalFileObject:
    data_repo = DataRepo()

    # Check if the input is a URL
    if image_path_or_url.startswith("http"):
        response = r.get(image_path_or_url)
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    else:
        # Read the image from a local file
        image = cv2.imread(image_path_or_url, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply the Canny edge detection
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Reverse the colors (invert the image)
    inverted_canny_edges = 255 - canny_edges

    # Convert the inverted Canny edge result to a PIL Image
    new_canny_image = Image.fromarray(inverted_canny_edges)

    # Save the new image
    unique_file_name = str(uuid.uuid4()) + ".png"
    file_path = f"videos/{project_uuid}/assets/resources/masks/{unique_file_name}"
    hosted_url = save_or_host_file(new_canny_image, file_path)

    file_data = {
        "name": unique_file_name,
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_path})

    canny_image_file = data_repo.create_file(**file_data)
    return canny_image_file

# the input image is an image created by the PIL library
def create_or_update_mask(timing_uuid, image) -> InternalFileObject:
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    unique_file_name = str(uuid.uuid4()) + ".png"
    file_location = f"videos/{timing.project.uuid}/assets/resources/masks/{unique_file_name}"

    hosted_url = save_or_host_file(image, file_location)
    # if mask is not present than creating a new one
    if not (timing.mask and timing.mask.location):
        file_data = {
            "name": unique_file_name,
            "type": InternalFileType.IMAGE.value
        }

        if hosted_url:
            file_data.update({'hosted_url': hosted_url})
        else:
            file_data.update({'local_path': file_location})

        mask_file: InternalFileObject = data_repo.create_file(**file_data)
        data_repo.update_specific_timing(timing_uuid, mask_id=mask_file.uuid)
    else:
        # if it is already present then just updating the file location
        if hosted_url:
            data_repo.update_file(timing.mask.uuid, hosted_url=hosted_url)
        else:
            data_repo.update_file(timing.mask.uuid, local_path=file_location)

    timing = data_repo.get_timing_from_uuid(timing_uuid)
    return timing.mask

# NOTE: method not in use
# def create_working_assets(video_name):
#     os.mkdir("videos/" + video_name)
#     os.mkdir("videos/" + video_name + "/assets")

#     os.mkdir("videos/" + video_name + "/assets/frames")

#     os.mkdir("videos/" + video_name + "/assets/frames/0_extracted")
#     os.mkdir("videos/" + video_name + "/assets/frames/1_selected")
#     os.mkdir("videos/" + video_name +
#              "/assets/frames/2_character_pipeline_completed")
#     os.mkdir("videos/" + video_name +
#              "/assets/frames/3_backdrop_pipeline_completed")

#     os.mkdir("videos/" + video_name + "/assets/resources")

#     os.mkdir("videos/" + video_name + "/assets/resources/backgrounds")
#     os.mkdir("videos/" + video_name + "/assets/resources/masks")
#     os.mkdir("videos/" + video_name + "/assets/resources/audio")
#     os.mkdir("videos/" + video_name + "/assets/resources/input_videos")
#     os.mkdir("videos/" + video_name + "/assets/resources/prompt_images")

#     os.mkdir("videos/" + video_name + "/assets/videos")

#     os.mkdir("videos/" + video_name + "/assets/videos/0_raw")
#     os.mkdir("videos/" + video_name + "/assets/videos/1_final")
#     os.mkdir("videos/" + video_name + "/assets/videos/2_completed")

#     data = {'key': ['last_prompt', 'last_model', 'last_strength', 'last_custom_pipeline', 'audio', 'input_type', 'input_video', 'extraction_type', 'width', 'height', 'last_negative_prompt', 'last_guidance_scale', 'last_seed', 'last_num_inference_steps', 'last_which_stage_to_run_on', 'last_custom_models', 'last_adapter_type', 'guidance_type', 'default_animation_style', 'last_low_threshold', 'last_high_threshold', 'last_stage_run_on', 'zoom_level', 'rotation_angle_value', 'x_shift', 'y_shift'],
#             'value': ['prompt', 'controlnet', '0.5', 'None', '', 'video', '', 'Extract manually', '', '', '', 7.5, 0, 50, 'Source Image', '', '', '', '', 100, 200, '', 100, 0, 0, 0]}

#     df = pd.DataFrame(data)

#     df.to_csv(f'videos/{video_name}/settings.csv', index=False)

#     df = pd.DataFrame(columns=['frame_time', 'frame_number', 'primary_image', 'alternative_images', 'custom_pipeline', 'negative_prompt', 'guidance_scale', 'seed', 'num_inference_steps',
#                       'model_id', 'strength', 'notes', 'source_image', 'custom_models', 'adapter_type', 'duration_of_clip', 'interpolated_video', 'timing_video', 'prompt', 'mask', 'canny_image', 'preview_video', 'animation_style', 'interpolation_steps', 'low_threshold', 'high_threshold', 'zoom_details', 'transformation_stage'])

#     df.loc[0] = [0, "", 0, "", "", "", 0, 0, 0, "", 0, "", "",
#                  "", "", 0, "", "", "", "", "", "", "", "", "", "", "", ""]

#     st.session_state['current_frame_index'] = 0

#     df.to_csv(f'videos/{video_name}/timings.csv', index=False)


def inpainting(input_image: str, prompt, negative_prompt, timing_uuid, invert_mask, pass_mask=False) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if pass_mask == False:
        mask = timing.mask.location
    else:
        # TODO: store the local temp files in the db too
        if SERVER != ServerType.DEVELOPMENT.value:
            mask = timing.project.get_temp_mask_file(TEMP_MASK_FILE).location
        else:
            mask = MASK_IMG_LOCAL_PATH

    if not mask.startswith("http"):
        mask = open(mask, "rb")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.andreas_sd_inpainting, mask=mask, image=input_image, prompt=prompt,
                                            invert_mask=invert_mask, negative_prompt=negative_prompt, num_inference_steps=25)

    file_name = str(uuid.uuid4()) + ".png"
    image_file = data_repo.create_file(
        name=file_name, type=InternalFileType.IMAGE.value, hosted_url=output[0])

    return image_file

# adds the image file in variant (alternative images) list

def drawing_mode(timing_details,project_settings,project_uuid,stage=WorkflowStageType.STYLED.value):

    data_repo = DataRepo()

    canvas1, canvas2 = st.columns([1, 1.5])
    timing = data_repo.get_timing_from_uuid(
        st.session_state['current_frame_uuid'])

    with canvas1:
        width = int(project_settings.width)
        height = int(project_settings.height)

        if timing.source_image and timing.source_image.location != "":
            if timing.source_image.location.startswith("http"):
                canvas_image = r.get(
                    timing.source_image.location)
                canvas_image = Image.open(
                    BytesIO(canvas_image.content))
            else:
                canvas_image = Image.open(
                    timing.source_image.location)
        else:
            canvas_image = Image.new(
                "RGB", (width, height), "white")
        if 'drawing_input' not in st.session_state:
            st.session_state['drawing_input'] = 'Magic shapes 🪄'
        col1, col2 = st.columns([6, 5])

        with col1:
            st.session_state['drawing_input'] = st.radio(
                "Drawing tool:",
                ("Draw lines ✏️", "Erase Lines ❌", "Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Make Lines ║", "Make squares □"), horizontal=True,
            )

            if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                drawing_mode = "transform"

            elif st.session_state['drawing_input'] == "Make shapes 🪄":
                drawing_mode = "polygon"

            elif st.session_state['drawing_input'] == "Draw lines ✏️":
                drawing_mode = "freedraw"

            elif st.session_state['drawing_input'] == "Erase Lines ❌":
                drawing_mode = "freedraw"

            elif st.session_state['drawing_input'] == "Make Lines ║":
                drawing_mode = "line"

            elif st.session_state['drawing_input'] == "Make squares □":
                drawing_mode = "rect"

        
        with col2:

            stroke_width = st.slider(
                "Stroke width: ", 1, 100, 2)
            if st.session_state['drawing_input'] == "Erase Lines ❌":
                stroke_colour = "#ffffff"
            else:
                stroke_colour = st.color_picker(
                    "Stroke color hex: ", value="#000000")
            fill = st.checkbox("Fill shapes", value=False)
            if fill == True:
                fill_color = st.color_picker(
                    "Fill color hex: ")
            else:
                fill_color = ""
        

        st.markdown("***")
        
                                            
        threshold1, threshold2 = st.columns([1, 1])
        with threshold1:
            low_threshold = st.number_input(
                "Low Threshold", min_value=0, max_value=255, value=100, step=1)
        with threshold2:
            high_threshold = st.number_input(
                "High Threshold", min_value=0, max_value=255, value=200, step=1)

        if 'canny_image' not in st.session_state:
            st.session_state['canny_image'] = None

        if st.button("Extract Canny From image"):
            if stage == "source":
                image_path = timing_details[st.session_state['current_frame_index'] - 1].source_image.location 
        
            elif stage == "styled":
                image_path = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
            
            
            canny_image = extract_canny_lines(
                    image_path, project_uuid, low_threshold, high_threshold)
            
            st.session_state['canny_image'] = canny_image.uuid

        if st.session_state['canny_image']:
            canny_image = data_repo.get_file_from_uuid(st.session_state['canny_image'])
            
            canny_action_1, canny_action_2 = st.columns([2, 1])
            with canny_action_1:
                st.image(canny_image.location)
                                                                            
                if st.button(f"Make Into Image"):
                    data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=st.session_state['canny_image'])
                    st.session_state['reset_canvas'] = True
                    st.session_state['canny_image'] = None
                    st.experimental_rerun()

    with canvas2:
        realtime_update = True

        if "reset_canvas" not in st.session_state:
            st.session_state['reset_canvas'] = False

        if st.session_state['reset_canvas'] != True:
            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=stroke_width,
                stroke_color=stroke_colour,
                background_color="rgb(255, 255, 255)",
                background_image=canvas_image,
                update_streamlit=realtime_update,
                height=height,
                width=width,
                drawing_mode=drawing_mode,
                display_toolbar=True,
                key="full_app_draw",
            )

            if 'image_created' not in st.session_state:
                st.session_state['image_created'] = 'no'

            if canvas_result.image_data is not None:
                img_data = canvas_result.image_data
                im = Image.fromarray(
                    img_data.astype("uint8"), mode="RGBA")
        else:
            st.session_state['reset_canvas'] = False
            canvas_result = st_canvas()
            time.sleep(0.1)
            st.experimental_rerun()
        if canvas_result is not None:
            st.write("You can save the image below")
            if canvas_result.json_data is not None and not canvas_result.json_data.get('objects'):
                st.button("Save New Image", key="save_canvas", disabled=True, help="Draw something first")
            else:                
                if st.button("Save New Image", key="save_canvas_active",type="primary"):
                    if canvas_result.image_data is not None:
                        # overlay the canvas image on top of the canny image and save the result
                        # if canny image is from a url, then we need to download it first
                        if timing.source_image and timing.source_image.location:
                            if timing.source_image.location.startswith("http"):
                                canny_image = r.get(
                                    timing.source_image.location)
                                canny_image = Image.open(
                                    BytesIO(canny_image.content))
                            else:
                                canny_image = Image.open(
                                    timing.source_image.location)
                        else:
                            canny_image = Image.new(
                                "RGB", (width, height), "white")

                        canny_image = canny_image.convert("RGBA")
                        # canvas_image = canvas_image.convert("RGBA")
                        canvas_image = im
                        canvas_image = canvas_image.convert("RGBA")

                        # converting the images to the same size and mode
                        if canny_image.size != canvas_image.size:
                            canny_image = canny_image.resize(
                                canvas_image.size)

                        if canny_image.mode != canvas_image.mode:
                            canny_image = canny_image.convert(
                                canvas_image.mode)

                        new_canny_image = Image.alpha_composite(
                            canny_image, canvas_image)
                        if new_canny_image.mode != "RGB":
                            new_canny_image = new_canny_image.convert(
                                "RGB")

                        unique_file_name = str(uuid.uuid4()) + ".png"
                        file_location = f"videos/{timing.project.uuid}/assets/resources/masks/{unique_file_name}"
                        hosted_url = save_or_host_file(new_canny_image, file_location)
                        file_data = {
                            "name": str(uuid.uuid4()) + ".png",
                            "type": InternalFileType.IMAGE.value,
                            "project_id": project_uuid
                        }

                        if hosted_url:
                            file_data.update({'hosted_url': hosted_url})
                        else:
                            file_data.update({'local_path': file_location})

                        canny_image = data_repo.create_file(
                            **file_data)
                        data_repo.update_specific_timing(
                            st.session_state['current_frame_uuid'], source_image_id=canny_image.uuid)
                        st.success("New Canny Image Saved")
                        st.session_state['reset_canvas'] = True
                        time.sleep(1)
                        st.experimental_rerun()





def add_image_variant(image_file_uuid: str, timing_uuid: str):
    data_repo = DataRepo()
    image_file: InternalFileObject = data_repo.get_file_from_uuid(
        image_file_uuid)
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    alternative_image_list = timing.alternative_images_list + [image_file]
    alternative_image_uuid_list = [img.uuid for img in alternative_image_list]
    primary_image_uuid = alternative_image_uuid_list[0]
    alternative_image_uuid_list = json.dumps(alternative_image_uuid_list)

    data_repo.update_specific_timing(
        timing_uuid, alternative_images=alternative_image_uuid_list)

    if not timing.primary_image:
        data_repo.update_specific_timing(
            timing_uuid, primary_image_id=primary_image_uuid)

    return len(alternative_image_list)

# image_list is a list of uploaded_obj
def convert_image_list_to_file_list(image_list):
    data_repo = DataRepo()
    file_list = []
    for image in image_list:
        img = Image.open(image)
        filename = str(uuid.uuid4())
        file_path = "videos/training_data/" + filename
        hosted_url = save_or_host_file(img, file_path)
        data = {
            "name": str(uuid.uuid4()),
            "type": InternalFileType.IMAGE.value,
        }

        if hosted_url:
            data['hosted_url'] = hosted_url
        else:
            data['local_url'] = file_path

        image_file = data_repo.create_file(**data)
        file_list.append(image_file)
    return file_list

# DOUBT: why do we need controller in dreambooth training?
# INFO: images_list passed here are converted to internal files after they are used for training


def train_dreambooth_model(instance_prompt, class_prompt, training_file_url, max_train_steps, model_name, images_list: List[str], controller_type):
    ml_client = get_ml_client()
    response = ml_client.dreambooth_training(
        training_file_url, instance_prompt, class_prompt, max_train_steps, model_name, controller_type, len(images_list))
    training_status = response["status"]
    
    model_id = response["id"]
    if training_status == "queued":
        file_list = convert_image_list_to_file_list(images_list)
        file_uuid_list = [file.uuid for file in file_list]
        file_uuid_list = json.dumps(file_uuid_list)

        model_data = {
            "name": model_name,
            "user_id": get_current_user_uuid(),
            "replicate_model_id": model_id,
            "replicate_url": response["model"],
            "diffusers_url": "",
            "category": AIModelType.DREAMBOOTH.value,
            "training_image_list": file_uuid_list,
            "keyword": instance_prompt,
            "custom_trained": True
        }

        data_repo = DataRepo()
        data_repo.create_ai_model(**model_data)

        return "Success - Training Started. Please wait 10-15 minutes for the model to be trained."
    else:
        return "Failed"

# INFO: images_list passed here are converted to internal files after they are used for training


def train_lora_model(training_file_url, type_of_task, resolution, model_name, images_list):
    data_repo = DataRepo()
    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.clones_lora_training, instance_data=training_file_url,
                                            task=type_of_task, resolution=int(resolution))

    file_list = convert_image_list_to_file_list(images_list)
    file_uuid_list = [file.uuid for file in file_list]
    file_uuid_list = json.dumps(file_uuid_list)
    model_data = {
        "name": model_name,
        "user_id": get_current_user_uuid(),
        "replicate_url": output,
        "diffusers_url": "",
        "category": AIModelType.LORA.value,
        "training_image_list": file_uuid_list,
        "custom_trained": True
    }

    data_repo.create_ai_model(**model_data)
    return f"Successfully trained - the model '{model_name}' is now available for use!"

# TODO: making an exception for this, passing just the image urls instead of
# image files


def train_model(images_list, instance_prompt, class_prompt, max_train_steps,
                model_name, type_of_model, type_of_task, resolution, controller_type):
    # prepare and upload the training data (images.zip)
    ml_client = get_ml_client()
    try:
        training_file_url = ml_client.upload_training_data(images_list)
    except Exception as e:
        raise e

    # training the model
    model_name = model_name.replace(" ", "-").lower()
    if type_of_model == "Dreambooth":
        return train_dreambooth_model(instance_prompt, class_prompt, training_file_url,
                                      max_train_steps, model_name, images_list, controller_type)
    elif type_of_model == "LoRA":
        return train_lora_model(training_file_url, type_of_task, resolution, model_name, images_list)


def get_model_details_from_csv(model_name):
    with open('models.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == model_name:
                model_details = {
                    'name': row[0],
                    'id': row[1],
                    'keyword': row[2],
                    'version': row[3],
                    'training_images': row[4],
                    'model_type': row[5],
                    'model_url': row[6],
                    'controller_type': row[7]
                }
                return model_details


def remove_background(input_image):
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.pollination_modnet, image=input_image)
    return output


def replace_background(project_uuid, background_image) -> InternalFileObject:
    data_repo = DataRepo()
    project = data_repo.get_project_from_uuid(project_uuid)

    if background_image.startswith("http"):
        response = r.get(background_image)
        background_image = Image.open(BytesIO(response.content))
    else:
        background_image = Image.open(f"{background_image}")
    
    if SERVER == ServerType.DEVELOPMENT.value:
        foreground_image = Image.open(SECOND_MASK_FILE_PATH)
    else:
        path = project.get_temp_mask_file(SECOND_MASK_FILE).location
        response = r.get(path)
        foreground_image = Image.open(BytesIO(response.content))

    background_image.paste(foreground_image, (0, 0), foreground_image)
    filename = str(uuid.uuid4()) + ".png"
    background_img_path = f"videos/{project_uuid}/replaced_bg.png"
    hosted_url = save_or_host_file(background_image, background_img_path)
    file_data = {
        "name": filename,
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': background_img_path})
    
    image_file = data_repo.create_file(**file_data)

    return image_file


def prompt_clip_interrogator(input_image, which_model, best_or_fast):
    if which_model == "Stable Diffusion 1.5":
        which_model = "ViT-L-14/openai"
    elif which_model == "Stable Diffusion 2":
        which_model = "ViT-H-14/laion2b_s32b_b79k"

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.clip_interrogator, image=input_image, clip_model_name=which_model, mode=best_or_fast)

    return output


def prompt_model_real_esrgan_upscaling(input_image):
    data_repo = DataRepo()
    app_settings = data_repo.get_app_setting_from_uuid()

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.real_esrgan_upscale, image=input_image, upscale=2
    )

    filename = str(uuid.uuid4()) + ".png"
    image_file = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                       hosted_url=output)
    return output


def touch_up_image(image: InternalFileObject):
    data_repo = DataRepo()

    input_image = image.location
    if not input_image.startswith("http"):
        input_image = open(image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.gfp_gan, img=input_image)

    filename = str(uuid.uuid4()) + ".png"
    image_file = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                       hosted_url=output)

    return image_file

# TODO: don't save or upload image where just passing the PIL object can work
def resize_image(video_name, new_width, new_height, image_file: InternalFileObject) -> InternalFileObject:
    image = image_file.location
    response = r.get(image)
    image = Image.open(BytesIO(response.content))
    resized_image = image.resize((new_width, new_height))

    time.sleep(0.1)

    unique_id = str(uuid.uuid4())
    filepath = "videos/" + str(video_name) + \
        "/temp_image-" + unique_id + ".png"
    
    hosted_url = save_or_host_file(resized_image, filepath)
    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': filepath})

    data_repo = DataRepo()
    image_file = data_repo.create_file(**file_data)

    # not uploading or removing the created image as of now
    # resized_image = upload_image(
    #     "videos/" + str(video_name) + "/temp_image.png")

    # os.remove("videos/" + str(video_name) + "/temp_image.png")

    return image_file


def face_swap(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    model_category = timing['model']['category']

    # TODO: check this logic by running the code
    if model_category == "Dreambooth":
        custom_model_uuid = timing.custom_model_id_list[0]
        # custom_model = timing_details[index_of_current_item]["custom_models"]
    if model_category == "LoRA":
        # custom_model = ast.literal_eval(
        #     timing_details[index_of_current_item]["custom_models"][1:-1])[0]
        custom_model_uuid = timing.custom_model_id_list[0]

    model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(
        custom_model_uuid)

    # source_face = ast.literal_eval(get_model_details_from_csv(
    #     custom_model)["training_images"][1:-1])[0]

    source_face = json.loads(model.training_image_list)[0]
    target_face = source_image

    if not source_face.startswith("http"):
        source_face = open(source_face, "rb")

    if not target_face.startswith("http"):
        target_face = open(target_face, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.arielreplicate, source_path=source_face, target_path=target_face)

    filename = str(uuid.uuid4()) + ".png"
    image_file = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                       hosted_url=output)

    return image_file


def prompt_model_stylegan_nada(timing_uuid, input_image):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.stylegan_nada, input=input_image,
                                            output_style=timing.prompt)
    output = resize_image(timing.project.name, 512, 512, output)

    return output


def prompt_model_stable_diffusion_xl(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    app_setting: InternalSettingObject = data_repo.get_app_setting_from_uuid()
    api_key = app_setting.stability_key_decrypted

    # if the image starts with http, it's a URL, otherwise it's a file path
    if source_image.startswith("http"):
        response = r.get(source_image)
        source_image = Image.open(BytesIO(response.content))
    else:
        source_image = Image.open(source_image)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    source_image.save(temp_file, "PNG")
    temp_file.close()

    source_image.seek(0)

    multipart_data = MultipartEncoder(
        fields={
            "text_prompts[0][text]": timing.prompt,
            "init_image": (os.path.basename(temp_file.name), open(temp_file.name, "rb"), "image/png"),
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": timing.strength,
            "cfg_scale": timing.guidance_scale,
            "clip_guidance_preset": "FAST_BLUE",
            "samples": "1",
            "steps": timing.num_inteference_steps,
        }
    )

    print(multipart_data)

    response = r.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image",
        headers={
            "Content-Type": multipart_data.content_type,
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        data=multipart_data,
    )
    os.unlink(temp_file.name)

    print(response)

    if response.status_code != 200:
        st.error("An error occurred: " + str(response.text))
        time.sleep(5)
        return None
    else:
        data = response.json()
        generated_image = base64.b64decode(data["artifacts"][0]["base64"])
        # generate a random file name with uuid at the location
        filename = str(uuid.uuid4()) + ".png"
        image_file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                                               local_path="videos/" + str(timing.project.name) + "/assets/frames/2_character_pipeline_completed/")

    return image_file


def prompt_model_stability(timing_uuid, input_image_file: InternalFileObject):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    index_of_current_item = timing.aux_frame_index
    input_image = input_image_file.location
    prompt = timing.prompt
    strength = timing.strength
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.img2img_sd_2_1,
        image=input_image,
        prompt_strength=float(strength),
        prompt=prompt,
        negative_prompt=timing.negative_prompt,
        width=project_settings.width,
        height=project_settings.height,
        guidance_scale=timing.guidance_scale,
        seed=timing.seed,
        num_inference_steps=timing.num_inteference_steps
    )

    filename = str(uuid.uuid4()) + ".png"
    image_file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                                           hosted_url=output[0], tag=InternalFileTag.GENERATED_VIDEO.value)

    return image_file


def prompt_model_dreambooth(timing_uuid, source_image_file: InternalFileObject):
    data_repo = DataRepo()

    if not ('dreambooth_model_uuid' in st.session_state and st.session_state['dreambooth_model_uuid']):
        st.error('No dreambooth model selected')
        return

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)
    
    dreambooth_model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(st.session_state['dreambooth_model_uuid'])
    
    model_name = dreambooth_model.name
    image_number = timing.aux_frame_index
    prompt = timing.prompt
    strength = timing.strength
    negative_prompt = timing.negative_prompt
    guidance_scale = timing.guidance_scale
    seed = timing.seed
    num_inference_steps = timing.num_inteference_steps

    model_id = dreambooth_model.replicate_url

    ml_client = get_ml_client()

    source_image = source_image_file.location
    if timing_details[image_number].adapter_type == "Yes":
        if source_image.startswith("http"):
            control_image = source_image
        else:
            control_image = open(source_image, "rb")
    else:
        control_image = None

    # version of models that were custom created has to be fetched
    if not dreambooth_model.version:
        version = ml_client.get_model_version_from_id(model_id)
        data_repo.update_ai_model(uuid=dreambooth_model.uuid, version=version)
    else:
        version = dreambooth_model.version

    model_version = ml_client.get_model_by_name(
        f"{REPLICATE_USER}/{model_name}", version)

    if source_image.startswith("http"):
        input_image = source_image
    else:
        input_image = open(source_image, "rb")

    input_data = {
        "image": input_image,
        "prompt": prompt,
        "prompt_strength": float(strength),
        "height": int(project_settings.height),
        "width": int(project_settings.width),
        "disable_safety_check": True,
        "negative_prompt": negative_prompt,
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps)
    }

    if control_image != None:
        input_data['control_image'] = control_image

    output = model_version.predict(**input_data)

    for i in output:
        filename = str(uuid.uuid4()) + ".png"
        image_file = data_repo.create_file(
            name=filename, type=InternalFileType.IMAGE.value, hosted_url=i, tag=InternalFileTag.GENERATED_VIDEO.value)
        return image_file

    return None


def get_duration_from_video(input_video_file: InternalFileObject):
    input_video = input_video_file.local_path
    video_capture = cv2.VideoCapture(input_video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_duration = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate
    video_capture.release()
    return total_duration


'''
get audio_bytes of correct duration for a given frame
'''


def get_audio_bytes_for_slice(timing_uuid):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    # TODO: add null check for the audio
    audio = AudioSegment.from_file(project_settings.audio.local_path)

    # DOUBT: is it checked if it is the last frame or not?
    audio = audio[timing.frame_time *
                  1000: data_repo.get_next_timing(timing_uuid)['frame_time'] * 1000]
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes.seek(0)
    return audio_bytes


def slice_part_of_video(project_name, index_of_current_item, video_start_percentage, video_end_percentage, slice_name, timing_details):
    input_video = timing_details[int(
        index_of_current_item)]["interpolated_video"]
    total_clip_duration = get_duration_from_video(input_video)
    start_time = float(video_start_percentage) * float(total_clip_duration)
    end_time = float(video_end_percentage) * float(total_clip_duration)
    clip = VideoFileClip(input_video).subclip(
        t_start=start_time, t_end=end_time)
    output_video = "videos/" + \
        str(project_name) + "/assets/videos/0_raw/" + str(slice_name) + ".mp4"
    clip.write_videofile(output_video, audio=False)
    clip.close()


def update_speed_of_video_clip(video_file: InternalFileObject, save_to_new_location, timing_uuid) -> InternalFileObject:
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    desired_duration = timing.clip_duration
    animation_style = timing.animation_style
    location_of_video = video_file.local_path

    temp_video_file, temp_output_file = None, None
    if video_file.hosted_url:
        temp_video_file = generate_temp_file(video_file.hosted_url, '.mp4')

    location_of_video = temp_video_file.name if temp_video_file else video_file.local_path
    

    if animation_style == AnimationStyleType.DIRECT_MORPHING.value:

        # Load the video clip
        clip = VideoFileClip(location_of_video)

        clip = clip.set_fps(120)

        # Calculate the number of frames to keep
        input_duration = clip.duration
        total_frames = len(list(clip.iter_frames()))
        target_frames = int(total_frames * (desired_duration / input_duration))

        # Determine which frames to keep
        keep_every_n_frames = total_frames / target_frames
        frames_to_keep = [int(i * keep_every_n_frames)
                          for i in range(target_frames)]

        # Create a new video clip with the selected frames
        updated_clip = concatenate_videoclips(
            [clip.subclip(i/clip.fps, (i+1)/clip.fps) for i in frames_to_keep])

        file_name = ''.join(random.choices(
                string.ascii_lowercase + string.digits, k=16)) + ".mp4"
        new_file_location = "videos/" + \
            str(timing.project.uuid) + \
            "/assets/videos/1_final/" + str(file_name)
        
        temp_output_file = generate_temp_file(timing.interpolated_clip.hosted_url, '.mp4')

        output_clip.write_videofile(
            filename=temp_output_file.name, codec="libx265")
        
        video_bytes = None
        with open(temp_output_file.name, 'rb') as f:
            video_bytes = f.read()
        
        hosted_url = save_or_host_file_bytes(video_bytes, new_file_location)
        
        if save_to_new_location:
            if hosted_url:
                video_file: InternalFileObject = data_repo.create_file(
                    name=new_file_name, type=InternalFileType.VIDEO.value, hosted_url=hosted_url)
            else:
                video_file: InternalFileObject = data_repo.create_file(
                    name=new_file_name, type=InternalFileType.VIDEO.value, local_path=new_file_location)
        else:
            os.remove(location_of_video)
            location_of_video = new_file_location
            if hosted_url:
                video_file: InternalFileObject = data_repo.create_file(
                    name=new_file_name, type=InternalFileType.VIDEO.value, hosted_url=hosted_url)
            else:
                data_repo.create_or_update_file(
                    video_file.uuid, type=InternalFileType.VIDEO.value, local_path=location_of_video)

    elif animation_style == AnimationStyleType.INTERPOLATION.value:

        clip = VideoFileClip(location_of_video)
        input_video_duration = clip.duration
        desired_duration = timing.clip_duration
        desired_speed_change = float(
            input_video_duration) / float(desired_duration)

        print("Desired Speed Change: " + str(desired_speed_change))

        # Apply the speed change using moviepy
        output_clip = clip.fx(vfx.speedx, desired_speed_change)

        # Save the output video
        new_file_name = ''.join(random.choices(
            string.ascii_lowercase + string.digits, k=16)) + ".mp4"
        new_file_location = "videos/" + \
            str(timing.project.uuid) + \
            "/assets/videos/1_final/" + str(new_file_name)
        
        temp_output_file = generate_temp_file(timing.interpolated_clip.hosted_url, '.mp4')

        output_clip.write_videofile(
            filename=temp_output_file.name, codec="libx264", preset="fast")
        
        video_bytes = None
        with open(temp_output_file.name, 'rb') as f:
            video_bytes = f.read()
        
        hosted_url = save_or_host_file_bytes(video_bytes, new_file_location)

        # TODO: simplify this logic
        if save_to_new_location:
            if hosted_url:
                video_file: InternalFileObject = data_repo.create_file(
                    name=new_file_name, type=InternalFileType.VIDEO.value, hosted_url=hosted_url)
            else:
                video_file: InternalFileObject = data_repo.create_file(
                    name=new_file_name, type=InternalFileType.VIDEO.value, local_path=new_file_location)
        else:
            os.remove(location_of_video)
            location_of_video = new_file_location
            if hosted_url:
                video_file: InternalFileObject = data_repo.create_file(
                    name=new_file_name, type=InternalFileType.VIDEO.value, hosted_url=hosted_url)
            else:
                data_repo.create_or_update_file(
                    video_file.uuid, type=InternalFileType.VIDEO.value, local_path=location_of_video)

    if temp_video_file:
        os.remove(temp_video_file.name)
    
    if temp_output_file:
        os.remove(temp_output_file.name)

    return video_file


'''


def update_video_speed(project_name, index_of_current_item, duration_of_static_time, total_clip_duration, timing_details):

    slice_part_of_video(project_name, index_of_current_item,
                        0, 0.00000000001, "static", timing_details)

    slice_part_of_video(project_name, index_of_current_item,
                        0, 1, "moving", timing_details)

    video_capture = cv2.VideoCapture(
        "videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    total_duration_of_static = video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT) / frame_rate

    desired_speed_change_of_static = float(
        duration_of_static_time) / float(total_duration_of_static)

    update_slice_of_video_speed(
        project_name, "static.mp4", desired_speed_change_of_static)

    video_capture = cv2.VideoCapture(
        "videos/" + str(project_name) + "/assets/videos/0_raw/moving.mp4")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    total_duration_of_moving = video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT) / frame_rate

    total_duration_of_moving = float(total_duration_of_moving)

    total_clip_duration = float(total_clip_duration)

    duration_of_static_time = float(duration_of_static_time)

    desired_speed_change_of_moving = (
        total_clip_duration - duration_of_static_time) / total_duration_of_moving

    update_slice_of_video_speed(
        project_name, "moving.mp4", desired_speed_change_of_moving)

    file_name = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=16)) + ".mp4"

    if duration_of_static_time == 0:

        # shutil.move("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4", "videos/" + str(video_name) + "/assets/videos/1_final/" + str(video_number) + ".mp4")
        os.rename("videos/" + str(project_name) + "/assets/videos/0_raw/moving.mp4",
                  "videos/" + str(project_name) + "/assets/videos/1_final/" + str(file_name))
        os.remove("videos/" + str(project_name) +
                  "/assets/videos/0_raw/static.mp4")
    else:
        final_clip = concatenate_videoclips([VideoFileClip("videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4"),
                                            VideoFileClip("videos/" + str(project_name) + "/assets/videos/0_raw/moving.mp4")])

        final_clip.write_videofile(
            "videos/" + str(project_name) + "/assets/videos/0_raw/full_output.mp4", fps=30)

        os.remove("videos/" + str(project_name) +
                  "/assets/videos/0_raw/moving.mp4")
        os.remove("videos/" + str(project_name) +
                  "/assets/videos/0_raw/static.mp4")
        os.rename("videos/" + str(project_name) + "/assets/videos/0_raw/full_output.mp4",
                  "videos/" + str(file_name))

    update_specific_timing_value(project_name, index_of_current_item, "timed_clip",
                                 "videos/" + str(project_name) + "/assets/videos/1_final/" + str(file_name))

'''

# DOUBT: there is one other method with exact same name, commenting this one
# def calculate_desired_duration_of_each_clip(project: InternalProjectObject):
#     data_repo = DataRepo()

#     timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
#         project.id)

#     for timing in timing_details:
#         total_duration_of_frame = calculate_desired_duration_of_individual_clip(
#             timing.uuid)
#         data_repo.update_specific_timing(
#             timing.uuid, clip_duration=total_duration_of_frame)


'''
gives the time difference between the current and the next keyframe. If it's the
last keyframe then we can set static_time. This time difference is used as the interpolated
clip length
'''


def calculate_desired_duration_of_individual_clip(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    length_of_list = len(timing_details)

    # last frame
    if timing.aux_frame_index == length_of_list - 1:
        time_of_frame = timing.frame_time
        duration_of_static_time = 0.0   # can be changed
        end_duration_of_frame = float(
            time_of_frame) + float(duration_of_static_time)
        total_duration_of_frame = float(
            end_duration_of_frame) - float(time_of_frame)
    else:
        time_of_frame = timing.frame_time
        time_of_next_frame = data_repo.get_next_timing(timing_uuid).frame_time
        total_duration_of_frame = float(
            time_of_next_frame) - float(time_of_frame)

    return total_duration_of_frame


def calculate_desired_duration_of_each_clip(project_uuid):
    data_repo = DataRepo()
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)

    length_of_list = len(timing_details)

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        length_of_list = len(timing_details)
        timing_item: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            project_uuid, index_of_current_item)

        # last frame
        if index_of_current_item == (length_of_list - 1):
            time_of_frame = timing_item.frame_time
            duration_of_static_time = 0.0
            end_duration_of_frame = float(
                time_of_frame) + float(duration_of_static_time)
            total_duration_of_frame = float(
                end_duration_of_frame) - float(time_of_frame)
        else:
            time_of_frame = timing_item.frame_time
            next_timing = data_repo.get_next_timing(timing_item.uuid)
            time_of_next_frame = next_timing.frame_time
            total_duration_of_frame = float(
                time_of_next_frame) - float(time_of_frame)

        duration_of_static_time = 0.0
        duration_of_morph = float(
            total_duration_of_frame) - float(duration_of_static_time)

        data_repo.update_specific_timing(
            timing_item.uuid, clip_duration=total_duration_of_frame)


def hair_swap(source_image, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    app_secret = data_repo.get_app_secrets_from_user_uuid(
        timing.project.user_uuid)

    # DOUBT: what's the video name here?
    video_name = ""
    source_hair = upload_file("videos/" + str(video_name) + "/face.png", app_secret['aws_access_key'],
                              app_secret['aws_secret_key'])

    target_hair = upload_file("videos/" + str(video_name) +
                              "/assets/frames/2_character_pipeline_completed/" +
                              str(timing.aux_frame_index) + ".png",
                              app_secret['aws_access_key'], app_secret['aws_secret_key'])

    if not source_hair.startswith("http"):
        source_hair = open(source_hair, "rb")

    if not target_hair.startswith("http"):
        target_hair = open(target_hair, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.cjwbw_style_hair, source_image=source_hair, target_image=target_hair)

    return output


def prompt_model_depth2img(strength, timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    prompt = timing.prompt
    num_inference_steps = timing.num_inteference_steps
    guidance_scale = timing.guidance_scale
    negative_prompt = timing.negative_prompt
    if not source_image.startswith("http"):
        source_image = open(source_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.jagilley_controlnet_depth2img, input_image=source_image,
                                            prompt_strength=float(strength), prompt=prompt, negative_prompt=negative_prompt,
                                            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    filename = str(uuid.uuid4()) + ".png"
    image_file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                                           hosted_url=output[0])
    return image_file


def prompt_model_blip2(input_image, query):
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.salesforce_blip_2, image=input_image, question=query)

    return output


def facial_expression_recognition(input_image):
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
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


def prompt_model_pix2pix(timing_uuid, input_image_file: InternalFileObject):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    prompt = timing.prompt
    guidance_scale = timing.guidance_scale
    seed = timing.seed
    input_image = input_image_file.location
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.arielreplicate, input_image=input_image, instruction_text=prompt,
                                            seed=seed, cfg_image=1.2, cfg_text=guidance_scale, resolution=704)

    filename = str(uuid.uuid4()) + ".png"
    image_file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                                           hosted_url=output)
    return image_file

# TODO: make this into an unified method, which just takes model_uuid and gives the output
# without need the if-else clause and other methods


def restyle_images(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    model_name = timing.model.name
    strength = timing.strength

    if model_name == "stable-diffusion-img2img-v2.1":
        output_file = prompt_model_stability(timing_uuid, source_image)
    elif model_name == "depth2img":
        output_file = prompt_model_depth2img(
            strength, timing_uuid, source_image)
    elif model_name == "pix2pix":
        output_file = prompt_model_pix2pix(timing_uuid, source_image)
    elif model_name == "LoRA":
        output_file = prompt_model_lora(timing_uuid, source_image)
    elif model_name == "controlnet":
        output_file = prompt_model_controlnet(timing_uuid, source_image)
    elif model_name == "Dreambooth":
        output_file = prompt_model_dreambooth(timing_uuid, source_image)
    elif model_name == 'StyleGAN-NADA':
        output_file = prompt_model_stylegan_nada(timing_uuid, source_image)
    elif model_name == "stable_diffusion_xl":
        output_file = prompt_model_stable_diffusion_xl(
            timing_uuid, source_image)
    elif model_name == "real-esrgan-upscaling":
        output_file = prompt_model_real_esrgan_upscaling(source_image)
    elif model_name == 'controlnet_1_1_x_realistic_vision_v2_0':
        output_file = prompt_model_controlnet_1_1_x_realistic_vision_v2_0(
            source_image)
    elif model_name == 'urpm-v1.3':
        output_file = prompt_model_urpm_v1_3(source_image)

    return output_file


def custom_pipeline_mystique(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    output_image_file: InternalFileObject = face_swap(
        timing_uuid, source_image)
    output_image_file = touch_up_image(output_image_file)
    output_image_file = resize_image(
        timing.project.name, project_settings.width, project_settings.height, output_image_file)

    if timing.model.category == AIModelType.DREAMBOOTH.value:
        output_image_file = prompt_model_dreambooth(
            timing_uuid, output_image_file)
    elif timing.model.category == AIModelType.LORA.value:
        output_image_file = prompt_model_lora(timing_uuid, output_image_file)

    return output_image_file


def create_timings_row_at_frame_number(project_uuid, index_of_frame):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_timing_list_from_project(
        project_uuid)
    
    # remove the interpolated video from the current row and the row before and after - unless it is the first or last row
    timing_data = {
        "project_id": project_uuid,
        "frame_time": 0.0,
        "animation_style": AnimationStyleType.INTERPOLATION.value,
        "aux_frame_index": index_of_frame
    }
    timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)

    prev_timing: InternalFrameTimingObject = data_repo.get_prev_timing(
        timing.uuid)
    if prev_timing:
        data_repo.update_specific_timing(
            prev_timing.uuid, interpolated_clip_id=None)

    next_timing: InternalAIModelObject = data_repo.get_next_timing(timing.uuid)
    if next_timing:
        data_repo.update_specific_timing(
            next_timing.uuid, interpolated_clip_id=None)

    return timing


def get_models() -> List[InternalAIModelObject]:
    # df = pd.read_csv('models.csv')
    # models = df[df.columns[0]].tolist()
    # return models

    # get the default user and all models belonging to him
    data_repo = DataRepo()
    model_list = data_repo.get_all_model_list()

    return model_list


def find_clip_duration(timing_uuid, total_number_of_videos):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    total_clip_duration = timing.clip_duration
    total_clip_duration = float(total_clip_duration)

    if timing.aux_frame_index == total_number_of_videos:
        total_clip_duration = timing.clip_duration
        # duration_of_static_time = float(timing_details[timing.aux_frame_index]['static_time'])
        # duration_of_static_time = float(duration_of_static_time) / 2
        duration_of_static_time = 0

    elif timing.aux_frame_index == 0:
        # duration_of_static_time = float(timing_details[timing.aux_frame_index]['static_time'])
        duration_of_static_time = 0
    else:
        # duration_of_static_time = float(timing_details[timing.aux_frame_index]['static_time'])
        duration_of_static_time = 0

    return total_clip_duration, duration_of_static_time


def add_audio_to_video_slice(video_file, audio_bytes):
    video_location = video_file.local_path
    # Save the audio bytes to a temporary file
    audio_file = "temp_audio.wav"
    with open(audio_file, 'wb') as f:
        f.write(audio_bytes.getvalue())

    # Create an input video stream
    video_stream = ffmpeg.input(video_location)

    # Create an input audio stream
    audio_stream = ffmpeg.input(audio_file)

    # Add the audio stream to the video stream
    output_stream = ffmpeg.output(video_stream, audio_stream, "output_with_audio.mp4",
                                  vcodec='copy', acodec='aac', strict='experimental')

    # Run the ffmpeg command
    output_stream.run()

    # Remove the original video file and the temporary audio file
    os.remove(video_location)
    os.remove(audio_file)

    # TODO: handle online update in this case
    # Rename the output file to have the same name as the original video file
    os.rename("output_with_audio.mp4", video_location)


def calculate_desired_speed_change(input_video_location, target_duration):
    # Load the video clip
    input_clip = VideoFileClip(input_video_location)

    # Get the duration of the input video clip
    input_duration = input_clip.duration

    # Calculate the desired speed change
    desired_speed_change = target_duration / input_duration

    return desired_speed_change


def get_actual_clip_duration(clip_location):
    # Load the video clip
    clip = VideoFileClip(clip_location)

    # Get the duration of the video clip
    duration = clip.duration

    rounded_duration = round(duration, 5)

    return rounded_duration


def calculate_dynamic_interpolations_steps(clip_duration):

    if clip_duration < 0.17:
        interpolation_steps = 2
    elif clip_duration < 0.3:
        interpolation_steps = 3
    elif clip_duration < 0.57:
        interpolation_steps = 4
    elif clip_duration < 1.1:
        interpolation_steps = 5
    elif clip_duration < 2.17:
        interpolation_steps = 6
    elif clip_duration < 4.3:
        interpolation_steps = 7
    else:
        interpolation_steps = 8
    return interpolation_steps


def render_video(final_video_name, project_uuid, quality):
    data_repo = DataRepo()

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)

    calculate_desired_duration_of_each_clip(project_uuid)

    total_number_of_videos = len(timing_details) - 1

    for i in range(0, total_number_of_videos):
        index_of_current_item = i
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            project_uuid, i)

        timing = timing_details[i]
        if quality == VideoQuality.HIGH.value:
            data_repo.update_specific_timing(
                current_timing.uuid, timed_clip_id=None)
            interpolation_steps = calculate_dynamic_interpolations_steps(
                timing_details[index_of_current_item].clip_duration)
            if not timing.interpolation_steps or timing.interpolation_steps < interpolation_steps:
                data_repo.update_specific_timing(
                    current_timing.uuid, interpolation_steps=interpolation_steps, interpolated_clip_id=None)
        else:
            if not timing.interpolation_steps or timing.interpolation_steps < 3:
                data_repo.update_specific_timing(
                    current_timing.uuid, interpolation_steps=3)

        if not timing.interpolated_clip:
            # DOUBT: if and else condition had the same code
            # if total_number_of_videos == index_of_current_item:
            #     video_location = create_individual_clip(
            #         index_of_current_item, project_name)
            #     update_specific_timing_value(
            #         project_name, index_of_current_item, "interpolated_video", video_location)
            # else:
            video_location = create_individual_clip(current_timing.uuid)
            data_repo.update_specific_timing(
                current_timing.uuid, interpolated_clip_id=video_location.uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    total_number_of_videos = len(timing_details) - 2

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        timing = timing_details[index_of_current_item]
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            timing.project.uuid, index_of_current_item)
        if index_of_current_item <= total_number_of_videos:
            if not current_timing.timed_clip:
                desired_duration = current_timing.clip_duration
                location_of_input_video_file = current_timing.interpolated_clip

                output_video = update_speed_of_video_clip(
                    location_of_input_video_file, True, timing.uuid)

                if quality == VideoQuality.PREVIEW.value:
                    print("")
                    '''
                    clip = VideoFileClip(location_of_output_video)

                    number_text = TextClip(str(index_of_current_item), fontsize=24, color='white')
                    number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(number_text.w + 10, number_text.h + 10))
                    number_background = number_background.set_position(('right', 'bottom')).set_duration(clip.duration)
                    number_text = number_text.set_position((number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)

                    clip_with_number = CompositeVideoClip([clip, number_background, number_text])

                    # remove existing preview video
                    os.remove(location_of_output_video)
                    clip_with_number.write_videofile(location_of_output_video, codec='libx264', bitrate='3000k')
                    '''

                data_repo.update_specific_timing(
                    current_timing.uuid, timed_clip_id=output_video.uuid)

    video_list = []
    temp_file_list = []

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    # TODO: CORRECT-CODE
    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            project_uuid, index_of_current_item)
        if index_of_current_item <= total_number_of_videos:
            temp_video_file = None
            if current_timing.timed_clip.hosted_url:
                temp_video_file = generate_temp_file(current_timing.timed_clip.hosted_url, '.mp4')
                temp_file_list.append(temp_video_file)

            file_path = temp_video_file.name if temp_video_file else current_timing.timed_clip.local_path
        
            video_list.append(file_path)

    video_clip_list = [VideoFileClip(v) for v in video_list]
    finalclip = concatenate_videoclips(video_clip_list)

    output_video_file = f"videos/{timing.project.uuid}/assets/videos/2_completed/{final_video_name}.mp4"
    if project_settings.audio:
        temp_audio_file = None
        if 'http' in project_settings.audio.location:
            temp_audio_file = generate_temp_file(project_settings.audio.location, '.mp4')
            temp_file_list.append(temp_audio_file)

        audio_location = temp_audio_file.name if temp_audio_file else project_settings.audio.location
        
        audio_clip = AudioFileClip(audio_location)
        finalclip = finalclip.set_audio(audio_clip)

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    finalclip.write_videofile(
        temp_video_file.name,
        fps=60,  # or 60 if your original video is 60fps
        audio_bitrate="128k",
        bitrate="5000k",
        codec="libx264",
        audio_codec="aac"
    )

    temp_video_file.close()
    video_bytes = None
    with open(temp_video_file.name, "rb") as f:
        video_bytes = f.read()
    
    hosted_url = save_or_host_file_bytes(video_bytes, output_video_file)

    file_data = {
        "name": final_video_name,
        "type": InternalFileType.VIDEO.value,
        "tag": InternalFileTag.GENERATED_VIDEO.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": output_video_file})

    data_repo.create_file(**file_data)

    for file in temp_file_list:
        os.remove(file.name)


def create_gif_preview(project_uuid):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(
        project_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)
    list_of_images = []

    for i in timing_details:
        # make index_of_current_item the index of the current item
        index_of_current_item = timing_details.index(i)
        source_image = timing_details[index_of_current_item].primary_image_location
        list_of_images.append(source_image)

    frames = []
    for url in list_of_images:
        response = r.get(url)
        frame = Image.open(BytesIO(response.content))
        draw = ImageDraw.Draw(frame)
        font_url = 'https://banodoco.s3.amazonaws.com/training_data/arial.ttf'
        font_file = "arial.ttf"
        urllib.request.urlretrieve(font_url, font_file)
        font = ImageFont.truetype(font_file, 40)
        index_of_current_item = list_of_images.index(url)
        draw.text((frame.width - 60, frame.height - 60),
                  str(index_of_current_item), font=font, fill=(255, 255, 255, 255))
        frames.append(np.array(frame))
    imageio.mimsave(f'videos/{project.name}/preview_gif.gif', frames, fps=0.5)

    # TODO: a function to save temp images


def create_depth_mask_image(input_image, layer, timing_uuid):
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.cjwbw_midas, image=input_image, model_type="dpt_beit_large_512")
    try:
        urllib.request.urlretrieve(output, "videos/temp/depth.png")
    except Exception as e:
        print(e)

    depth_map = Image.open("videos/temp/depth.png")
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


def prompt_model_controlnet(timing_uuid, input_image):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if timing.adapter_type == "normal":
        model = REPLICATE_MODEL.jagilley_controlnet_normal
    elif timing.adapter_type == "canny":
        model = REPLICATE_MODEL.jagilley_controlnet_canny
    elif timing.adapter_type == "hed":
        model = REPLICATE_MODEL.jagilley_controlnet_hed
    elif timing.adapter_type == "scribble":
        model = REPLICATE_MODEL.jagilley_controlnet_scribble
        if timing.canny_image != "":
            input_image = timing.canny_image
    elif timing.adapter_type == "seg":
        model = REPLICATE_MODEL.jagilley_controlnet_seg
    elif timing.adapter_type == "hough":
        model = REPLICATE_MODEL.jagilley_controlnet_hough
    elif timing.adapter_type == "depth2img":
        model = REPLICATE_MODEL.jagilley_controlnet_depth2img
    elif timing.adapter_type == "pose":
        model = REPLICATE_MODEL.jagilley_controlnet_pose

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    inputs = {
        'image': input_image,
        'prompt': timing.prompt,
        'num_samples': "1",
        'image_resolution': "512",
        'ddim_steps': timing.num_inteference_steps,
        'scale': timing.guidance_scale,
        'eta': 0,
        'seed': timing.seed,
        'a_prompt': "best quality, extremely detailed",
        'n_prompt': timing.negative_prompt + ", longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        'detect_resolution': 512,
        'bg_threshold': 0,
        'low_threshold': timing.low_threshold,
        'high_threshold': timing.high_threshold,
    }

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(model, **inputs)

    return output[1]


def prompt_model_urpm_v1_3(timing_uuid, source_image):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    if not source_image.startswith("http"):
        source_image = open(source_image, "rb")

    inputs = {
        'image': source_image,
        'prompt': timing.prompt,
        'negative_prompt': timing.negative_prompt,
        'strength': timing.strength,
        'guidance_scale': timing.guidance_scale,
        'num_inference_steps': timing.num_inference_steps,
        'upscale': 1,
        'seed': timing.seed,
    }

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.urpm, **inputs)

    return output[0]


def prompt_model_controlnet_1_1_x_realistic_vision_v2_0(timing_uuid, input_image):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    inputs = {
        'image': input_image,
        'prompt': timing.prompt,
        'ddim_steps': timing.num_inference_steps,
        'strength': timing.strength,
        'scale': timing.guidance_scale,
        'seed': timing.seed,
    }

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(
        REPLICATE_MODEL.controlnet_1_1_x_realistic_vision_v2_0, **inputs)

    return output[1]


def prompt_model_lora(timing_uuid, source_image_file: InternalFileObject) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    # lora_models = timing.custom_model_id_list
    # default_model_url = DEFAULT_LORA_MODEL_URL

    # lora_model_urls = []

    # for lora_model_uuid in lora_models:
    #     if lora_model_uuid != "":
    #         lora_model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(
    #             lora_model_uuid)
    #         print(lora_model)
    #         if lora_model.replicate_url != "":
    #             lora_model_url = lora_model.replicate_url
    #         else:
    #             lora_model_url = default_model_url
    #     else:
    #         lora_model_url = default_model_url

    #     lora_model_urls.append(lora_model_url)

    lora_urls = ""
    lora_scales = ""
    if "lora_model_1_url" in st.session_state and st.session_state["lora_model_1_url"]:
        lora_urls += st.session_state["lora_model_1_url"]
        lora_scales += "0.5"
    if "lora_model_2_url" in st.session_state and st.session_state["lora_model_2_url"]:
        ctn = "" if not len(lora_urls) else " | "
        lora_urls += ctn + st.session_state["lora_model_2_url"]
        lora_scales += ctn + "0.5"
    if st.session_state["lora_model_3_url"]:
        ctn = "" if not len(lora_urls) else " | "
        lora_urls += ctn + st.session_state["lora_model_3_url"]
        lora_scales += ctn + "0.5"

    source_image = source_image_file.location
    if source_image[:4] == "http":
        input_image = source_image
    else:
        input_image = open(source_image, "rb")

    if timing.adapter_type != "None":
        if source_image[:4] == "http":
            adapter_condition_image = source_image
        else:
            adapter_condition_image = open(source_image, "rb")
    else:
        adapter_condition_image = ""

    inputs = {
        'prompt': timing.prompt,
        'negative_prompt': timing.negative_prompt,
        'width': project_settings.width,
        'height': project_settings.height,
        'num_outputs': 1,
        'image': input_image,
        'num_inference_steps': timing.num_inteference_steps,
        'guidance_scale': timing.guidance_scale,
        'prompt_strength': timing.strength,
        'scheduler': "DPMSolverMultistep",
        'lora_urls': lora_urls,
        'lora_scales': lora_scales,
        'adapter_type': timing.adapter_type,
        'adapter_condition_image': adapter_condition_image,
    }

    ml_client = get_ml_client()
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        try:
            output = ml_client.predict_model_output(
                REPLICATE_MODEL.clones_lora_training_2, **inputs)
            print(output)
            filename = str(uuid.uuid4()) + ".png"
            file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                                             hosted_url=output[0])
            return file
        except replicate.exceptions.ModelError as e:
            if "NSFW content detected" in str(e):
                print("NSFW content detected. Attempting to rerun code...")
                attempts += 1
                continue
            else:
                raise e
        except Exception as e:
            raise e

    # filename = "default_3x_failed-656a7e5f-eca9-4f92-a06b-e1c6ff4a5f5e.png"     # const filename
    # file = data_repo.get_file_from_name(filename)
    # if file:
    #     return file
    # else:
    #     file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
    #                                                      hosted_url="https://i.ibb.co/ZG0hxzj/Failed-3x-In-A-Row.png")
    #     return file







def save_audio_file(uploaded_file, project_uuid):
    data_repo = DataRepo()

    local_file_location = os.path.join(
        f"videos/{project_uuid}/assets/resources/audio", uploaded_file.name)

    audio_bytes = uploaded_file.read()
    hosted_url = save_or_host_file_bytes(audio_bytes, local_file_location, ".mp3")

    file_data = {
        "name": str(uuid.uuid4()) + ".mp3",
        "type": InternalFileType.AUDIO.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": local_file_location})

    audio_file: InternalFileObject = data_repo.create_file(
        **file_data)
    data_repo.update_project_setting(
        project_uuid, audio_id=audio_file.uuid)
    
    return audio_file
    

def attach_audio_element(project_uuid, expanded):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(
        uuid=project_uuid)
    project_setting: InternalSettingObject = data_repo.get_project_setting(
        project_id=project_uuid)

    with st.expander("Audio"):
        uploaded_file = st.file_uploader("Attach audio", type=[
                                         "mp3"], help="This will attach this audio when you render a video")
        if st.button("Upload and attach new audio"):
            if uploaded_file:
                save_audio_file(uploaded_file, project_uuid)
                st.experimental_rerun()
            else:
                st.warning('No file selected')

        if project_setting.audio:
            # TODO: store "extracted_audio.mp3" in a constant
            if project_setting.audio.name == "extracted_audio.mp3":
                st.info("You have attached the audio from the video you uploaded.")

            if project_setting.audio.location:
                st.audio(project_setting.audio.location)



def execute_image_edit(type_of_mask_selection, type_of_mask_replacement,
                       background_image, editing_image, prompt, negative_prompt,
                       width, height, layer, timing_uuid) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project = timing.project
    app_secret = data_repo.get_app_secrets_from_user_uuid(
        timing.project.user_uuid)
    index_of_current_item = timing.aux_frame_index

    if type_of_mask_selection == "Automated Background Selection":
        removed_background = remove_background(editing_image)
        response = r.get(removed_background)
        img = Image.open(BytesIO(response.content))
        hosted_url = save_or_host_file(img, SECOND_MASK_FILE_PATH)
        if hosted_url:
            add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_url)

        if type_of_mask_replacement == "Replace With Image":
            edited_image = replace_background(project.uuid, background_image)

        elif type_of_mask_replacement == "Inpainting":
            if SERVER == ServerType.DEVELOPMENT.value:
                image = Image.open(SECOND_MASK_FILE_PATH)
            else:
                path = project.get_temp_mask_file(SECOND_MASK_FILE).location
                response = r.get(path)
                image = Image.open(BytesIO(response.content))

            converted_image = Image.new("RGB", image.size, (255, 255, 255))
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    if pixel[3] == 0:
                        converted_image.putpixel((x, y), (0, 0, 0))
                    else:
                        converted_image.putpixel((x, y), (255, 255, 255))
            create_or_update_mask(timing_uuid, converted_image)
            edited_image = inpainting(
                editing_image, prompt, negative_prompt, timing.uuid, True)

    elif type_of_mask_selection == "Manual Background Selection":
        if type_of_mask_replacement == "Replace With Image":
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content))
            else:
                bg_img = Image.open(editing_image)

            mask_location = timing.mask.location
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                mask_img = Image.open(BytesIO(response.content))
            else:
                mask_img = Image.open(mask_location)

            # TODO: fix this logic, if the uploaded image and the image to be editted are of different sizes then
            # this code will cause issues
            result_img = Image.new("RGBA", bg_img.size, (255, 255, 255, 0))
            for x in range(bg_img.size[0]):
                for y in range(bg_img.size[1]):
                    if x < mask_img.size[0] and y < mask_img.size[1]:
                        if mask_img.getpixel((x, y)) == (0, 0, 0, 255):
                            result_img.putpixel((x, y), (255, 255, 255, 0))
                        else:
                            result_img.putpixel((x, y), bg_img.getpixel((x, y)))
            
            hosted_manual_bg_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
            if hosted_manual_bg_url:
                add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_manual_bg_url)
            edited_image = replace_background(
                project.uuid, SECOND_MASK_FILE_PATH, background_image)
        elif type_of_mask_replacement == "Inpainting":
            mask_location = timing.mask.location
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                im = Image.open(BytesIO(response.content))
            else:
                im = Image.open(mask_location)
            if "A" in im.getbands():
                mask = Image.new('RGB', (width, height), color=(255, 255, 255))
                mask.paste(im, (0, 0), im)
                create_or_update_mask(timing.uuid, mask)
            edited_image = inpainting(
                editing_image, prompt, negative_prompt, timing_uuid, True)
    elif type_of_mask_selection == "Automated Layer Selection":
        mask_location = create_depth_mask_image(
            editing_image, layer, timing.uuid)
        if type_of_mask_replacement == "Replace With Image":
            if mask_location.startswith("http"):
                mask = Image.open(
                    BytesIO(r.get(mask_location).content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new(
                'RGBA', bg_img.size, (0, 0, 0, 0)), mask)
            hosted_automated_bg_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
            if hosted_automated_bg_url:
                add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_automated_bg_url)
            edited_image = replace_background(
                project.uuid, SECOND_MASK_FILE_PATH, background_image)
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(
                editing_image, prompt, negative_prompt, timing_uuid, True)

    elif type_of_mask_selection == "Re-Use Previous Mask":
        mask_location = timing.mask.location
        if type_of_mask_replacement == "Replace With Image":
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                mask = Image.open(BytesIO(response.content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new(
                'RGBA', bg_img.size, (0, 0, 0, 0)), mask)
            hosted_image_replace_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
            if hosted_image_replace_url:
                add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_image_replace_url)
            
            edited_image = replace_background(
                project.uuid, SECOND_MASK_FILE_PATH, background_image)
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(
                editing_image, prompt, negative_prompt, timing_uuid, True)

    elif type_of_mask_selection == "Invert Previous Mask":
        if type_of_mask_replacement == "Replace With Image":
            mask_location = timing.mask.location
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                mask = Image.open(BytesIO(response.content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            inverted_mask = ImageOps.invert(mask)
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new(
                'RGBA', bg_img.size, (0, 0, 0, 0)), inverted_mask)
            # TODO: standardise temproray fixes
            hosted_prvious_invert_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
            if hosted_prvious_invert_url:
                add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_prvious_invert_url)
            
            edited_image = replace_background(
                project.uuid, SECOND_MASK_FILE_PATH, background_image)
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(
                editing_image, prompt, negative_prompt, timing_uuid, False)

    return edited_image


def page_switcher(pages, page):
    section = [section["section_name"]
               for section in pages if page in section["pages"]][0]
    index_of_section = [section["section_name"]
                        for section in pages].index(section)
    index_of_page_in_section = pages[index_of_section]["pages"].index(page)

    return index_of_page_in_section, index_of_section
