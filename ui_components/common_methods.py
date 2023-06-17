from typing import List
import streamlit as st
import os
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
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
import json
import tempfile
import boto3
import time
import zipfile
from math import cos, sin, ceil, radians, gcd
import random
import uuid
from io import BytesIO
import ast
import numpy as np
from shared.constants import AIModelType, InternalFileTag, InternalFileType
from pydub import AudioSegment
import shutil
from moviepy.editor import concatenate_videoclips, TextClip, VideoFileClip, vfx
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from backend.models import InternalFileObject
from shared.file_upload.s3 import upload_file
from shared.utils import is_online_file_path
from ui_components.constants import VideoQuality, WorkflowStageType
from ui_components.models import InternalAIModelObject, InternalAppSettingObject, InternalBackupObject, InternalFrameTimingObject, InternalProjectObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo
from shared.constants import InternalResponse, AnimationStyleType
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import DEFAULT_LORA_MODEL_URL, REPLICATE_MODEL
from ui_components.models import InternalFileObject
import utils.local_storage.local_storage as local_storage


from moviepy.video.fx.all import speedx
import moviepy.editor
from streamlit_cropper import st_cropper

from utils.ml_processor.replicate.replicate import ReplicateProcessor


def resize_and_rotate_element(stage, project_uuid):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(project_uuid)
    timing_details: InternalFrameTimingObject = data_repo.get_timing_list_from_project(project.uuid)

    if "rotated_image" not in st.session_state:
        st.session_state['rotated_image'] = ""

    with st.expander("Zoom Image"):
        select1, select2 = st.columns([2, 1])
        with select1:
            rotation_angle = st.number_input(
                "Rotate image by: ", 0.0, 360.0, 0.0)
            zoom_value = st.number_input("Zoom image by: ", 0.1, 5.0, 1.0)
        with select2:
            fill_with = st.radio("Fill blank space with: ", ["Blur", None])
        if st.button("Rotate Image"):
            if stage == WorkflowStageType.SOURCE.value:
                res = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                input_image = res.source_image
            elif stage == WorkflowStageType.STYLED.value:
                current_frame = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                input_image = current_frame.primary_image
            
            if input_image:
                unique_filename = str(uuid.uuid4())
                temp_local_location = f'videos/temp/{unique_filename}.png'
                # if rotation_angle != 0:
                st.session_state['rotated_image'] = rotate_image(input_image.location, rotation_angle)
                st.session_state['rotated_image'].save(temp_local_location)
                # else:
                #     st.session_state['rotated_image'] = input_image.location
                #     if st.session_state['rotated_image'].startswith("http"):
                #         st.session_state['rotated_image'] = r.get(
                #             st.session_state['rotated_image'])
                #         st.session_state['rotated_image'] = Image.open(
                #             BytesIO(st.session_state['rotated_image'].content))
                #     else:
                #         st.session_state['rotated_image'] = Image.open(
                #             st.session_state['rotated_image'])

                #     st.session_state['rotated_image'].save(temp_local_location)

                if zoom_value != 1.0:
                    st.session_state['rotated_image'] = zoom_image(temp_local_location, zoom_value, fill_with)
                
        if st.session_state['rotated_image'] != "":
            st.image(st.session_state['rotated_image'], caption="Rotated image", width=300)

            btn1, btn2 = st.columns(2)
            with btn1:
                if st.button("Save image", type="primary"):
                    file_name = str(uuid.uuid4()) + ".png"

                    if stage == WorkflowStageType.SOURCE.value:
                        time.sleep(1)
                        save_location = f"videos/{project.uuid}/assets/frames/1_selected/{file_name}"
                        st.session_state['rotated_image'].save(save_location)

                        current_frame = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                        source_image = current_frame.source_image

                        # if source image is already present then updating it
                        if source_image:
                            data_repo.update_file(source_image.uuid, local_path=save_location)
                        # or else creating a new image
                        else:
                            file_data = {
                                "name": str(uuid.uuid4()) + ".png",
                                "type": InternalFileType.IMAGE.value,
                                "local_path": save_location,
                                "project_id": project_uuid
                            }
                            file: InternalFileObject = data_repo.create_file(**file_data)
                            data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=file.uuid)

                        st.session_state['rotated_image'] = ""
                        st.experimental_rerun()

                    elif stage == WorkflowStageType.STYLED.value:
                        save_location = f"videos/{project.uuid}/assets/frames/2_character_pipeline_completed/{file_name}"
                        st.session_state['rotated_image'].save(save_location)

                        file_data = {
                            "name": str(uuid.uuid4()) + ".png",
                            "type": InternalFileType.IMAGE.value,
                            "local_path": save_location,
                            "project_id": project_uuid
                        }
                        file: InternalFileObject = data_repo.create_file(**file_data)

                        number_of_image_variants = add_image_variant(file.uuid, st.session_state['current_frame_uuid'])
                        promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                        st.session_state['rotated_image'] = ""
                        st.experimental_rerun()
            with btn2:
                if st.button("Clear Current Image"):
                    st.session_state['rotated_image'] = ""
                    st.experimental_rerun()


def create_individual_clip(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    img1 = data_repo.get_primary_variant_location(timing_uuid)
    next_timing = data_repo.get_next_timing(timing_uuid)
    img2 = data_repo.get_primary_variant_location(next_timing.uuid)

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

    video_file = data_repo.create_file(name=file_name, type=InternalFileType.VIDEO.value, \
                                       hosted_url=output, local_path=video_location, project_id=timing.project.uuid,\
                                        tag=InternalFileTag.GENERATED_VIDEO.value)

    return video_file


def create_video_without_interpolation(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    image_path_or_url = data_repo.get_primary_variant_location(timing_uuid)

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


def zoom_image(location, zoom_factor, fill_with=None):
    blur_radius = 5
    edge_blur_radius = 15

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0")

    # Check if the provided location is a URL
    if location.startswith('http') or location.startswith('https'):
        response = r.get(location)
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(f"File not found: {location}")
        image = Image.open(location)

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


def crop_image_element(stage):
    timing_uuid = st.session_state['current_frame_uuid']
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    if not timing.source_image:
        st.error("Please select a source image before cropping")
        return
    else:
        if stage == WorkflowStageType.SOURCE.value:
            input_image = timing.source_image.location
        elif stage == WorkflowStageType.STYLED.value:
            input_image = data_repo.get_primary_variant_location(timing_uuid)

        if 'current_working_image_number' not in st.session_state:
            st.session_state['current_working_image_number'] = st.session_state['current_frame_uuid']

        def get_working_image():
            st.session_state['working_image'] = get_pillow_image(input_image)
            st.session_state['working_image'] = ImageOps.expand(
                st.session_state['working_image'], border=200, fill="black")
            st.session_state['current_working_image_number'] = st.session_state['current_frame_uuid']

        if 'working_image' not in st.session_state or st.session_state['current_working_image_number'] != st.session_state['current_frame_uuid']:
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
                if st.button("Reset image"):
                    st.session_state['degree'] = 0
                    get_working_image()
                    st.session_state['degrees_rotated_to'] = 0
                    st.experimental_rerun()

        with options2:
            if st.button("Flip horizontally", key="cropbtn1"):
                st.session_state['working_image'] = st.session_state['working_image'].transpose(
                    Image.FLIP_LEFT_RIGHT)

                # save
            if st.button("Flip vertically", key="cropbtn2"):
                st.session_state['working_image'] = st.session_state['working_image'].transpose(
                    Image.FLIP_TOP_BOTTOM)

        with option3:
            brightness_factor = st.slider("Brightness", 0.0, 2.0, 1.0)
            if brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(
                    st.session_state['working_image'])
                st.session_state['working_image'] = enhancer.enhance(
                    brightness_factor)
        with option4:
            contrast_factor = st.slider("Contrast", 0.0, 2.0, 1.0)
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(
                    st.session_state['working_image'])
                st.session_state['working_image'] = enhancer.enhance(
                    contrast_factor)

        project_settings: InternalProjectObject = data_repo.get_project_settings(
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
                    if stage == "Source":
                        # resize the image to the original width and height
                        cropped_img = cropped_img.resize(
                            (width, height), Image.ANTIALIAS)
                        # generate a random filename and save it to /temp
                        file_name = f"temp/{uuid.uuid4()}.png"
                        cropped_img.save(file_name)
                        st.success("Cropped Image Saved Successfully")
                        data_repo.update_specific_timing(
                            st.session_state['current_frame_uuid'], source_image=file_name)
                        time.sleep(1)
                    st.experimental_rerun()
            with cropbtn2:
                st.warning("Warning: This will overwrite the original image")

            with st.expander("Inpaint in black space"):
                inpaint_prompt = st.text_area(
                    "Prompt", value=project_settings.last_prompt)
                inpaint_negative_prompt = st.text_input(
                    "Negative Prompt", value='branches, frame, fractals, text' + project_settings.last_negative_prompt)
                if 'inpainted_image' not in st.session_state:
                    st.session_state['inpainted_image'] = ""
                if st.button("Inpaint"):
                    saved_cropped_img = cropped_img.resize(
                        (width, height), Image.ANTIALIAS)
                    saved_cropped_img.save("temp/cropped.png")
                    # Convert image to grayscale
                    # Create a new image with the same size as the cropped image
                    mask = Image.new('RGB', cropped_img.size)

                    # Get the width and height of the image
                    width, height = cropped_img.size

                    for x in range(width):
                        for y in range(height):
                            # Get the RGB values of the pixel
                            r, g, b = cropped_img.getpixel((x, y))

                            # If the pixel is black, set it and its adjacent pixels to black in the new image
                            if r == 0 and g == 0 and b == 0:
                                mask.putpixel((x, y), (0, 0, 0))  # Black
                                # Adjust these values to change the range of adjacent pixels
                                for i in range(-2, 3):
                                    for j in range(-2, 3):
                                        # Check that the pixel is within the image boundaries
                                        if 0 <= x + i < width and 0 <= y + j < height:
                                            mask.putpixel(
                                                (x + i, y + j), (0, 0, 0))  # Black
                            # Otherwise, make the pixel white in the new image
                            else:
                                mask.putpixel((x, y), (255, 255, 255))  # White
                    # Save the mask image
                    mask.save('temp/mask.png')

                    st.session_state['inpainted_image'] = inpainting(
                        "temp/cropped.png", inpaint_prompt, inpaint_negative_prompt, st.session_state['current_frame_uuid'], True, pass_mask=True)

                if st.session_state['inpainted_image'] != "":
                    st.image(st.session_state['inpainted_image'],
                             caption="Inpainted Image", use_column_width=True, width=200)
                    if st.button("Make Source Image"):
                        data_repo.update_specific_timing(
                            st.session_state['current_frame_uuid'], source_image=st.session_state['inpainted_image'])
                        st.session_state['inpainted_image'] = ""
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


'''
returns the timed_clip, which is the interpolated video with correct length
'''
def create_or_get_single_preview_video(timing_uuid):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    project_details: InternalSettingObject = data_repo.get_project_setting(timing.project.uuid)

    if not timing.interpolated_clip:
        data_repo.update_specific_timing(timing_uuid, interpolation_steps=3)
        interpolated_video: InternalFileObject = prompt_interpolation_model(timing_uuid)
        data_repo.update_specific_timing(
            timing_uuid, interpolated_clip_id=interpolated_video.uuid)

    if not timing.timed_clip:
        # timed_clip has the correct length (equal to the time difference between the current and the next frame)
        # which the interpolated video may or maynot have
        clip_duration = calculate_desired_duration_of_individual_clip(timing_uuid)
        data_repo.update_specific_timing(timing_uuid, clip_duration=clip_duration)
        
        # TODO: fix refetching of variables
        timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
        output_video = update_speed_of_video_clip(
            timing.interpolated_clip, True, timing_uuid)
        data_repo.update_specific_timing(
            timing_uuid, timed_clip_id=output_video.uuid)

    # adding audio if the audio file is present
    if project_details.audio:
        audio_bytes = get_audio_bytes_for_slice(timing_uuid)
        add_audio_to_video_slice(timing.timed_clip, audio_bytes)

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    return timing.timed_clip


def single_frame_time_changer(timing_uuid):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    frame_time = st.number_input("Frame time (secs):", min_value=0.0, max_value=100.0,
                                 value=timing.frame_time, step=0.1, key=f"frame_time_{timing.aux_frame_index}")

    if frame_time != timing.frame_time:
        data_repo.update_specific_timing(timing_uuid, frame_time=frame_time)
        if timing.aux_frame_index != 0:
            prev_timing = data_repo.get_prev_timing(timing_uuid)
            data_repo.update_specific_timing(prev_timing.uuid, timed_clip_id=None)

        data_repo.update_specific_timing(timing_uuid, timed_clip_id=None)

        # if the frame time of this frame is more than the frame time of the next frame,
        # then we need to update the next frame's frame time, and all the frames after that
        # - shift them by the difference between the new frame time and the old frame time

        next_timing = data_repo.get_next_timing(timing_uuid)
        if next_timing and frame_time > next_timing.frame_time:
            for a in range(timing.aux_frame_index, len(timing_details)):
                frame = timing_details[a]
                this_frame_time = frame.frame_time
                # shift them by the difference between the new frame time and the old frame time
                new_frame_time = this_frame_time + \
                    (frame_time - timing.frame_time)
                data_repo.update_specific_timing(frame.uuid, frame_time=new_frame_time)
                data_repo.update_specific_timing(frame.uuid, timed_clip_id=None)
        st.experimental_rerun()


'''
preview_clips have frame numbers on them. Preview clip is generated from index-2 to index+2 frames
'''
def create_full_preview_video(timing_uuid, speed) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    index_of_item = timing.aux_frame_index

    num_timing_details = len(timing_details)
    clips = []

    print("HERE'S THE SHIT")

    print(f"index_of_item: {index_of_item}, num_timing_details: {num_timing_details}")

    for i in range(index_of_item - 2, index_of_item + 3):
        print(f"i: {i}")
        if i < 0 or i >= num_timing_details-1:
            continue

        primary_variant_location = data_repo.get_primary_variant_location(timing_details[i].uuid)

        print(f"primary_variant_location for i={i}: {primary_variant_location}")

        if not primary_variant_location:
            break

        preview_video = create_or_get_single_preview_video(timing_details[i].uuid)

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
        if preview_video.local_path:
            os.remove(preview_video.local_path)
        clip_with_number.write_videofile(
            preview_video.location, codec='libx264', bitrate='3000k')
        clips.append(preview_video)

    print(clips)
    video_clips = [VideoFileClip(v.location) for v in clips]
    combined_clip = concatenate_videoclips(video_clips)
    output_filename = str(uuid.uuid4()) + ".mp4"
    video_location = f"videos/{timing.project.uuid}/assets/videos/1_final/{output_filename}"
    combined_clip.write_videofile(video_location)

    if speed != 1.0:
        clip = VideoFileClip(video_location)
        output_clip = clip.fx(vfx.speedx, speed)
        os.remove(video_location)
        output_clip.write_videofile(video_location, codec="libx264", preset="fast")

    video_file = data_repo.create_file(name=output_filename, type=InternalFileType.VIDEO.value, local_path=video_location)

    return video_file


def back_and_forward_buttons():
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        st.session_state['current_frame_uuid'])
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    smallbutton0, smallbutton1, smallbutton2, smallbutton3, smallbutton4 = st.columns([
                                                                                      2, 2, 2, 2, 2])
    with smallbutton0:
        if timing.aux_frame_index > 1:
            if st.button(f"{timing.aux_frame_index-2} ⏮️", key=f"Previous Previous Image for {timing.aux_frame_index}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] - 2
                # st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index']].uuid
                st.experimental_rerun()
    with smallbutton1:
        # if it's not the first image
        if timing.aux_frame_index != 0:
            if st.button(f"{timing.aux_frame_index-1} ⏪", key=f"Previous Image for {timing.aux_frame_index}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] - 1
                # st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index']].uuid
                st.experimental_rerun()

    with smallbutton2:
        st.button(f"{timing.aux_frame_index} 📍", disabled=True)
    with smallbutton3:
        # if it's not the last image
        if timing.aux_frame_index != len(timing_details)-1:
            if st.button(f"{timing.aux_frame_index+1} ⏩", key=f"Next Image for {timing.aux_frame_index}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 1
                # st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index']].uuid
                st.experimental_rerun()
    with smallbutton4:
        if timing.aux_frame_index < len(timing_details)-2:
            if st.button(f"{timing.aux_frame_index+2} ⏭️", key=f"Next Next Image for {timing.aux_frame_index}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 2
                # st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index']].uuid
                st.experimental_rerun()


def styling_element(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    stages = ["Extracted Key Frames", "Current Main Variants"]

    if project_settings.default_stage != "":
        if 'index_of_which_stage_to_run_on' not in st.session_state:
            st.session_state['which_stage_to_run_on'] = project_settings.default_stage
            st.session_state['index_of_which_stage_to_run_on'] = stages.index(
                st.session_state['which_stage_to_run_on'])
    else:
        st.session_state['index_of_which_stage_to_run_on'] = 0

    stages1, stages2 = st.columns([1, 1])
    with stages1:
        st.session_state['which_stage_to_run_on'] = st.radio("What stage of images would you like to run styling on?", options=stages, horizontal=True,
                                                             index=st.session_state['index_of_which_stage_to_run_on'], help="Extracted frames means the original frames from the video.")
    with stages2:
        stage_frame: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
            st.session_state['current_frame_uuid'])
        if st.session_state['which_stage_to_run_on'] == "Extracted Key Frames":
            image = stage_frame.source_image.location
        else:
            image = data_repo.get_primary_variant_location(stage_frame.uuid)
            image = image.location if image else ""
        if image != "":
            st.image(image, use_column_width=True,
                     caption=f"Image {st.session_state['current_frame_uuid']}")
        else:
            st.error(
                f"No {st.session_state['which_stage_to_run_on']} image found for this variant")

    if stages.index(st.session_state['which_stage_to_run_on']) != st.session_state['index_of_which_stage_to_run_on']:
        st.session_state['index_of_which_stage_to_run_on'] = stages.index(
            st.session_state['which_stage_to_run_on'])
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

        model_list = data_repo.get_all_ai_model_list()
        model_name_list = [m.name for m in model_list]

        if 'index_of_default_model' not in st.session_state:
            if project_settings.default_model:
                st.session_state['model'] = project_settings.default_model.uuid
                st.session_state['index_of_default_model'] = next((i for i, obj in enumerate(model_list) if getattr(obj, 'uuid') == project_settings.default_model.uuid), None)
                st.write(f"Index of last model: {st.session_state['index_of_default_model']}")
            else:
                st.session_state['index_of_default_model'] = 0

        selected_model_name = st.selectbox(f"Which model would you like to use?", model_name_list, index=st.session_state['index_of_default_model'])
        st.session_state['model'] = next((obj.uuid for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_model_name), None)

        selected_model_index = next((i for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_model_name), None)
        if st.session_state['index_of_default_model'] != selected_model_index:
            st.session_state['index_of_default_model'] = selected_model_index
            st.experimental_rerun()

    current_model_name = data_repo.get_ai_model_from_uuid(st.session_state['model']).name

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
        if 'index_of_lora_model_1' not in st.session_state:
            st.session_state['index_of_lora_model_1'] = 0
            st.session_state['index_of_lora_model_2'] = 0
            st.session_state['index_of_lora_model_3'] = 0
        
        # df = pd.read_csv('models.csv')
        # filtered_df = df[df.iloc[:, 5] == 'LoRA']
        # lora_model_list = filtered_df.iloc[:, 0].tolist()
        lora_model_list = data_repo.get_all_ai_model_list(model_type=AIModelType.LORA.value)
        null_model = InternalAIModelObject(None, "", None, None, None, None, None, None, None)
        lora_model_list.insert(0, null_model)
        lora_model_name_list = [m.name for m in lora_model_list]

        selected_lora_1_name = st.selectbox(
            f"LoRA Model 1", lora_model_name_list, index=st.session_state['index_of_lora_model_1'])
        st.session_state['lora_model_1'] = next((obj.uuid for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_lora_1_name), "")
        selected_lora_1_index = next((i for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_lora_1_name), "")
        
        if st.session_state['index_of_lora_model_1'] != selected_lora_1_index:
            st.session_state['index_of_lora_model_1'] = selected_lora_1_index
            st.experimental_rerun()

        selected_lora_2_name = st.selectbox(
            f"LoRA Model 1", lora_model_name_list, index=st.session_state['index_of_lora_model_1'])
        st.session_state['lora_model_1'] = next((obj.uuid for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_lora_2_name), "")
        selected_lora_2_index = next((i for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_lora_2_name), "")

        if st.session_state['index_of_lora_model_2'] != selected_lora_2_index:
            st.session_state['index_of_lora_model_2'] = selected_lora_2_index
            st.experimental_rerun()

        selected_lora_3_name = st.selectbox(
            f"LoRA Model 1", lora_model_name_list, index=st.session_state['index_of_lora_model_1'])
        st.session_state['lora_model_1'] = next((obj.uuid for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_lora_3_name), "")
        selected_lora_3_index = next((i for i, obj in enumerate(model_list) if getattr(obj, 'name') == selected_lora_3_name), "")

        if st.session_state['index_of_lora_model_3'] != selected_lora_3_index:
            st.session_state['index_of_lora_model_3'] = selected_lora_3_index
            st.experimental_rerun()

        st.session_state['custom_models'] = [st.session_state['lora_model_1'],
                                             st.session_state['lora_model_2'], st.session_state['lora_model_3']]
        
        st.info("You can reference each model in your prompt using the following keywords: <1>, <2>, <3> - for example '<1> in the style of <2>.")

        lora_adapter_types = ['sketch', 'seg', 'keypose', 'depth', None]
        if "index_of_lora_adapter_type" not in st.session_state:
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

        dreambooth_model_list = data_repo.get_all_ai_model_list(model_type=AIModelType.DREAMBOOTH.value)
        dreambooth_model_name_list = [m.name for m in dreambooth_model_list]
        
        if 'index_of_dreambooth_model' not in st.session_state:
            st.session_state['index_of_dreambooth_model'] = 0

        selected_dreambooth_model_name = st.selectbox(
            f"Dreambooth Model", dreambooth_model_name_list, index=st.session_state['index_of_dreambooth_model'])
        st.session_state['custom_models'] = next((obj.uuid for i, obj in enumerate(dreambooth_model_list) if getattr(obj, 'name') == selected_dreambooth_model_name), "")
        selected_dreambooth_model_index = next((i for i, obj in enumerate(dreambooth_model_list) if getattr(obj, 'name') == selected_dreambooth_model_name), "")
        if st.session_state['index_of_dreambooth_model'] != selected_dreambooth_model_index:
            st.session_state['index_of_dreambooth_model'] = selected_dreambooth_model_index
    else:
        st.session_state['custom_models'] = []
        st.session_state['adapter_type'] = "N"

    if st.session_state['adapter_type'] == "canny":

        canny1, canny2 = st.columns(2)

        if project_settings.last_low_threshold:
            low_threshold_value = project_settings.last_low_threshold
        else:
            low_threshold_value = 50

        if project_settings.last_high_threshold:
            high_threshold_value = project_settings.last_high_threshold
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

    if current_model_name == AIModelType.STYLEGAN_NADA.value:
        st.warning("StyleGAN-NADA is a custom model that uses StyleGAN to generate a consistent character and style transformation. It only works for square images.")
        st.session_state['prompt'] = st.selectbox("What style would you like to apply to the character?", ['base', 'mona_lisa', 'modigliani', 'cubism', 'elf', 'sketch_hq', 'thomas', 'thanos', 'simpson', 'witcher',
                                                  'edvard_munch', 'ukiyoe', 'botero', 'shrek', 'joker', 'pixar', 'zombie', 'werewolf', 'groot', 'ssj', 'rick_morty_cartoon', 'anime', 'white_walker', 'zuckerberg', 'disney_princess', 'all', 'list'])
        st.session_state['strength'] = 0.5
        st.session_state['guidance_scale'] = 7.5
        st.session_state['seed'] = int(0)
        st.session_state['num_inference_steps'] = int(50)
    else:
        st.session_state['prompt'] = st.text_area(
            f"Prompt", label_visibility="visible", value=st.session_state['prompt_value'], height=150)
        if st.session_state['prompt'] != st.session_state['prompt_value']:
            st.session_state['prompt_value'] = st.session_state['prompt']
            st.experimental_rerun()
        with st.expander("💡 Learn about dynamic prompting"):
            st.markdown("## Why and how to use dynamic prompting")
            st.markdown("Why:")
            st.markdown("Dynamic prompting allows you to automatically vary the prompt throughout the clip based on changing features in the source image. This makes the output match the input more closely and makes character transformations look more natural.")
            st.markdown("How:")
            st.markdown(
                "You can include the following tags in the prompt to vary the prompt dynamically: [expression], [location], [mouth], and [looking]")
        
        if current_model_name == AIModelType.DREAMBOOTH.value:
            # TODO: fix custom model loading
            model_details = get_model_details_from_csv(st.session_state['custom_models'])
            st.info(
                f"Must include '{model_details['keyword']}' to run this model")
            if model_details['controller_type'] != "":
                st.session_state['adapter_type'] = st.selectbox(
                    f"Would you like to use the {model_details['controller_type']} controller?", ['Yes', 'No'])
            else:
                st.session_state['adapter_type'] = "No"

        else:
            if current_model_name == AIModelType.PIX_2_PIX.value:
                st.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")
        
        st.session_state['strength'] = st.slider(f"Strength", value=float(
            st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)

        with st.expander("Advanced settings 😏"):
            st.session_state['negative_prompt'] = st.text_area(
                f"Negative prompt", value=st.session_state['negative_prompt_value'], label_visibility="visible")
            if st.session_state['negative_prompt'] != st.session_state['negative_prompt_value']:
                st.session_state['negative_prompt_value'] = st.session_state['negative_prompt']
                st.experimental_rerun()
            st.session_state['guidance_scale'] = st.number_input(
                f"Guidance scale", value=float(st.session_state['guidance_scale']))
            st.session_state['seed'] = st.number_input(
                f"Seed", value=int(st.session_state['seed']))
            st.session_state['num_inference_steps'] = st.number_input(
                f"Inference steps", value=int(st.session_state['num_inference_steps']))

    if len(timing_details) > 1:
        st.markdown("***")
        st.markdown("## Batch queries")
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
                                                  'num_inference_steps'], st.session_state['which_stage_to_run_on'], st.session_state["promote_new_generation"], st.session_state['custom_models'], st.session_state['adapter_type'], st.session_state["use_new_settings"], st.session_state['low_threshold'], st.session_state['high_threshold'])
                st.experimental_rerun()


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
    project: InternalProjectObject = data_repo.get_project_from_uuid(project_uuid)
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

    frame_time = calculate_time_at_frame_number(input_video, float(extract_frame_number))

    data_repo.update_specific_timing(timing_uuid, frame_number=extract_frame_number)
    data_repo.update_specific_timing(timing_uuid, frame_time=frame_time)

    file_name = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=16)) + ".png"
    file_location = "videos/" + timing.project.uuid + "/assets/frames/1_selected/" + str(file_name)
    cv2.imwrite(file_location, frame)
    # img = Image.open("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    # img.save("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    
    final_image = data_repo.create_file(name=file_name, type=InternalFileType.IMAGE.value, \
                                        local_path=file_location)
    data_repo.update_specific_timing(timing_uuid, source_image_id=final_image.uuid)

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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(timing_uuid)

    current_primary_image = timing.primary_image
    current_alternative_images = timing.alternative_images
    current_source_image = timing.source_image

    if direction == "Up":
        prev_timing = data_repo.get_prev_timing(timing.uuid)
        previous_primary_image = prev_timing['primary_image']
        previous_alternative_images = prev_timing['alternative_images']
        previous_source_image = prev_timing['source_image']

        data_repo.update_specific_timing(prev_timing.uuid, primary_image_id=current_primary_image.uuid)
        print("current_alternative_images= ", current_alternative_images)
        data_repo.update_specific_timing(prev_timing.uuid, alternative_images=str(current_alternative_images))
        data_repo.update_specific_timing(prev_timing.uuid, source_image_id=current_source_image.uuid)
        data_repo.update_specific_timing(prev_timing.uuid, interpolated_video=None)
        data_repo.update_specific_timing(prev_timing.uuid, timed_clip_id=None)

        data_repo.update_specific_timing(timing.uuid, primary_image_id=previous_primary_image.uuid)
        data_repo.update_specific_timing(timing.uuid, alternative_images=str(previous_alternative_images))
        data_repo.update_specific_timing(timing.uuid, source_image_id=previous_source_image.uuid)

    elif direction == "Down":
        next_primary_image = data_repo.get_next_timing(timing.uuid).primary_image
        next_alternative_images = data_repo.get_next_timing(timing.uuid).alternative_images
        next_source_image = data_repo.get_next_timing(timing.uuid).source_image

        data_repo.update_specific_timing(
            data_repo.get_next_timing(timing.uuid).uuid, primary_image_id=current_primary_image.uuid)
        data_repo.update_specific_timing(
            data_repo.get_next_timing(timing.uuid).uuid, alternative_images=str(current_alternative_images))
        data_repo.update_specific_timing(
            data_repo.get_next_timing(timing.uuid).uuid, source_image_id=current_source_image.uuid)
        data_repo.update_specific_timing(
            data_repo.get_next_timing(timing.uuid).uuid, interpolated_video=None)
        data_repo.update_specific_timing(
            data_repo.get_next_timing(timing.uuid).uuid, timed_clip_id=None)

        data_repo.update_specific_timing(
            timing.uuid, primary_image_id=next_primary_image.uuid)
        data_repo.update_specific_timing(
            timing.uuid, alternative_images=str(next_alternative_images))
        data_repo.update_specific_timing(
            timing.uuid, source_image_id=next_source_image.uuid)

    data_repo.update_specific_timing(timing.uuid, interpolated_video=None)
    data_repo.update_specific_timing(timing.uuid, timed_clip_id=None)


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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    index_of_current_item = timing.aux_frame_index
    next_timing = data_repo.get_next_timing(timing_uuid)
    
    if next_timing:
        data_repo.update_specific_timing(next_timing.uuid, interpolated_video_id=None)
        if index_of_current_item < len(timing_details) - 1:
            data_repo.update_specific_timing(next_timing.uuid, interpolated_video_id=None)

        data_repo.update_specific_timing(next_timing.uuid, timed_clip_id=None)
        if index_of_current_item < len(timing_details) - 1:
            data_repo.update_specific_timing(next_timing.uuid, timed_clip_id=None)

    data_repo.delete_timing_from_uuid(timing.uuid)


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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

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
        which_stage_to_run_on,
        promote_new_generation, 
        custom_models, 
        adapter_type,
        update_inference_settings, 
        low_threshold, 
        high_threshold
    ):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    project_setting: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

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
            default_which_stage_to_run_on=which_stage_to_run_on,
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

    if which_stage_to_run_on == "Extracted Key Frames":
        source_image = timing.source_image
    else:
        variants: List[InternalFileObject] = data_repo.get_alternative_image_list(timing_uuid)
        number_of_variants = len(variants)
        primary_image = timing.primary_image
        source_image = variants[primary_image].location

    if st.session_state['custom_pipeline'] == "Mystique":
        output_file = custom_pipeline_mystique(timing_uuid, source_image)
    else:
        output_file = restyle_images(timing_uuid, source_image)

    if output_file != None:
        add_image_variant(output_file.uuid, timing_uuid)

        if promote_new_generation == True:
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
    data_repo.update_specific_timing(timing_uuid, primary_image_id=variant_to_promote.uuid)
    
    prev_timing = data_repo.get_prev_timing(timing_uuid)
    if prev_timing:
        data_repo.update_specific_timing(prev_timing.uuid, interpolated_clip_id=None)
        data_repo.update_specific_timing(timing_uuid, interpolated_clip_id=None)

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    frame_idx = timing.aux_frame_index

    # DOUBT: setting last interpolated_video to empty?
    if frame_idx < len(timing_details):
        data_repo.update_specific_timing(timing.uuid, interpolated_clip_id=None)

    if frame_idx > 1:
        data_repo.update_specific_timing(
            data_repo.get_prev_timing(timing_uuid).uuid, timed_clip_id=None)

    data_repo.update_specific_timing(timing_uuid, timed_clip_id=None)

    if frame_idx < len(timing_details):
        data_repo.update_specific_timing(timing.uuid, timed_clip_id=None)


def extract_canny_lines(image_path_or_url, project_name, low_threshold=50, high_threshold=150) -> InternalFileObject:
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
    file_location = f"videos/{project_name}/assets/resources/masks/{unique_file_name}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    new_canny_image.save(file_location)

    canny_image_file = data_repo.create_file(name=unique_file_name, type=InternalFileType.IMAGE.value, local_path=file_location)
    return canny_image_file

# the input image is an image created by the PIL library
def create_or_update_mask(timing_uuid, image) -> InternalFileObject:
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    unique_file_name = str(uuid.uuid4()) + ".png"
    file_location = f"videos/{timing.project.uuid}/assets/resources/masks/{unique_file_name}"
    image.save(file_location, "PNG")
    # if mask is not present than creating a new one
    if not (timing.mask and timing.mask.location):
        file_data = {
            "name": unique_file_name,
            "type": InternalFileType.IMAGE.value,
            "local_path": file_location
        }
        mask_file: InternalFileObject = data_repo.create_file(**file_data)

        data_repo.update_specific_timing(timing_uuid, mask_id=mask_file.uuid)
    else:
        # if it is already present then just updating the file location
        data_repo.update_file(timing.mask.uuid, local_path=file_location)

    timing = data_repo.get_timing_from_uuid(timing_uuid)
    return timing.mask


def create_working_assets(video_name):
    os.mkdir("videos/" + video_name)
    os.mkdir("videos/" + video_name + "/assets")

    os.mkdir("videos/" + video_name + "/assets/frames")

    os.mkdir("videos/" + video_name + "/assets/frames/0_extracted")
    os.mkdir("videos/" + video_name + "/assets/frames/1_selected")
    os.mkdir("videos/" + video_name +
             "/assets/frames/2_character_pipeline_completed")
    os.mkdir("videos/" + video_name +
             "/assets/frames/3_backdrop_pipeline_completed")

    os.mkdir("videos/" + video_name + "/assets/resources")

    os.mkdir("videos/" + video_name + "/assets/resources/backgrounds")
    os.mkdir("videos/" + video_name + "/assets/resources/masks")
    os.mkdir("videos/" + video_name + "/assets/resources/audio")
    os.mkdir("videos/" + video_name + "/assets/resources/input_videos")
    os.mkdir("videos/" + video_name + "/assets/resources/prompt_images")

    os.mkdir("videos/" + video_name + "/assets/videos")

    os.mkdir("videos/" + video_name + "/assets/videos/0_raw")
    os.mkdir("videos/" + video_name + "/assets/videos/1_final")
    os.mkdir("videos/" + video_name + "/assets/videos/2_completed")

    data = {'key': ['last_prompt', 'default_model', 'last_strength', 'last_custom_pipeline', 'audio', 'input_type', 'input_video', 'extraction_type', 'width', 'height', 'last_negative_prompt', 'last_guidance_scale', 'last_seed', 'last_num_inference_steps', 'last_which_stage_to_run_on', 'last_custom_models', 'last_adapter_type', 'guidance_type', 'default_animation_style', 'last_low_threshold', 'last_high_threshold', 'last_stage_run_on'],
            'value': ['prompt', 'controlnet', '0.5', 'None', '', 'video', '', 'Extract manually', '', '', '', 7.5, 0, 50, 'Extracted Frames', '', '', '', '', 100, 200, '']}

    df = pd.DataFrame(data)

    df.to_csv(f'videos/{video_name}/settings.csv', index=False)

    df = pd.DataFrame(columns=['frame_time', 'frame_number', 'primary_image', 'alternative_images', 'custom_pipeline', 'negative_prompt', 'guidance_scale', 'seed', 'num_inference_steps',
                      'model_id', 'strength', 'notes', 'source_image', 'custom_models', 'adapter_type', 'clip_duration', 'interpolated_video', 'timed_clip', 'prompt', 'mask', 'canny_image', 'preview_video', 'animation_style', 'interpolation_steps', 'low_threshold', 'high_threshold'])

    # df.loc[0] = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    df.to_csv(f'videos/{video_name}/timings.csv', index=False)


def inpainting(input_image, prompt, negative_prompt, timing_uuid, invert_mask, pass_mask=False) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    if pass_mask == False:
        mask = timing.mask.location
    else:
        mask = "temp/mask.png"

    if not mask.startswith("http"):
        mask = open(mask, "rb")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.andreas_sd_inpainting, mask=mask, image=input_image, prompt=prompt,
                                            invert_mask=invert_mask, negative_prompt=negative_prompt, num_inference_steps=25)

    file_name = str(uuid.uuid4()) + ".png"
    image_file = data_repo.create_file(name=file_name, type=InternalFileType.IMAGE.value, hosted_url=output[0])

    return image_file

# adds the image file in variant (alternative images) list 
def add_image_variant(image_file_uuid: str, timing_uuid: str):
    data_repo = DataRepo()
    image_file: InternalFileObject = data_repo.get_file_from_uuid(image_file_uuid)
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    # if not (timing.alternative_images_list and len(timing.alternative_images_list)):
    #     alternative_images = [image_file.uuid]
    #     additions = []
    # else:
    #     alternative_images = []

    #     additions = timing.alternative_images
    #     for addition in additions:
    #         alternative_images.append(addition)

    #     alternative_images.append(image_file.uuid)
    #     alternative_images = alternative_images

    alternative_image_list = timing.alternative_images_list + [image_file]
    alternative_image_uuid_list = [img.uuid for img in alternative_image_list]
    primary_image_uuid = alternative_image_uuid_list[0]
    alternative_image_uuid_list = json.dumps(alternative_image_uuid_list)

    data_repo.update_specific_timing(
        timing_uuid, alternative_images=alternative_image_uuid_list)

    if not timing.primary_image:
        data_repo.update_specific_timing(timing_uuid, primary_image_id=primary_image_uuid)

    return len(alternative_image_list)

# TODO: complete this function
def convert_image_list_to_file_list(images_list: str):
    data_repo = DataRepo()
    file_list = []
    for image in images_list:
        data = {
            "name": str(uuid.uuid4()),
            "type": InternalFileType.IMAGE.value
        }
        if image.startswith("http"):
            data['hosted_url'] = image
        else:
            data['local_url'] = image
        
        image_file = data_repo.create_file(**data)
        file_list.append(image_file)
    return file_list

# DOUBT: why do we need controller in dreambooth training?
# INFO: images_list passed here are converted to internal files after they are used for training
def train_dreambooth_model(instance_prompt, class_prompt, training_file_url, max_train_steps, model_name, images_list: List[str]):
    ml_client = get_ml_client()
    response = ml_client.dreambooth_training(
        training_file_url, instance_prompt, class_prompt, max_train_steps, model_name)
    training_status = response["status"]

    model_id = response["id"]
    if training_status == "queued":
        file_list = convert_image_list_to_file_list(images_list)
        file_uuid_list = [file.uuid for file in file_list]
        file_uuid_list = json.dump(file_uuid_list)

        model_data = {
            "name": model_name,
            "user_id": local_storage.get_current_user_uuid(),
            "replicate_model_id": model_id,
            "replicate_url": None,
            "diffusers_url": None,
            "category": AIModelType.DREAMBOOTH.value,
            "training_image_list": file_uuid_list,
            "keyword": instance_prompt,
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
    file_uuid_list = json.dump(file_uuid_list)
    model_data = {
        "name": model_name,
        "user_id": local_storage.get_current_user_uuid(),
        "replicate_url": output,
        "diffusers_url": None,
        "category": AIModelType.LORA.value,
        "training_image_list": file_uuid_list
    }

    data_repo.create_ai_model(**model_data)
    return f"Successfully trained - the model '{model_name}' is now available for use!"

# TODO: making an exception for this, passing just the image urls instead of 
# image files
def train_model(images_list, instance_prompt, class_prompt, max_train_steps,
                model_name, type_of_model, type_of_task, resolution):
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
                                      max_train_steps, model_name, images_list)
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


def replace_background(project_uuid, foreground_image, background_image) -> InternalFileObject:
    data_repo = DataRepo()

    if background_image.startswith("http"):
        response = r.get(background_image)
        background_image = Image.open(BytesIO(response.content))
    else:
        background_image = Image.open(f"{background_image}")
    foreground_image = Image.open(f"masked_image.png")
    background_image.paste(foreground_image, (0, 0), foreground_image)
    background_image.save(f"videos/{project_uuid}/replaced_bg.png")

    filename = str(uuid.uuid4()) + ".png"
    image_file = data_repo.create_file(name=filename, local_path=f"videos/{project_uuid}/replaced_bg.png", \
                                     type=InternalFileType.IMAGE.value)

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


def resize_image(video_name, new_width, new_height, image_file: InternalFileObject) -> InternalFileObject:
    image = image_file.location
    response = r.get(image)
    image = Image.open(BytesIO(response.content))
    resized_image = image.resize((new_width, new_height))

    time.sleep(0.1)

    unique_id = str(uuid.uuid4())
    filepath = "videos/" + str(video_name) + "/temp_image-" + unique_id + ".png"
    resized_image.save(filepath)

    data_repo = DataRepo()
    image_file = data_repo.create_file(name=unique_id + ".png", type=InternalFileType.IMAGE.value,
                                     local_path=filepath)

    # not uploading or removing the created image as of now
    # resized_image = upload_image(
    #     "videos/" + str(video_name) + "/temp_image.png")

    # os.remove("videos/" + str(video_name) + "/temp_image.png")

    return image_file


def face_swap(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    model_category = timing['model']['category']

    # TODO: check this logic by running the code
    if model_category == "Dreambooth":
        custom_model_uuid = timing.custom_model_id_list[0]
        # custom_model = timing_details[index_of_current_item]["custom_models"]
    if model_category == "LoRA":
        # custom_model = ast.literal_eval(
        #     timing_details[index_of_current_item]["custom_models"][1:-1])[0]
        custom_model_uuid = timing.custom_model_id_list[0]

    model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(custom_model_uuid)
    
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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.stylegan_nada, input=input_image,
                                            output_style=timing.prompt)
    output = resize_image(timing.project.name, 512, 512, output)

    return output


def prompt_model_stable_diffusion_xl(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    model_name = timing['model']['name']
    image_number = timing.aux_frame_index
    prompt = timing.prompt
    strength = timing.strength
    negative_prompt = timing.negative_prompt
    guidance_scale = timing.guidance_scale
    seed = timing.seed
    num_inference_steps = timing.num_inteference_steps

    model_details: InternalAIModelObject = data_repo.get_ai_model_from_uuid(timing['model'].uuid)
    model_id = model_details.replicate_model_id

    ml_client = get_ml_client()

    source_image = source_image_file.location
    if timing_details[image_number]["adapter_type"] == "Yes":
        if source_image.startswith("http"):
            control_image = source_image
        else:
            control_image = open(source_image, "rb")
    else:
        control_image = None

    # version of models that were custom created has to be fetched
    if model_details["version"] == "":
        version = ml_client.get_model_version_from_id(model_id)
        data_repo.update_ai_model(model_details.uuid, version=version)
    else:
        version = model_details["version"]

    # TODO: change to custom user
    model_version = ml_client.get_model_by_name(
        f"peter942/{model_name}", version)

    if source_image.startswith("http"):
        input_image = source_image
    else:
        input_image = open(source_image, "rb")

    input_data = {
        "image": input_image,
        "prompt": prompt,
        "prompt_strength": float(strength),
        "height": int(project_settings["height"]),
        "width": int(project_settings["width"]),
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
        image_file = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value, hosted_url=i, tag=InternalFileTag.GENERATED_VIDEO.value)
        return image_file
    
    return None


def update_slice_of_video_speed(video_name, input_video, desired_speed_change):

    clip = VideoFileClip("videos/" + str(video_name) +
                         "/assets/videos/0_raw/" + str(input_video))

    clip_location = "videos/" + \
        str(video_name) + "/assets/videos/0_raw/" + str(input_video)

    desired_speed_change_text = str(desired_speed_change) + "*PTS"

    video_stream = ffmpeg.input(str(clip_location))

    video_stream = video_stream.filter('setpts', desired_speed_change_text)

    ffmpeg.output(video_stream, "videos/" + str(video_name) +
                  "/assets/videos/0_raw/output_" + str(input_video)).run()

    video_capture = cv2.VideoCapture(
        "videos/" + str(video_name) + "/assets/videos/0_raw/output_" + str(input_video))

    os.remove("videos/" + str(video_name) +
              "/assets/videos/0_raw/" + str(input_video))
    os.rename("videos/" + str(video_name) + "/assets/videos/0_raw/output_" + str(input_video),
              "videos/" + str(video_name) + "/assets/videos/0_raw/" + str(input_video))


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

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
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

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    desired_duration = timing.clip_duration
    animation_style = timing.animation_style
    location_of_video = video_file.local_path

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

        if save_to_new_location:
            file_name = ''.join(random.choices(
                string.ascii_lowercase + string.digits, k=16)) + ".mp4"
            location_of_video = "videos/" + \
                str(timing.project.uuid) + \
                "/assets/videos/1_final/" + str(file_name)
        else:
            os.remove(location_of_video)
        
        # TODO: add this in save override of the model
        updated_clip.write_videofile(location_of_video, codec='libx265')
        if save_to_new_location:
            file_name = str(uuid.uuid4()) + ".mp4"
            file_data = {
                "name" : file_name, 
                "type" : InternalFileType.VIDEO.value, 
                "tag" : InternalFileTag.GENERATED_VIDEO.value, 
                "local_path" : location_of_video
            }
            video_file: InternalFileObject = data_repo.create_file(**file_data)
        else:
            data_repo.create_or_update_file(video_file.uuid, type=InternalFileType.VIDEO.value, local_path=location_of_video)

        clip.close()
        updated_clip.close()

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
        output_clip.write_videofile(
            new_file_location, codec="libx264", preset="fast")

        if save_to_new_location:
            video_file: InternalFileObject = data_repo.create_file(name=new_file_name, type=InternalFileType.VIDEO.value, local_path=new_file_location)
        else:
            os.remove(location_of_video)
            location_of_video = new_file_location
            data_repo.create_or_update_file(video_file.uuid, type=InternalFileType.VIDEO.value, local_path=location_of_video)

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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
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
        timing_item: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(project_uuid, index_of_current_item)

        # last frame
        if index_of_current_item == (length_of_list - 1):
            time_of_frame = timing_item.frame_time
            duration_of_static_time = 0.0
            end_duration_of_frame = float(time_of_frame) + float(duration_of_static_time)
            total_duration_of_frame = float(end_duration_of_frame) - float(time_of_frame)
        else:
            time_of_frame = timing_item.frame_time
            next_timing = data_repo.get_next_timing(timing_item.uuid)
            time_of_next_frame = next_timing.frame_time
            total_duration_of_frame = float(time_of_next_frame) - float(time_of_frame)

        duration_of_static_time = 0.0
        duration_of_morph = float(total_duration_of_frame) - float(duration_of_static_time)

        data_repo.update_specific_timing(timing_item.uuid, clip_duration=total_duration_of_frame)


def hair_swap(source_image, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    app_secret = data_repo.get_app_secrets_from_user_uuid(timing.project.user_uuid)

    # DOUBT: what's the video name here?
    video_name = ""
    source_hair = upload_file("videos/" + str(video_name) + "/face.png", app_secret['aws_access_key'],\
                              app_secret['aws_secret_key'])

    target_hair = upload_file("videos/" + str(video_name) +
                               "/assets/frames/2_character_pipeline_completed/" + str(timing.aux_frame_index) + ".png",\
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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    model_name = timing.model.name
    strength = timing.strength

    if model_name == "stable-diffusion-img2img-v2.1":
        output_file = prompt_model_stability(timing_uuid, source_image)
    elif model_name == "depth2img":
        output_file = prompt_model_depth2img(strength, timing_uuid, source_image)
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
        output_file = prompt_model_stable_diffusion_xl(timing_uuid, source_image)
    elif model_name == "real-esrgan-upscaling":
        output_file = prompt_model_real_esrgan_upscaling(source_image)

    return output_file


def custom_pipeline_mystique(timing_uuid, source_image) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    output_image_file: InternalFileObject = face_swap(timing_uuid, source_image)
    output_image_file = touch_up_image(output_image_file)
    output_image_file = resize_image(timing.project.name, project_settings.width, project_settings.height, output_image_file)

    if timing.model.category == AIModelType.DREAMBOOTH.value:
        output_image_file = prompt_model_dreambooth(timing_uuid, output_image_file)
    elif timing.model.category == AIModelType.LORA.value:
        output_image_file = prompt_model_lora(timing_uuid, output_image_file)

    return output_image_file


def create_timings_row_at_frame_number(project_uuid, index_of_frame):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_timing_list_from_project(project_uuid)
    
    # remove the interpolated video from the current row and the row before and after - unless it is the first or last row
    timing_data = {
        "project_id": project_uuid,
        "frame_time": 0.0,
        "animation_style": AnimationStyleType.INTERPOLATION.value,
        "aux_frame_index": index_of_frame
    }
    timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)
    
    prev_timing: InternalFrameTimingObject = data_repo.get_prev_timing(timing.uuid)
    if prev_timing:
        data_repo.update_specific_timing(prev_timing.uuid, interpolated_clip_id=None)

    next_timing: InternalAIModelObject = data_repo.get_next_timing(timing.uuid)
    if next_timing:
        data_repo.update_specific_timing(next_timing.uuid, interpolated_clip_id=None)

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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

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

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(project_uuid)

    calculate_desired_duration_of_each_clip(project_uuid)

    total_number_of_videos = len(timing_details) - 1

    for i in range(0, total_number_of_videos):
        index_of_current_item = i
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(project_uuid, i)

        timing = timing_details[i]
        if quality == VideoQuality.HIGH.value:
            data_repo.update_specific_timing(current_timing.uuid, timed_clip="")
            interpolation_steps = calculate_dynamic_interpolations_steps(
                timing_details[index_of_current_item].clip_duration)
            if not timing.interpolation_steps or timing.interpolation_steps < interpolation_steps:
                data_repo.update_specific_timing(current_timing.uuid, interpolation_steps=interpolation_steps, interpolated_clip="")
        else:
            if not timing.interpolation_steps or timing.interpolation_steps < 3:
                data_repo.update_specific_timing(current_timing.uuid, interpolation_steps=3)

        if not timing.interpolated_clip:
            # DOUBT: if and else condition had the same code
            # if total_number_of_videos == index_of_current_item:
            #     video_location = create_individual_clip(
            #         index_of_current_item, project_name)
            #     update_specific_timing_value(
            #         project_name, index_of_current_item, "interpolated_video", video_location)
            # else:
            video_location = create_individual_clip(current_timing.uuid)
            data_repo.update_specific_timing(current_timing.uuid, interpolated_clip_id=video_location.uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(timing.project.uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(timing.project.uuid)
    total_number_of_videos = len(timing_details) - 2

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        timing = timing_details[index_of_current_item]
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(timing.project.uuid, index_of_current_item)
        if index_of_current_item <= total_number_of_videos:
            if not current_timing.timed_clip:
                desired_duration = current_timing.clip_duration
                location_of_input_video_file = current_timing.interpolated_clip
                duration_of_input_video = float(get_duration_from_video(location_of_input_video_file))
                output_video = update_speed_of_video_clip(location_of_input_video_file, True, timing.uuid)
                
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
                
                data_repo.update_specific_timing(current_timing.uuid, timed_clip_id=output_video.uuid)

    video_list = []

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(timing.project.uuid)

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(project_uuid, index_of_current_item)
        if index_of_current_item <= total_number_of_videos:
            video_file = current_timing.timed_clip
            video_list.append(video_file.location)

    video_clip_list = [VideoFileClip(v) for v in video_list]
    finalclip = concatenate_videoclips(video_clip_list)

    output_video_file = f"videos/{timing.project.uuid}/assets/videos/2_completed/{final_video_name}.mp4"
    if project_settings.audio:
        audio_location = project_settings.audio.local_path
        audio_clip = AudioFileClip(audio_location)
        finalclip = finalclip.set_audio(audio_clip)
    
    # TODO: add this in the save override
    finalclip.write_videofile(
        output_video_file,
        fps=60,  # or 60 if your original video is 60fps
        audio_bitrate="128k",
        bitrate="5000k",
        codec="libx264",
        audio_codec="aac"
    )

    file_data = {
        "name": final_video_name,
        "type": InternalFileType.VIDEO.value,
        "local_path": output_video_file,
        "tag" : InternalFileTag.GENERATED_VIDEO.value,
        "project_id": project_uuid
    }

    data_repo.create_file(**file_data)


def create_gif_preview(project_uuid):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(project_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(project_uuid)
    list_of_images = []

    for i in timing_details:
        # make index_of_current_item the index of the current item
        index_of_current_item = timing_details.index(i)
        source_image = timing_details[index_of_current_item].primary_image.location
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
        urllib.request.urlretrieve(output, "depth.png")
    except Exception as e:
        print(e)

    depth_map = Image.open("depth.png")
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
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

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


def prompt_model_lora(timing_uuid, source_image_file: InternalFileObject) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(timing.project.uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(timing.project.uuid)

    lora_models = timing.custom_model_id_list
    default_model_url = DEFAULT_LORA_MODEL_URL

    lora_model_urls = []

    for lora_model_uuid in lora_models:
        if lora_model_uuid != "":
            lora_model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(lora_model_uuid)
            print(lora_model)
            if lora_model.replicate_url != "":
                lora_model_url = lora_model.replicate_url
            else:
                lora_model_url = default_model_url
        else:
            lora_model_url = default_model_url

        lora_model_urls.append(lora_model_url)

    lora_model_1_model_url = lora_model_urls[0]
    lora_model_2_model_url = lora_model_urls[1]
    lora_model_3_model_url = lora_model_urls[2]

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
        'num_inference_steps': timing.num_inference_steps,
        'guidance_scale': timing.guidance_scale,
        'prompt_strength': timing.strength,
        'scheduler': "DPMSolverMultistep",
        'lora_urls': lora_model_1_model_url + "|" + lora_model_2_model_url + "|" + lora_model_3_model_url,
        'lora_scales': "0.5 | 0.5 | 0.5",
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
            return output[0]
        except replicate.exceptions.ModelError as e:
            if "NSFW content detected" in str(e):
                print("NSFW content detected. Attempting to rerun code...")
                attempts += 1
                continue
            else:
                raise e
        except Exception as e:
            raise e

    filename = "default_3x_failed-656a7e5f-eca9-4f92-a06b-e1c6ff4a5f5e.png"     # const filename
    file = data_repo.get_file_from_name(filename)
    if file:
        return file
    else:
        file: InternalFileObject = data_repo.create_file(name=filename, type=InternalFileType.IMAGE.value,
                                            hosted_url="https://i.ibb.co/ZG0hxzj/Failed-3x-In-A-Row.png")
        return file


def attach_audio_element(project_uuid, expanded):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(uuid=project_uuid)
    project_setting: InternalSettingObject = data_repo.get_project_setting(project_id=project_uuid)

    with st.expander("Audio"):
        uploaded_file = st.file_uploader("Attach audio", type=[
                                         "mp3"], help="This will attach this audio when you render a video")
        if st.button("Upload and attach new audio"):
            if uploaded_file:
                local_file_location = os.path.join(f"videos/{project.uuid}/assets/resources/audio", uploaded_file.name)
                if not os.path.exists(f"videos/{project.uuid}/assets/resources/audio"):
                    os.makedirs(f"videos/{project.uuid}/assets/resources/audio")

                with open(local_file_location, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                    file_data = {
                        "name": str(uuid.uuid4()) + ".mp3",
                        "type": InternalFileType.AUDIO.value,
                        "local_path": local_file_location,
                        "project_id": project_uuid
                    }
                    audio_file: InternalFileObject = data_repo.create_file(**file_data)
                    data_repo.update_project_setting(project_uuid, audio_id=audio_file.uuid)
                    st.experimental_rerun()
            else:
                st.warning('No file selected')
        
        if project_setting.audio:
            # TODO: store "extracted_audio.mp3" in a constant
            if project_setting.audio.name == "extracted_audio.mp3":
                st.info("You have attached the audio from the video you uploaded.")
            
            if project_setting.audio.location:
                st.audio(project_setting.audio.location)


def execute_image_edit(type_of_mask_selection, type_of_mask_replacement, \
                       background_image, editing_image, prompt, negative_prompt, \
                        width, height, layer, timing_uuid) -> InternalFileObject:
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.project.uuid
    app_secret = data_repo.get_app_secrets_from_user_uuid(timing.project.user_uuid)
    index_of_current_item = timing.aux_frame_index

    if type_of_mask_selection == "Automated Background Selection":
        removed_background = remove_background(editing_image)
        response = r.get(removed_background)
        with open("masked_image.png", "wb") as f:
            f.write(response.content)
        if type_of_mask_replacement == "Replace With Image":
            replace_background(project_uuid, "masked_image.png", background_image)
            file_location = f"videos/{project_uuid}/replaced_bg.png"
            file_data = {
                "name": str(uuid.uuid4()) + ".png",
                "type": InternalFileType.IMAGE.value,
                "local_path": file_location,
            }
            edited_image: InternalFileObject = data_repo.create_file(**file_data)

        elif type_of_mask_replacement == "Inpainting":
            image = Image.open("masked_image.png")
            converted_image = Image.new("RGB", image.size, (255, 255, 255))
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    if pixel[3] == 0:
                        converted_image.putpixel((x, y), (0, 0, 0))
                    else:
                        converted_image.putpixel((x, y), (255, 255, 255))
            create_or_update_mask(timing_uuid, converted_image)
            edited_image = inpainting(editing_image, prompt, negative_prompt, timing.uuid, True)

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

            result_img = Image.new("RGBA", bg_img.size, (255, 255, 255, 0))
            for x in range(bg_img.size[0]):
                for y in range(bg_img.size[1]):
                    if mask_img.getpixel((x, y)) == (0, 0, 0, 255):
                        result_img.putpixel((x, y), (255, 255, 255, 0))
                    else:
                        result_img.putpixel((x, y), bg_img.getpixel((x, y)))
            result_img.save("masked_image.png")
            edited_image = replace_background(project_uuid, "masked_image.png", background_image)
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
            edited_image = inpainting(editing_image, prompt, negative_prompt, timing_uuid, True)
    elif type_of_mask_selection == "Automated Layer Selection":
        mask_location = create_depth_mask_image(editing_image, layer, timing.uuid)
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
            masked_img.save("masked_image.png")
            edited_image = replace_background(project_uuid, "masked_image.png", background_image)
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(editing_image, prompt, negative_prompt, timing_uuid, True)

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
            masked_img.save("masked_image.png")
            edited_image = replace_background(project_uuid, "masked_image.png", background_image)
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(editing_image, prompt, negative_prompt, timing_uuid, True)

    elif type_of_mask_selection == "Invert Previous Mask":
        mask_location = timing.mask.location
        if type_of_mask_replacement == "Replace With Image":
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
            masked_img.save("masked_image.png")
            edited_image = replace_background(project_uuid, "masked_image.png", background_image)
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(editing_image, prompt, negative_prompt, timing_uuid, False)

    return edited_image


def page_switcher(pages, page):
    section = [section["section_name"]
               for section in pages if page in section["pages"]][0]
    index_of_section = [section["section_name"]
                        for section in pages].index(section)
    index_of_page_in_section = pages[index_of_section]["pages"].index(page)

    return index_of_page_in_section, index_of_section