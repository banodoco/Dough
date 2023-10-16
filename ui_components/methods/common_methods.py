import io
from typing import List
import os
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from moviepy.editor import *
import cv2
import requests as r
import math
import json
import time
import uuid
from io import BytesIO
import numpy as np
import urllib3
from shared.constants import SERVER, InferenceType, InternalFileType, ServerType
from pydub import AudioSegment
from backend.models import InternalFileObject
from ui_components.constants import SECOND_MASK_FILE, SECOND_MASK_FILE_PATH, WorkflowStageType
from ui_components.methods.file_methods import add_temp_file_to_project, convert_bytes_to_file, generate_pil_image, generate_temp_file, save_or_host_file, save_or_host_file_bytes
from ui_components.methods.video_methods import calculate_desired_duration_of_individual_clip, update_speed_of_video_clip
from ui_components.models import InternalAIModelObject, InternalFrameTimingObject, InternalSettingObject
from utils.constants import ImageStage
from utils.data_repo.data_repo import DataRepo
from shared.constants import AnimationStyleType

from ui_components.models import InternalFileObject
from typing import Union
import streamlit as st

from utils.media_processor.video import VideoProcessor

def clone_styling_settings(source_frame_number, target_frame_uuid):
    data_repo = DataRepo()
    target_timing = data_repo.get_timing_from_uuid(target_frame_uuid)
    timing_details = data_repo.get_timing_list_from_project(
        target_timing.project.uuid)
    
    primary_image = data_repo.get_file_from_uuid(timing_details[source_frame_number].primary_image.uuid)
    params = primary_image.inference_params

    if params:
        target_timing.prompt = params.prompt
        target_timing.negative_prompt = params.negative_prompt
        target_timing.guidance_scale = params.guidance_scale
        target_timing.seed = params.seed
        target_timing.num_inference_steps = params.num_inference_steps
        target_timing.strength = params.strength
        target_timing.adapter_type = params.adapter_type
        target_timing.low_threshold = params.low_threshold
        target_timing.high_threshold = params.high_threshold
    
        if params.model_uuid:
            model = data_repo.get_ai_model_from_uuid(params.model_uuid)
            target_timing.model = model

# TODO: image format is assumed to be PNG, change this later
def save_new_image(img: Union[Image.Image, str, np.ndarray, io.BytesIO], project_uuid) -> InternalFileObject:
    img = generate_pil_image(img)
    
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

def save_and_promote_image(image, project_uuid, frame_uuid, save_type):
    data_repo = DataRepo()

    try:
        saved_image = save_new_image(image, project_uuid)

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

def fetch_image_by_stage(project_uuid, stage, frame_idx):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    if stage == WorkflowStageType.SOURCE.value:
        return timing_details[frame_idx].source_image
    elif stage == WorkflowStageType.STYLED.value:
        return timing_details[frame_idx].primary_image
    else:
        return None


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
    
def update_timings_in_order(project_uuid):
    data_repo = DataRepo()

    timing_list: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(project_uuid)

    # Iterate through the timing objects
    for i, timing in enumerate(timing_list):
        # Set the frame time to the index of the timing object only if it's different from the current one
        if timing.frame_time != float(i):
            print(f"Updating timing {timing.uuid} frame time to {float(i)}")
            data_repo.update_specific_timing(timing.uuid, frame_time=float(i))


def change_position_input(timing_uuid, src):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)

    min_value = 1
    max_value = len(timing_list)

    new_position = st.number_input("Move to new position:", min_value=min_value, max_value=max_value,
                                   value=timing.aux_frame_index + 1, step=1, key=f"new_position_{timing.uuid}_{src}")
    
    if st.button('Update Position',key=f"change_frame_position_{timing.uuid}_{src}"): 
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=new_position - 1)
        st.rerun()
        

def move_frame(direction, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if direction == "Up":
        if timing.aux_frame_index == 0:
            st.error("This is the first frame")       
            time.sleep(1)     
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index - 1)
    elif direction == "Down":
        timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)
        if timing.aux_frame_index == len(timing_list) - 1:
            st.error("This is the last frame")
            time.sleep(1)
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index + 1)

def move_frame_back_button(timing_uuid, orientation):
    direction = "Up"
    if orientation == "side-to-side":
        arrow = "⬅️"        
    else:  # up-down
        arrow = "⬆️"        
    if st.button(arrow, key=f"move_frame_back_{timing_uuid}", help="Move frame back"):
        move_frame(direction, timing_uuid)
        st.rerun()


def move_frame_forward_button(timing_uuid, orientation):
    direction = "Down"
    if orientation == "side-to-side":
        arrow = "➡️"        
    else:  # up-down
        arrow = "⬇️"

    if st.button(arrow, key=f"move_frame_forward_{timing_uuid}", help="Move frame forward"):
        move_frame(direction, timing_uuid)
        st.rerun()


def delete_frame_button(timing_uuid, show_label=False):
    if show_label:
        label = "Delete Frame 🗑️"
    else:
        label = "🗑️"

    if st.button(label, key=f"delete_frame_{timing_uuid}", help="Delete frame"):
        delete_frame(timing_uuid)
        st.rerun()

def delete_frame(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    next_timing = data_repo.get_next_timing(timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)

    if next_timing:
        data_repo.update_specific_timing(
            next_timing.uuid,
            interpolated_clip_list=None,
            preview_video_id=None,
            timed_clip_id=None
        )

    data_repo.delete_timing_from_uuid(timing.uuid)
    
    if timing.aux_frame_index == len(timing_details) - 1:
        st.session_state['current_frame_index'] = max(1, st.session_state['current_frame_index'] - 1)
        st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid

def save_uploaded_image(image, project_uuid, frame_uuid, save_type):
    data_repo = DataRepo()

    try:
        saved_image = save_new_image(image, project_uuid)

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
    
def jump_to_single_frame_view_button(display_number, timing_details):
    if st.button(f"Jump to #{display_number}"):
        st.session_state['prev_frame_index'] = display_number
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
        st.session_state['frame_styling_view_type'] = "Individual View"
        st.session_state['change_view_type'] = True
        st.rerun()

def replace_image_widget(timing_uuid, stage, options=["Other Frame", "Uploaded Frame"]):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(timing.project.uuid)

    replace_with = options[0] if len(options) == 1 else st.radio("Replace with:", options, horizontal=True, key=f"replace_with_what_{stage}_{timing_uuid}")

    if replace_with == "Other Frame":
        which_stage_to_use_for_replacement = st.radio("Select stage to use:", [
            ImageStage.MAIN_VARIANT.value, ImageStage.SOURCE_IMAGE.value], key=f"which_stage_to_use_for_replacement_{stage}_{timing_uuid}", horizontal=True)
        which_image_to_use_for_replacement = st.number_input("Select image to use:", min_value=0, max_value=len(
            timing_details)-1, value=0, key=f"which_image_to_use_for_replacement_{stage}")

        if which_stage_to_use_for_replacement == ImageStage.SOURCE_IMAGE.value:                                    
            selected_image = timing_details[which_image_to_use_for_replacement].source_image
        elif which_stage_to_use_for_replacement == ImageStage.MAIN_VARIANT.value:
            selected_image = timing_details[which_image_to_use_for_replacement].primary_image

        st.image(selected_image.local_path, use_column_width=True)

        if st.button("Replace with selected frame", disabled=False,key=f"replace_with_selected_frame_{stage}_{timing_uuid}"):
            if stage == "source":
                data_repo.update_specific_timing(timing.uuid, source_image_id=selected_image.uuid)                                        
                st.success("Replaced")
                time.sleep(1)
                st.rerun()
            else:
                number_of_image_variants = add_image_variant(
                    selected_image.uuid, timing.uuid)
                promote_image_variant(
                    timing.uuid, number_of_image_variants - 1)
                st.success("Replaced")
                time.sleep(1)
                st.rerun()

    elif replace_with == "Uploaded Frame":
        if stage == "source":
            uploaded_file = st.file_uploader("Upload Source Image", type=[
                "png", "jpeg"], accept_multiple_files=False)
            if uploaded_file != None:
                if st.button("Upload Source Image"):
                    if uploaded_file:
                        timing = data_repo.get_timing_from_uuid(timing.uuid)
                        if save_and_promote_image(uploaded_file, timing.project.uuid, timing.uuid, "source"):
                            time.sleep(1.5)
                            st.rerun()
        else:
            replacement_frame = st.file_uploader("Upload Styled Image", type=[
                "png", "jpeg"], accept_multiple_files=False, key=f"replacement_frame_upload_{stage}_{timing_uuid}")
            if replacement_frame != None:
                if st.button("Replace frame", disabled=False):                    
                    timing = data_repo.get_timing_from_uuid(timing.uuid)
                    if replacement_frame:
                        save_and_promote_image(replacement_frame, timing.project.uuid, timing.uuid, "styled")
                        st.success("Replaced")
                        time.sleep(1)
                        st.rerun()

def promote_image_variant(timing_uuid, variant_to_promote_frame_number: str):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    variant_to_promote = timing.alternative_images_list[variant_to_promote_frame_number]
    data_repo.update_specific_timing(
        timing_uuid, primary_image_id=variant_to_promote.uuid)

    prev_timing = data_repo.get_prev_timing(timing_uuid)
    if prev_timing:
        data_repo.update_specific_timing(
            prev_timing.uuid, interpolated_clip_list=None)
        data_repo.update_specific_timing(
            timing_uuid, interpolated_clip_list=None)

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    frame_idx = timing.aux_frame_index

    # DOUBT: setting last interpolated_video to empty?
    if frame_idx < len(timing_details):
        data_repo.update_specific_timing(
            timing.uuid, interpolated_clip_list=None)

    if frame_idx > 1:
        data_repo.update_specific_timing(
            data_repo.get_prev_timing(timing_uuid).uuid, timed_clip_id=None)

    data_repo.update_specific_timing(timing_uuid, timed_clip_id=None)

    if frame_idx < len(timing_details):
        data_repo.update_specific_timing(timing.uuid, timed_clip_id=None)

# updates the clip duration of the variant_to_promote and sets it as the timed_clip
def promote_video_variant(timing_uuid, variant_to_promote_frame_number: str):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    variant_to_promote = timing.interpolated_clip_list[variant_to_promote_frame_number]

    if variant_to_promote.location.startswith(('http://', 'https://')):
        temp_video_path, _ = urllib3.request.urlretrieve(variant_to_promote.location)
        video = VideoFileClip(temp_video_path)
    else:
        video = VideoFileClip(variant_to_promote.location)

    if video.duration != timing.clip_duration:
        video_bytes = VideoProcessor.update_video_speed(
            variant_to_promote.location,
            timing.animation_style,
            timing.clip_duration
        )

        hosted_url = save_or_host_file_bytes(video_bytes, variant_to_promote.local_path)
        if hosted_url:
            data_repo.update_file(video.uuid, hosted_url=hosted_url)

    data_repo.update_specific_timing(timing.uuid, timed_clip_id=variant_to_promote.uuid)



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
    return timing.mask.location

# adds the image file in variant (alternative images) list
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
        filename = str(uuid.uuid4()) + ".png"
        file_path = "videos/training_data/" + filename
        hosted_url = save_or_host_file(img, file_path)
        data = {
            "name": str(uuid.uuid4()),
            "type": InternalFileType.IMAGE.value,
        }

        if hosted_url:
            data['hosted_url'] = hosted_url
        else:
            data['local_path'] = file_path

        image_file = data_repo.create_file(**data)
        file_list.append(image_file)
    return file_list

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

# TODO: don't save or upload image where just passing the PIL object can work
def resize_image(video_name, new_width, new_height, image_file: InternalFileObject) -> InternalFileObject:
    if 'http' in image_file.location:
        response = r.get(image_file.location)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file.location)
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

    return image_file

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

# calculates and updates clip duration of all the timings
def update_clip_duration_of_all_timing_frames(project_uuid):
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

        total_duration_of_frame = round(total_duration_of_frame, 2)
        data_repo.update_specific_timing(timing_item.uuid, clip_duration=total_duration_of_frame)


def create_timings_row_at_frame_number(project_uuid, index_of_frame, frame_time=0.0):
    data_repo = DataRepo()
    
    # remove the interpolated video from the current row and the row before and after - unless it is the first or last row
    timing_data = {
        "project_id": project_uuid,
        "frame_time": frame_time,
        "animation_style": AnimationStyleType.INTERPOLATION.value,
        "aux_frame_index": index_of_frame
    }
    timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)

    prev_timing: InternalFrameTimingObject = data_repo.get_prev_timing(
        timing.uuid)
    if prev_timing:
        prev_clip_duration = calculate_desired_duration_of_individual_clip(prev_timing.uuid)
        data_repo.update_specific_timing(
            prev_timing.uuid, interpolated_clip_list=None, clip_duration=prev_clip_duration)

    next_timing: InternalAIModelObject = data_repo.get_next_timing(timing.uuid)
    if next_timing:
        data_repo.update_specific_timing(
            next_timing.uuid, interpolated_clip_list=None)

    return timing


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

def execute_image_edit(type_of_mask_selection, type_of_mask_replacement,
                       background_image, editing_image, prompt, negative_prompt,
                       width, height, layer, timing_uuid) -> InternalFileObject:
    from ui_components.methods.ml_methods import inpainting, remove_background, create_depth_mask_image
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project = timing.project

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


# if the output is present it adds it to the respective place or else it updates the inference log
# NOTE: every function used in this should not change/modify session state in anyway
def process_inference_output(**kwargs):
    data_repo = DataRepo()

    inference_type = kwargs.get('inference_type')
    # ------------------- FRAME TIMING IMAGE INFERENCE -------------------
    if inference_type == InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value:
        output = kwargs.get('output')
        if output:
            timing_uuid = kwargs.get('timing_uuid')
            promote_new_generation = kwargs.get('promote_new_generation')

            timing = data_repo.get_timing_from_uuid(timing_uuid)
            if not timing:
                return False
            
            filename = str(uuid.uuid4()) + ".png"
            log_uuid = kwargs.get('log_uuid')
            log = data_repo.get_inference_log_from_uuid(log_uuid)
            output_file = data_repo.create_file(
                name=filename, 
                type=InternalFileType.IMAGE.value,
                hosted_url=output[0], 
                inference_log_id=log.uuid
            )
            
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
        else:
            log_uuid = kwargs.get('log_uuid')
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)
    
    # --------------------- SINGLE PREVIEW VIDEO INFERENCE -------------------
    elif inference_type == InferenceType.SINGLE_PREVIEW_VIDEO.value:
        output = kwargs.get('output')
        file_bytes = None
        if isinstance(output, str) and output.startswith('http'):
            temp_output_file = generate_temp_file(output, '.mp4')
            file_bytes = None
            with open(temp_output_file.name, 'rb') as f:
                file_bytes = f.read()

            os.remove(temp_output_file.name)

        if file_bytes:
            file_data = {
                "file_location_to_save": kwargs.get('file_location_to_save'),
                "mime_type": kwargs.get('mime_type'),
                "file_bytes": file_bytes,
                "project_uuid": kwargs.get('project_uuid'),
                "inference_log_id": kwargs.get('log_uuid')
            }

            timing_uuid = kwargs.get('timing_uuid')
            timing = data_repo.get_timing_from_uuid(timing_uuid)
            if not timing:
                return False
            
            video_fie = convert_bytes_to_file(**file_data)
            data_repo.add_interpolated_clip(timing_uuid, interpolated_clip_id=video_fie.uuid)

        else:
            log_uuid = kwargs.get('log_uuid')
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)
    
    # --------------------- MULTI VIDEO INFERENCE (INTERPOLATION + MORPHING) -------------------
    elif inference_type == InferenceType.FRAME_INTERPOLATION.value:
        output = kwargs.get('output')
        log_uuid = kwargs.get('log_uuid')

        if output:
            settings = kwargs.get('settings')
            timing_uuid = kwargs.get('timing_uuid')
            timing = data_repo.get_timing_from_uuid(timing_uuid)
            if not timing:
                return False
            
            # output can also be an url
            if isinstance(output, str) and output.startswith("http"):
                temp_output_file = generate_temp_file(output, '.mp4')
                output = None
                with open(temp_output_file.name, 'rb') as f:
                    output = f.read()

                os.remove(temp_output_file.name)

            if 'normalise_speed' in settings and settings['normalise_speed']:
                output = VideoProcessor.update_video_bytes_speed(output, AnimationStyleType.INTERPOLATION.value, timing.clip_duration)

            video_location = "videos/" + str(timing.project.uuid) + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".mp4"
            video = convert_bytes_to_file(
                file_location_to_save=video_location,
                mime_type="video/mp4",
                file_bytes=output,
                project_uuid=timing.project.uuid,
                inference_log_id=log_uuid
            )

            data_repo.add_interpolated_clip(timing_uuid, interpolated_clip_id=video.uuid)
            if not timing.timed_clip:
                output_video = update_speed_of_video_clip(video, timing_uuid)
                data_repo.update_specific_timing(timing_uuid, timed_clip_id=output_video.uuid)
        
        else:
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    return True