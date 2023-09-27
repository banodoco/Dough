import io
from typing import List
import streamlit as st
from streamlit_drawable_canvas import st_canvas
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
from shared.constants import SERVER, InternalFileType, ServerType
from pydub import AudioSegment
from backend.models import InternalFileObject
from ui_components.constants import CROPPED_IMG_LOCAL_PATH, MASK_IMG_LOCAL_PATH, SECOND_MASK_FILE, SECOND_MASK_FILE_PATH, TEMP_MASK_FILE, WorkflowStageType
from ui_components.methods.file_methods import add_temp_file_to_project, generate_pil_image, save_or_host_file, save_or_host_file_bytes
from ui_components.methods.ml_methods import create_depth_mask_image, inpainting, remove_background
from ui_components.methods.video_methods import calculate_desired_duration_of_individual_clip
from ui_components.models import InternalAIModelObject, InternalFrameTimingObject, InternalSettingObject
from utils.constants import ImageStage
from utils.data_repo.data_repo import DataRepo
from shared.constants import AnimationStyleType

from ui_components.models import InternalFileObject

from typing import Union
from streamlit_image_comparison import image_comparison


def add_key_frame(selected_image, inherit_styling_settings, how_long_after, which_stage_for_starting_image):
    data_repo = DataRepo()
    project_uuid = st.session_state['project_uuid']
    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    project_settings = data_repo.get_project_setting(project_uuid)
    

    if len(timing_details) == 0:
        index_of_current_item = 1
    else:
        index_of_current_item = min(len(timing_details), st.session_state['current_frame_index'])

    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    if len(timing_details) == 0:
        key_frame_time = 0.0
    elif index_of_current_item == len(timing_details):
        key_frame_time = float(timing_details[index_of_current_item - 1].frame_time) + how_long_after
    else:
        key_frame_time = (float(timing_details[index_of_current_item - 1].frame_time) + float(
            timing_details[index_of_current_item].frame_time)) / 2.0

    if len(timing_details) == 0:
        new_timing = create_timings_row_at_frame_number(project_uuid, 0)
    else:
        new_timing = create_timings_row_at_frame_number(project_uuid, index_of_current_item, frame_time=key_frame_time)
        
        clip_duration = calculate_desired_duration_of_individual_clip(new_timing.uuid)
        data_repo.update_specific_timing(new_timing.uuid, clip_duration=clip_duration)

    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    if selected_image:
        save_uploaded_image(selected_image, project_uuid, timing_details[index_of_current_item].uuid, "source")
        save_uploaded_image(selected_image, project_uuid, timing_details[index_of_current_item].uuid, "styled")

    if inherit_styling_settings == "Yes":    
        clone_styling_settings(index_of_current_item - 1, timing_details[index_of_current_item].uuid)

    data_repo.update_specific_timing(timing_details[index_of_current_item].uuid, \
                                        animation_style=project_settings.default_animation_style)

    if len(timing_details) == 1:
        st.session_state['current_frame_index'] = 1
        st.session_state['current_frame_uuid'] = timing_details[0].uuid
    else:
        st.session_state['current_frame_index'] = min(len(timing_details), st.session_state['current_frame_index'])
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid

    st.session_state['page'] = "Styling"
    st.session_state['section_index'] = 0
    st.experimental_rerun()

def clone_styling_settings(source_frame_number, target_frame_uuid):
    data_repo = DataRepo()
    target_timing = data_repo.get_timing_from_uuid(target_frame_uuid)
    timing_details = data_repo.get_timing_list_from_project(
        target_timing.project.uuid)

    data_repo.update_specific_timing(
        target_frame_uuid, 
        custom_pipeline=timing_details[source_frame_number].custom_pipeline,
        negative_prompt=timing_details[source_frame_number].negative_prompt,
        guidance_scale=timing_details[source_frame_number].guidance_scale,
        seed=timing_details[source_frame_number].seed,
        num_inteference_steps=timing_details[source_frame_number].num_inteference_steps,
        transformation_stage=timing_details[source_frame_number].transformation_stage,
        strength=timing_details[source_frame_number].strength,
        custom_models=timing_details[source_frame_number].custom_model_id_list,
        adapter_type=timing_details[source_frame_number].adapter_type,
        low_threshold=timing_details[source_frame_number].low_threshold,
        high_threshold=timing_details[source_frame_number].high_threshold,
        prompt=timing_details[source_frame_number].prompt
    )
    
    if timing_details[source_frame_number].model:
        data_repo.update_specific_timing(
            target_frame_uuid, model_id=timing_details[source_frame_number].model.uuid)

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

def fetch_image_by_stage(project_uuid, stage):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    if stage == WorkflowStageType.SOURCE.value:
        return timing_details[st.session_state['current_frame_index'] - 1].source_image
    elif stage == WorkflowStageType.STYLED.value:
        return timing_details[st.session_state['current_frame_index'] - 1].primary_image
    else:
        return None

def zoom_inputs(position='in-frame', horizontal=False):
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

def move_frame(direction, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if direction == "Up":
        if timing.aux_frame_index == 0:
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index - 1)

    elif direction == "Down":
        timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)
        if timing.aux_frame_index == len(timing_list) - 1:
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index + 1)

    # updating clip_duration
    update_clip_duration_of_all_timing_frames(timing.project.uuid)

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

def replace_image_widget(timing_uuid, stage):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(timing.project.uuid)
                                        
    replace_with = st.radio("Replace with:", [
        "Uploaded Frame", "Other Frame"], horizontal=True, key=f"replace_with_what_{stage}")
    

    if replace_with == "Other Frame":
                    
        which_stage_to_use_for_replacement = st.radio("Select stage to use:", [
            ImageStage.MAIN_VARIANT.value, ImageStage.SOURCE_IMAGE.value], key=f"which_stage_to_use_for_replacement_{stage}", horizontal=True)
        which_image_to_use_for_replacement = st.number_input("Select image to use:", min_value=0, max_value=len(
            timing_details)-1, value=0, key=f"which_image_to_use_for_replacement_{stage}")
        
        if which_stage_to_use_for_replacement == ImageStage.SOURCE_IMAGE.value:                                    
            selected_image = timing_details[which_image_to_use_for_replacement].source_image
            
        
        elif which_stage_to_use_for_replacement == ImageStage.MAIN_VARIANT.value:
            selected_image = timing_details[which_image_to_use_for_replacement].primary_image
            
        
        st.image(selected_image.local_path, use_column_width=True)

        if st.button("Replace with selected frame", disabled=False,key=f"replace_with_selected_frame_{stage}"):
            if stage == "source":
                                            
                data_repo.update_specific_timing(timing.uuid, source_image_id=selected_image.uuid)                                        
                st.success("Replaced")
                time.sleep(1)
                st.experimental_rerun()
                
            else:
                number_of_image_variants = add_image_variant(
                    selected_image.uuid, timing.uuid)
                promote_image_variant(
                    timing.uuid, number_of_image_variants - 1)
                st.success("Replaced")
                time.sleep(1)
                st.experimental_rerun()
                                            
    elif replace_with == "Uploaded Frame":
        if stage == "source":
            uploaded_file = st.file_uploader("Upload Source Image", type=[
                "png", "jpeg"], accept_multiple_files=False)
            if st.button("Upload Source Image"):
                if uploaded_file:
                    timing = data_repo.get_timing_from_uuid(timing.uuid)
                    if save_uploaded_image(uploaded_file, timing.project.uuid, timing.uuid, "source"):
                        time.sleep(1.5)
                        st.experimental_rerun()
        else:
            replacement_frame = st.file_uploader("Upload a replacement frame here", type=[
                "png", "jpeg"], accept_multiple_files=False, key=f"replacement_frame_upload_{stage}")
            if st.button("Replace frame", disabled=False):
                images_for_model = []
                timing = data_repo.get_timing_from_uuid(timing.uuid)
                if replacement_frame:
                    saved_file = save_uploaded_image(replacement_frame, timing.project.uuid, timing.uuid, "styled")
                    if saved_file:
                        number_of_image_variants = add_image_variant(saved_file.uuid, timing.uuid)
                        promote_image_variant(
                            timing.uuid, number_of_image_variants - 1)
                        st.success("Replaced")
                        time.sleep(1)
                        st.experimental_rerun()

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
            if stage == WorkflowStageType.SOURCE.value:
                image_path = timing_details[st.session_state['current_frame_index'] - 1].source_image.location 
        
            elif stage == WorkflowStageType.STYLED.value:
                image_path = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
            
            
            canny_image = extract_canny_lines(
                    image_path, project_uuid, low_threshold, high_threshold)
            
            st.session_state['canny_image'] = canny_image.uuid

        if st.session_state['canny_image']:
            canny_image = data_repo.get_file_from_uuid(st.session_state['canny_image'])
            
            canny_action_1, canny_action_2 = st.columns([2, 1])
            with canny_action_1:
                st.image(canny_image.location)
                                                                            
                if st.button(f"Make Into Guidance Image"):
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

    # not uploading or removing the created image as of now
    # resized_image = upload_image(
    #     "videos/" + str(video_name) + "/temp_image.png")

    # os.remove("videos/" + str(video_name) + "/temp_image.png")

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

        duration_of_static_time = 0.0

        data_repo.update_specific_timing(
            timing_item.uuid, clip_duration=total_duration_of_frame)

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
