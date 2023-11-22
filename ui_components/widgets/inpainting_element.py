import json
import random
import string
import time
from io import BytesIO
from typing import List
import uuid
import numpy as np
import requests as r
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shared.constants import InferenceType, InternalFileTag, InternalFileType, ProjectMetaData
from ui_components.constants import CROPPED_IMG_LOCAL_PATH, MASK_IMG_LOCAL_PATH, TEMP_MASK_FILE, DefaultProjectSettingParams, WorkflowStageType
from ui_components.methods.file_methods import add_temp_file_to_project, save_or_host_file
from utils.data_repo.data_repo import DataRepo

from utils import st_memory
from utils.data_repo.data_repo import DataRepo
from ui_components.methods.common_methods import add_image_variant, execute_image_edit, create_or_update_mask, process_inference_output, promote_image_variant
from ui_components.models import InternalFrameTimingObject, InternalProjectObject, InternalSettingObject
from streamlit_image_comparison import image_comparison


def inpainting_element(timing_uuid):

    stage = WorkflowStageType.STYLED.value

    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_shot(
        timing.shot.uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.shot.project.uuid)

    if "type_of_mask_replacement" not in st.session_state:
        st.session_state["type_of_mask_replacement"] = "Replace With Image"
        st.session_state["index_of_type_of_mask_replacement"] = 0

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

        if len(timing.alternative_images_list) == 0:
            st.info("You need to add a style first in the Style Selection section.")
        else:
            
            variants = timing.alternative_images_list
            editing_image = timing.primary_image_location if timing.primary_image_location is not None else ""

            width = int(project_settings.width)
            height = int(project_settings.height)

            if editing_image == "":
                st.error(
                    f"You don't have a {stage} image yet so you can't edit it.")
            else:
                with main_col_1:
                    if 'index_of_type_of_mask_selection' not in st.session_state:
                        st.session_state['index_of_type_of_mask_selection'] = 0

                    type_of_mask_selection = "Manual Background Selection"

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
                            col1, col2 = st.columns([8, 3])

                            with col1:
                                st.session_state['drawing_input'] = st.radio(
                                    "Drawing tool:",
                                    ("Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Make squares □", "Draw lines ✏️"), horizontal=True)

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

                            is_completely_transparent = np.all(canvas_result.image_data[:, :, 3] == 0) \
                                if canvas_result.image_data is not None else False
                            
                            if not is_completely_transparent and canvas_result.image_data is not None:
                                img_data = canvas_result.image_data
                                im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
                                im_rgb = Image.new("RGB", im.size, (255, 255, 255))
                                im_rgb.paste(im, mask=im.split()[3])
                                im = im_rgb
                                im = ImageOps.invert(im)    # inverting for sdxl inpainting
                                create_or_update_mask(st.session_state['current_frame_uuid'], im)
                    else:
                        image_file = data_repo.get_file_from_uuid(st.session_state['edited_image'])
                        image_comparison(
                            img1=editing_image,
                            img2=image_file.location, starting_position=5, label1="Original", label2="Edited")
                        if st.button("Reset Canvas"):
                            st.session_state['edited_image'] = ""
                            st.rerun()


                with main_col_1:
         
                    st.session_state["type_of_mask_replacement"] = "Inpainting"
                    
                    btn1, btn2 = st.columns([1, 1])
                    with btn1:
                        prompt = st.text_area("Prompt:", help="Describe the whole image, but focus on the details you want changed!",
                                                value=DefaultProjectSettingParams.batch_prompt)
                    with btn2:
                        negative_prompt = st.text_area(
                            "Negative Prompt:", help="Enter any things you want to make the model avoid!", value=DefaultProjectSettingParams.batch_negative_prompt)

                    edit1, edit2 = st.columns(2)

                    with edit1:
                        if st.button(f'Run Edit On Current Image'):
                            if st.session_state["type_of_mask_replacement"] == "Inpainting":
                                edited_image, log = execute_image_edit(
                                                        type_of_mask_selection, 
                                                        st.session_state["type_of_mask_replacement"],
                                                        "", 
                                                        editing_image, 
                                                        prompt, 
                                                        negative_prompt, 
                                                        width, 
                                                        height, 
                                                        'Foreground', 
                                                        st.session_state['current_frame_uuid']
                                                    )
                                
                                inference_data = {
                                    "inference_type": InferenceType.FRAME_INPAINTING.value,
                                    "output": edited_image,
                                    "log_uuid": log.uuid,
                                    "timing_uuid": st.session_state['current_frame_uuid'],
                                    "promote_generation": False,
                                    "stage": stage
                                }

                                process_inference_output(**inference_data)

     


def replace_with_image(stage, output_file, current_frame_uuid, promote=False):
    data_repo = DataRepo()

    if stage == WorkflowStageType.SOURCE.value:
        data_repo.update_specific_timing(current_frame_uuid, source_image_id=output_file.uuid)
    elif stage == WorkflowStageType.STYLED.value:
        number_of_image_variants = add_image_variant(output_file.uuid, current_frame_uuid)
        if promote:
            promote_image_variant(current_frame_uuid, number_of_image_variants - 1)
    
    st.rerun()

# cropped_img here is a PIL image object
def inpaint_in_black_space_element(cropped_img, project_uuid, stage=WorkflowStageType.SOURCE.value):
    from ui_components.methods.ml_methods import inpainting

    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        project_uuid)

    st.markdown("##### Inpaint in black space:")

    inpaint_prompt = st.text_area("Prompt", value=DefaultProjectSettingParams.batch_prompt)
    inpaint_negative_prompt = st.text_input(
        "Negative Prompt", value='edge,branches, frame, fractals, text' + DefaultProjectSettingParams.batch_negative_prompt)
    if 'precision_cropping_inpainted_image_uuid' not in st.session_state:
        st.session_state['precision_cropping_inpainted_image_uuid'] = ""

    def inpaint(promote=False):
        width = int(project_settings.width)
        height = int(project_settings.height)

        saved_cropped_img = cropped_img.resize(
            (width, height), Image.ANTIALIAS)
        
        hosted_cropped_img_path = save_or_host_file(saved_cropped_img, CROPPED_IMG_LOCAL_PATH)
        mask = Image.new('RGB', cropped_img.size)
        width, height = cropped_img.size
        for x in range(width):
            for y in range(height):
                pixel = cropped_img.getpixel((x, y))
                if cropped_img.mode == 'RGB':
                    r, g, b = pixel
                elif cropped_img.mode == 'RGBA':
                    r, g, b, a = pixel
                elif cropped_img.mode == 'L':
                    brightness = pixel
                else:
                    raise ValueError(
                        f'Unsupported image mode: {cropped_img.mode}')

                if r == 0 and g == 0 and b == 0:
                    mask.putpixel((x, y), (0, 0, 0))  # Black
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if 0 <= x + i < width and 0 <= y + j < height:
                                mask.putpixel((x + i, y + j),
                                              (0, 0, 0))  # Black
                else:
                    mask.putpixel((x, y), (255, 255, 255))  # White

        mask = ImageOps.invert(mask)
        hosted_url = save_or_host_file(mask, MASK_IMG_LOCAL_PATH)
        add_temp_file_to_project(project_uuid, TEMP_MASK_FILE, hosted_url or MASK_IMG_LOCAL_PATH)

        cropped_img_path = hosted_cropped_img_path if hosted_cropped_img_path else CROPPED_IMG_LOCAL_PATH
        output, log = inpainting(cropped_img_path, inpaint_prompt,
                                    inpaint_negative_prompt, st.session_state['current_frame_uuid'], True)
        
        inference_data = {
            "inference_type": InferenceType.FRAME_INPAINTING.value,
            "output": output,
            "log_uuid": log.uuid,
            "timing_uuid": st.session_state['current_frame_uuid'],
            "promote_generation": promote,
            "stage": stage
        }

        process_inference_output(**inference_data)


    if st.button("Inpaint"):
        inpaint(promote=False)
    
    if st.button("Inpaint and Promote"):
        inpaint(promote=True)