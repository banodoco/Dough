from math import gcd
import time
import uuid
import streamlit as st
from PIL import ImageOps, Image
from streamlit_cropper import st_cropper
from backend.models import InternalFileObject
from shared.constants import InternalFileType

from ui_components.methods.common_methods import apply_image_transformations, fetch_image_by_stage
from ui_components.constants import WorkflowStageType
from ui_components.methods.file_methods import generate_pil_image, save_or_host_file
from ui_components.models import InternalProjectObject
from ui_components.widgets.image_zoom_widgets import reset_zoom_element, save_zoomed_image, zoom_inputs
from ui_components.widgets.inpainting_element import inpaint_in_black_space_element
from utils.data_repo.data_repo import DataRepo

from utils import st_memory


def cropping_selector_element(shot_uuid):
    selector1, selector2, _ = st.columns([1, 1, 1])
    with selector1:
        # crop_stage = st_memory.radio("Which stage to work on?", ["Styled Key Frame", "Unedited Key Frame"], key="crop_stage", horizontal=True)
        crop_stage = "Styled Key Frame"
        # how_to_crop = st_memory.radio("How to crop:", options=["Precision Cropping","Manual Cropping"], key="how_to_crop",horizontal=True)
        how_to_crop = "Precision Cropping"
    
        
                                        
    if crop_stage == "Styled Key Frame":
        stage_name = WorkflowStageType.STYLED.value
    elif crop_stage == "Unedited Key Frame":
        stage_name = WorkflowStageType.SOURCE.value
                                        
    if how_to_crop == "Manual Cropping":
        manual_cropping_element(stage_name, st.session_state['current_frame_uuid'])
    elif how_to_crop == "Precision Cropping":
        precision_cropping_element(stage_name, shot_uuid)

def precision_cropping_element(stage, shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    input_image = fetch_image_by_stage(shot_uuid, stage, st.session_state['current_frame_index'] - 1)

    if 'zoom_level_input' not in st.session_state:
        st.session_state['zoom_level_input'] = 100
        st.session_state['rotation_angle_input'] = 0
        st.session_state['x_shift'] = 0
        st.session_state['y_shift'] = 0
        st.session_state['flip_vertically'] = False
        st.session_state['flip_horizontally'] = False

    # TODO: CORRECT-CODE check if this code works
    if not input_image:
        st.error("Please select a source image before cropping")
        return
    else:
        input_image = generate_pil_image(input_image.location)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Precision Cropping:")

        if st.button("Reset Cropping"):
            reset_zoom_element()
        
        
        zoom_inputs()
        st.caption("Input Image:")
        st.image(input_image, caption="Input Image", width=300)

    with col2:
        st.caption("Output Image:")
        output_image = apply_image_transformations(
            input_image, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'], st.session_state['flip_vertically'], st.session_state['flip_horizontally'])
        st.image(output_image, use_column_width=True)

        if st.button("Save Image"):
            save_zoomed_image(output_image, st.session_state['current_frame_uuid'], stage, promote=True)
            st.success("Image saved successfully!")
            time.sleep(1)
            st.rerun()

        inpaint_in_black_space_element(output_image, shot.project.uuid, stage, shot_uuid)


def manual_cropping_element(stage, timing_uuid):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.shot.project.uuid

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

        if 'current_stage' not in st.session_state:
            st.session_state['current_stage'] = stage

        def get_working_image():
            st.session_state['working_image'] = generate_pil_image(input_image)
            st.session_state['working_image'] = st.session_state['working_image'].convert('RGB')
            st.session_state['working_image'] = ImageOps.expand(
                st.session_state['working_image'], border=200, fill="black")
            st.session_state['current_working_image_number'] = st.session_state['current_frame_index']
            st.session_state['current_stage'] = stage

        if 'working_image' not in st.session_state or st.session_state['current_working_image_number'] != st.session_state['current_frame_index'] or st.session_state['current_stage'] != stage:
            get_working_image()
            st.rerun()

        options1, _, _, _ = st.columns([3, 1, 1, 1])
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
                    st.rerun()

            with sub_options_2:
                st.write("")
                st.write("")
                if st.button("Reset image"):
                    st.session_state['degree'] = 0
                    get_working_image()
                    st.session_state['degrees_rotated_to'] = 0
                    st.rerun()
        
        project_settings: InternalProjectObject = data_repo.get_project_setting(
            timing.shot.project.uuid)

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
                    save_zoomed_image(cropped_img, st.session_state['current_frame_uuid'], stage, promote=True)                        
                    st.success("Image saved successfully!")
                    time.sleep(0.5)
                    st.rerun()
                
            with cropbtn2:
                st.warning("Warning: This will overwrite the original image")

            inpaint_in_black_space_element(cropped_img, project_uuid, stage, timing.shot.uuid)
