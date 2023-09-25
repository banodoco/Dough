from math import gcd
import os
import time
import uuid
import streamlit as st
from PIL import ImageOps, Image
from streamlit_cropper import st_cropper
from backend.models import InternalFileObject
from shared.constants import InternalFileType

from ui_components.methods.common_methods import apply_image_transformations, fetch_image_by_stage, inpaint_in_black_space_element, reset_zoom_element, save_zoomed_image, zoom_inputs
from ui_components.constants import WorkflowStageType
from ui_components.methods.file_methods import generate_pil_image, save_or_host_file
from ui_components.models import InternalProjectObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo


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
            input_image, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'])
        st.image(output_image, use_column_width=True)

        if st.button("Save Image"):
            save_zoomed_image(output_image, st.session_state['current_frame_uuid'], stage, promote=True)
            st.success("Image saved successfully!")
            time.sleep(1)
            st.experimental_rerun()

        inpaint_in_black_space_element(
            output_image, project_settings.project.uuid, stage)


from math import gcd
import os
import time
import uuid
import streamlit as st
from PIL import ImageOps, Image
from streamlit_cropper import st_cropper
from backend.models import InternalFileObject
from shared.constants import InternalFileType

from ui_components.methods.common_methods import apply_image_transformations, fetch_image_by_stage, inpaint_in_black_space_element, reset_zoom_element, save_zoomed_image, zoom_inputs
from ui_components.constants import WorkflowStageType
from ui_components.methods.file_methods import generate_pil_image, save_or_host_file
from ui_components.models import InternalProjectObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo


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
