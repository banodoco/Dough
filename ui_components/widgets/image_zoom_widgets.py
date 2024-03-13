import uuid
import streamlit as st
from backend.models import InternalFileObject
from shared.constants import InternalFileType
from ui_components.constants import WorkflowStageType
from ui_components.methods.common_methods import add_image_variant, promote_image_variant
from ui_components.methods.file_methods import save_or_host_file

from utils.data_repo.data_repo import DataRepo

def zoom_inputs(position='in-frame', horizontal=False):
    if horizontal:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
    else:
        col1 = col2 = col3 = col4 = col5 = col6 = st

    col1.number_input(
        "Zoom In/Out", min_value=10, max_value=1000, step=10, key=f"zoom_level_input", value=100)
    
    # col2.number_input(
    #     "Rotate Counterclockwise/Clockwise", min_value=-360, max_value=360, step=5, key="rotation_angle_input", value=0)
    st.session_state['rotation_angle_input'] = 0
    
    col3.number_input(
        "Shift Left/Right", min_value=-1000, max_value=1000, step=5, key=f"x_shift", value=0)
    
    col4.number_input(
        "Shift Down/Up", min_value=-1000, max_value=1000, step=5, key=f"y_shift", value=0)

    col5.checkbox(
        "Flip Vertically ↕️", key=f"flip_vertically", value=False)

    col6.checkbox(
        "Flip Horizontally ↔️", key=f"flip_horizontally", value=False)


def save_zoomed_image(image, timing_uuid, stage, promote=False):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.shot.project.uuid

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
            st.session_state['current_frame_uuid'], source_image_id=source_image.uuid, update_in_place=True)
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


def reset_zoom_element():
    st.session_state['zoom_level_input_key'] = 100
    st.session_state['rotation_angle_input_key'] = 0
    st.session_state['x_shift_key'] = 0
    st.session_state['y_shift_key'] = 0
    st.session_state['zoom_level_input'] = 100
    st.session_state['rotation_angle_input'] = 0
    st.session_state['x_shift'] = 0
    st.session_state['y_shift'] = 0
    st.session_state['flip_vertically'] = False
    st.session_state['flip_horizontally'] = False
    st.rerun()