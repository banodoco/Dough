import uuid
import streamlit as st
from backend.models import InternalFileObject
from shared.constants import InternalFileType
from ui_components.constants import WorkflowStageType
from ui_components.methods.common_methods import add_image_variant, promote_image_variant
from ui_components.methods.file_methods import save_or_host_file

from utils.data_repo.data_repo import DataRepo


def zoom_inputs(position="in-frame", horizontal=False, shot_uuid=None):
    if horizontal:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)
    else:
        col1 = col2 = col3 = col4 = col5 = col6 = st

    if "zoom_level_input_default" not in st.session_state:
        st.session_state["zoom_level_input_default"] = 100
        st.session_state["rotation_angle_input_default"] = 0
        st.session_state["x_shift_default"] = 0
        st.session_state["y_shift_default"] = 0
        st.session_state["flip_vertically_default"] = False
        st.session_state["flip_horizontally_default"] = False

    col1.number_input(
        "Zoom in/out:",
        min_value=10,
        max_value=1000,
        step=5,
        key=f"zoom_level_input",
        value=st.session_state["zoom_level_input_default"],
    )

    col2.number_input(
        "Rotate:",
        min_value=-360,
        max_value=360,
        step=5,
        key="rotation_angle_input",
        value=st.session_state["rotation_angle_input_default"],
    )
    # st.session_state['rotation_angle_input'] = 0

    col3.number_input(
        "Shift left/right:",
        min_value=-1000,
        max_value=1000,
        step=5,
        key=f"x_shift",
        value=st.session_state["x_shift_default"],
    )

    col4.number_input(
        "Shift down/up:",
        min_value=-1000,
        max_value=1000,
        step=5,
        key=f"y_shift",
        value=st.session_state["y_shift_default"],
    )

    col5.checkbox(
        "Flip vertically ↕️", key=f"flip_vertically", value=str(st.session_state["flip_vertically_default"])
    )

    col6.checkbox(
        "Flip horizontally ↔️",
        key=f"flip_horizontally",
        value=str(st.session_state["flip_horizontally_default"]),
    )


def save_zoomed_image(image, timing_uuid, stage, promote=False):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.shot.project.uuid

    file_name = str(uuid.uuid4()) + ".png"

    if stage == WorkflowStageType.SOURCE.value:
        save_location = f"videos/{project_uuid}/assets/frames/modified/{file_name}"
        hosted_url = save_or_host_file(image, save_location)
        file_data = {"name": file_name, "type": InternalFileType.IMAGE.value, "project_id": project_uuid}

        if hosted_url:
            file_data.update({"hosted_url": hosted_url})
        else:
            file_data.update({"local_path": save_location})

        source_image: InternalFileObject = data_repo.create_file(**file_data)
        data_repo.update_specific_timing(
            st.session_state["current_frame_uuid"], source_image_id=source_image.uuid, update_in_place=True
        )
    elif stage == WorkflowStageType.STYLED.value:
        save_location = f"videos/{project_uuid}/assets/frames/modified/{file_name}"
        hosted_url = save_or_host_file(image, save_location)
        file_data = {"name": file_name, "type": InternalFileType.IMAGE.value, "project_id": project_uuid}

        if hosted_url:
            file_data.update({"hosted_url": hosted_url})
        else:
            file_data.update({"local_path": save_location})

        styled_image: InternalFileObject = data_repo.create_file(**file_data)
        number_of_image_variants = add_image_variant(styled_image.uuid, timing_uuid)
        if promote:
            promote_image_variant(timing_uuid, number_of_image_variants - 1)


def reset_zoom_element():
    st.session_state["zoom_level_input_default"] = 100
    st.session_state["zoom_level_input"] = 100
    st.session_state["rotation_angle_input_default"] = 0
    st.session_state["rotation_angle_input"] = 0
    st.session_state["x_shift_default"] = 0
    st.session_state["x_shift"] = 0
    st.session_state["y_shift_default"] = 0
    st.session_state["y_shift"] = 0
    st.session_state["flip_vertically_default"] = False
    st.session_state["flip_vertically"] = False
    st.session_state["flip_horizontally_default"] = False
    st.session_state["flip_horizontally"] = False
