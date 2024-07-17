import time
import streamlit as st
from shared.constants import AppSubPage, CreativeProcessPage
from ui_components.constants import WorkflowStageType
from ui_components.methods.common_methods import (
    add_image_variant,
    promote_image_variant,
    save_and_promote_image,
)
from ui_components.models import InternalFrameTimingObject
from utils.state_refresh import refresh_app
from utils.constants import ImageStage

from utils.data_repo.data_repo import DataRepo
from utils.state_refresh import refresh_app


def change_frame_shot(timing_uuid, src):
    """
    used to move a frame from one shot to another
    """
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.shot.project.uuid

    shot_list = data_repo.get_shot_list(project_uuid)
    shot_names = [shot.name for shot in shot_list]

    new_shot = st.selectbox("Move to new shot:", shot_names, key=f"new_shot_{timing.uuid}_{src}")
    if st.button("Move to shot", key=f"change_frame_position_{timing.uuid}_{src}", use_container_width=True):
        shot = next(
            (obj for obj in shot_list if obj.name == new_shot), None
        )  # NOTE: this assumes unique name for different shots
        if shot:
            data_repo.update_specific_timing(timing.uuid, shot_id=shot.uuid)
            st.success("Success")
            time.sleep(0.3)
            refresh_app()


def move_frame(direction, timing_uuid):
    """
    arrows that change frame position by 1 step
    """
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    if direction == "Up":
        if timing.aux_frame_index == 0:
            st.error("This is the first frame")
            time.sleep(0.5)
            return

        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index - 1)
    elif direction == "Down":
        timing_list = data_repo.get_timing_list_from_shot(timing.shot.uuid)
        if timing.aux_frame_index == len(timing_list) - 1:
            st.error("This is the last frame")
            time.sleep(0.5)
            return

        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index + 1)


def move_frame_back_button(timing_uuid, orientation):
    direction = "Up"
    if orientation == "side-to-side":
        arrow = "⬅️"
    else:  # up-down
        arrow = "⬆️"
    if st.button(
        arrow, key=f"move_frame_back_{timing_uuid}", help="Move frame back", use_container_width=True
    ):
        move_frame(direction, timing_uuid)
        refresh_app(maintain_state=True)


def move_frame_forward_button(timing_uuid, orientation):
    direction = "Down"
    if orientation == "side-to-side":
        arrow = "➡️"
    else:  # up-down
        arrow = "⬇️"

    if st.button(
        arrow, key=f"move_frame_forward_{timing_uuid}", help="Move frame forward", use_container_width=True
    ):
        move_frame(direction, timing_uuid)
        refresh_app(maintain_state=True)


def delete_frame_button(timing_uuid, show_label=False):
    if show_label:
        label = "Delete Frame 🗑️"
    else:
        label = "🗑️"

    if st.button(label, key=f"delete_frame_{timing_uuid}", help="Delete frame", use_container_width=True):
        delete_frame(timing_uuid)
        refresh_app()


def delete_frame(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    shot_uuid = timing.shot.uuid
    next_timing = data_repo.get_next_timing(timing_uuid)
    timing_list = data_repo.get_timing_list_from_shot(timing.shot.uuid)

    data_repo.delete_timing_from_uuid(timing.uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    if len(timing_list) == 0:
        print("No more frames in this shot")
    # this is the last frame
    elif not next_timing:
        st.session_state["current_frame_index"] = max(1, st.session_state["current_frame_index"] - 1)
        st.session_state["prev_frame_index"] = st.session_state["current_frame_index"]
        st.session_state["current_frame_uuid"] = timing_list[st.session_state["current_frame_index"] - 1].uuid
    # this is the first frame or something in the middle
    else:
        st.session_state["current_frame_index"] = min(
            len(timing_list) - 1, st.session_state["current_frame_index"] + 1
        )
        st.session_state["prev_frame_index"] = st.session_state["current_frame_index"]
        st.session_state["current_frame_uuid"] = timing_list[st.session_state["current_frame_index"] - 1].uuid


def replace_image_widget(timing_uuid, stage, options=["Uploaded Frame", "Other Frame"]):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_list = data_repo.get_timing_list_from_shot(timing.shot.uuid)

    btn_text = "Upload source image" if stage == WorkflowStageType.SOURCE.value else "Replace frame"
    uploaded_file = st.file_uploader(
        btn_text,
        type=["png", "jpeg"],
        accept_multiple_files=False,
        key=f"uploaded_file_{stage}_{timing_uuid}",
    )
    if uploaded_file != None:
        if st.button(btn_text):
            if uploaded_file:
                timing = data_repo.get_timing_from_uuid(timing.uuid)
                if save_and_promote_image(uploaded_file, timing.shot.uuid, timing.uuid, stage):
                    st.success("Replaced")
                    time.sleep(1.5)
                    refresh_app()


"""
TODO: 
1. use timing_list to validate the range of display_number
2. set shot_uuid from the timing_list in the session_state as well
"""


def jump_to_single_frame_view_button(display_number, timing_list, src, uuid=None):
    if st.button(f"Jump to #{display_number}", key=f"{src}_{uuid}", use_container_width=True):
        st.session_state["current_frame_sidebar_selector"] = display_number
        st.session_state["current_subpage"] = AppSubPage.KEYFRAME.value
        st.session_state["page"] = CreativeProcessPage.ADJUST_SHOT.value
        refresh_app()
