import time
from typing import Union
import streamlit as st
from shared.constants import AnimationStyleType
from ui_components.models import InternalFileObject, InternalFrameTimingObject
from utils.common_decorators import update_refresh_lock
from utils.state_refresh import refresh_app
from utils.data_repo.data_repo import DataRepo
from ui_components.methods.file_methods import generate_pil_image, save_or_host_file
from ui_components.methods.common_methods import add_image_variant, save_new_image
from PIL import Image


def add_key_frame_section(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    selected_image_location = ""

    uploaded_images = st.file_uploader(
        "Upload images:",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"uploaded_image_{shot_uuid}",
        help="You can upload multiple images",
        accept_multiple_files=True,
    )

    if st.button(
        f"Add key frame(s)",
        use_container_width=True,
        key=f"add_key_frame_btn_{shot_uuid}",
        type="primary",
    ):
        update_refresh_lock(True)
        if uploaded_images:
            progress_bar = st.progress(0)
            # Remove sorting to maintain upload order
            for i, uploaded_image in enumerate(uploaded_images):
                image = Image.open(uploaded_image)
                file_location = f"videos/{shot.uuid}/assets/frames/base/{uploaded_image.name}"
                selected_image_location = save_or_host_file(image, file_location)
                selected_image_location = selected_image_location or file_location
                add_key_frame(selected_image_location, shot_uuid, refresh_state=False)
                progress_bar.progress((i + 1) / len(uploaded_images))
        else:
            st.error("Please generate new images or upload them")
            time.sleep(0.7)
        update_refresh_lock(False)
        refresh_app()


def display_selected_key_frame(selected_image_location, apply_zoom_effects):
    selected_image = None
    if selected_image_location:
        # if apply_zoom_effects == "Yes":
        # image_preview = generate_pil_image(selected_image_location)
        # selected_image = apply_image_transformations(image_preview, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'], st.session_state['flip_vertically'], st.session_state['flip_horizontally'])

        selected_image = generate_pil_image(selected_image_location)
        st.info("Starting Image:")
        st.image(selected_image)
    else:
        st.error("No Starting Image Found")

    return selected_image


def add_key_frame_element(shot_uuid):
    add1, add2 = st.columns(2)
    with add1:
        selected_image_location, inherit_styling_settings = add_key_frame_section(shot_uuid)
    with add2:
        selected_image = display_selected_key_frame(selected_image_location, False)

    return selected_image, inherit_styling_settings


def add_key_frame(
    selected_image: Union[Image.Image, InternalFileObject],
    shot_uuid,
    target_frame_position=None,
    refresh_state=True,
    update_cur_frame_idx=True,
):
    """
    either a pil image or a internalfileobject can be passed to this method, for adding it inside a shot
    """
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    # creating frame inside the shot at target_frame_position
    len_shot_timing_list = len(timing_list) if len(timing_list) > 0 else 0
    target_frame_position = len_shot_timing_list if target_frame_position is None else target_frame_position
    target_aux_frame_index = min(len(timing_list), target_frame_position)

    if isinstance(selected_image, InternalFileObject):
        saved_image = selected_image
    else:
        shot = data_repo.get_shot_from_uuid(shot_uuid)
        saved_image = save_new_image(selected_image, shot.project.uuid)

    timing_data = {
        "shot_id": shot_uuid,
        "animation_style": AnimationStyleType.CREATIVE_INTERPOLATION.value,
        "aux_frame_index": target_aux_frame_index,
        "source_image_id": saved_image.uuid,
        "primary_image_id": saved_image.uuid,
    }
    new_timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)

    if update_cur_frame_idx:
        timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
        # this part of code updates current_frame_index when a new keyframe is added
        if len(timing_list) <= 1:
            st.session_state["current_frame_index"] = 1
            st.session_state["current_frame_uuid"] = timing_list[0].uuid
        else:
            st.session_state["prev_frame_index"] = min(len(timing_list), target_aux_frame_index + 1)
            st.session_state["current_frame_index"] = min(len(timing_list), target_aux_frame_index + 1)
            st.session_state["current_frame_uuid"] = timing_list[
                st.session_state["current_frame_index"] - 1
            ].uuid

        print(
            f"Updated session state: current_frame_index: {st.session_state['current_frame_index']}, current_frame_uuid: {st.session_state['current_frame_uuid']}"
        )

    if refresh_state:
        refresh_app()

    return new_timing
