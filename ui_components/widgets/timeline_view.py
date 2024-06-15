from io import BytesIO
import os
import shutil
from typing import List
import uuid
from zipfile import ZipFile
from ui_components.methods.file_methods import save_or_host_file
from ui_components.widgets.add_key_frame_element import add_key_frame
import requests
import streamlit as st
from ui_components.methods.common_methods import add_new_shot
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from ui_components.widgets.common_element import duplicate_shot_button
from ui_components.widgets.display_element import individual_video_display_element
from ui_components.widgets.shot_view import (
    shot_keyframe_element,
    shot_adjustment_button,
    shot_animation_button,
    update_shot_name,
    update_shot_duration,
    move_shot_buttons,
    delete_shot_button,
    create_video_download_button,
)
from utils.data_repo.data_repo import DataRepo
from utils import st_memory
from PIL import Image

def timeline_view(shot_uuid, stage, view='sidebar'):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    shot_list = data_repo.get_shot_list(shot.project.uuid)

    if view == 'main':
        _, header_col_2 = st.columns([5.5, 1.5])
        items_per_row = 4
    else:  # sidebar view
        items_per_row = 4  # Changed to 4 per row for sidebar view
        shot_list.reverse()  # Reverse the order of shots for sidebar view

    # Pagination setup for sidebar view
    if view == 'sidebar':
        total_pages = (len(shot_list) + items_per_row - 1) // items_per_row
        if total_pages > 1:
            page = st.radio("Select Page", list(range(1, total_pages + 1)), horizontal=True)
        else:
            page = 1
        start_index = (page - 1) * items_per_row
        end_index = min(start_index + items_per_row, len(shot_list))
        shot_list = shot_list[start_index:end_index]

        # Add new shot button at the top for sidebar view
        with st.container():
            st.markdown("### Add new shot")
            add_new_shot_element(shot, data_repo)
            st.markdown("***")

    for idx, shot in enumerate(shot_list):
        timing_list: List[InternalFrameTimingObject] = shot.timing_list
        if idx % items_per_row == 0:
            if view == 'main':
                grid = st.columns(items_per_row)
            else:
                # Ensure grid is only as large as the number of shots in the last segment
                grid = [st.container() for _ in range(min(items_per_row, len(shot_list) - idx))]

        with grid[idx % items_per_row]:
            st.info(f"##### {shot.name}")
            if shot.main_clip and shot.main_clip.location and view == 'main':
                individual_video_display_element(shot.main_clip)
            else:
                num_columns = 4  # Set to 4 images per row regardless of the number of images
                
                if timing_list:
                    grid_timing = st.columns(num_columns)
                    for j, timing in enumerate(timing_list):
                        with grid_timing[j % num_columns]:
                            if timing.primary_image and timing.primary_image.location:
                                st.image(timing.primary_image.location, use_column_width=True)
                    for j in range(len(timing_list), num_columns):
                        with grid_timing[j]:
                            st.empty()
                else:
                    st.warning("No images in shot.")  # Warning if no images are present

            switch1, switch2 = st.columns([1, 1])
            with switch1:
                shot_adjustment_button(shot)
            with switch2:
                shot_animation_button(shot)
            if view == 'main':
                with st.expander("Details & settings:", expanded=False):
                    update_shot_name(shot.uuid)
                    move_shot_buttons(shot, "side")
                    delete_shot_button(shot.uuid)
                    duplicate_shot_button(shot.uuid, position="timeline_view")
                    if shot.main_clip:
                        create_video_download_button(shot.main_clip.location, tag="main_clip")

        if (idx + 1) % items_per_row == 0 or idx == len(shot_list) - 1:
            st.markdown("***")

        if view == 'main' and idx == len(shot_list) - 1:
            with grid[(idx + 1) % items_per_row]:
                st.markdown("###### Add new shot")
                add_new_shot_element(shot, data_repo,show_image_uploader=True)

def add_new_shot_element(shot, data_repo, show_image_uploader=False):
    new_shot_name = st.text_input("Shot Name:", max_chars=25)
    if show_image_uploader:
        with st.expander("Upload images", expanded=False):
            uploaded_images = st.file_uploader(
                "Upload images:",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"uploaded_image_{shot.uuid}",
                help="You can upload multiple images",
                accept_multiple_files=True,
            )
    else:
        uploaded_images = None

    if st.button("Add new shot", type="primary", key=f"add_shot_btn_{shot.uuid}"):

        new_shot = add_new_shot(shot.project.uuid)
        if new_shot_name != "":
            data_repo.update_shot(uuid=new_shot.uuid, name=new_shot_name)
        project_uuid = shot.project.uuid
        shot_list = data_repo.get_shot_list(project_uuid)
        len_shot_list = len(shot_list) - 1
        st.session_state["last_shot_number"] = len_shot_list
        if uploaded_images:
            progress_bar = st.progress(0)
            uploaded_images = sorted(uploaded_images, key=lambda x: x.name)
            for i, uploaded_image in enumerate(uploaded_images):
                image = Image.open(uploaded_image)
                file_location = f"videos/{new_shot.uuid}/assets/frames/base/{uploaded_image.name}"
                selected_image_location = save_or_host_file(image, file_location)
                selected_image_location = selected_image_location or file_location
                add_key_frame(selected_image_location, new_shot.uuid, refresh_state=False)
                progress_bar.progress((i + 1) / len(uploaded_images))

        st.rerun()
