from io import BytesIO
import os
import shutil
from typing import List
import uuid
from zipfile import ZipFile

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
def timeline_view(shot_uuid, stage, view='sidebar'):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    shot_list = data_repo.get_shot_list(shot.project.uuid)

    if view == 'main':
        _, header_col_2 = st.columns([5.5, 1.5])
        items_per_row = 4
    else:  # sidebar view
        items_per_row = 1

    for idx, shot in enumerate(shot_list):
        timing_list: List[InternalFrameTimingObject] = shot.timing_list
        if idx % items_per_row == 0:
            if view == 'main':
                grid = st.columns(items_per_row)
            else:
                grid = [st.container()]  # Use the sidebar as the grid for sidebar view

        with grid[idx % items_per_row]:  # Correct indexing for main view
            st.info(f"##### {shot.name}")
            if shot.main_clip and shot.main_clip.location and view == 'main':
                individual_video_display_element(shot.main_clip)
            else:
                # Ensure a minimum of 4 columns for images
                num_columns = max(len(timing_list), 4)
                grid_timing = st.columns(num_columns)
                for j, timing in enumerate(timing_list):
                    with grid_timing[j]:
                        if timing.primary_image and timing.primary_image.location:
                            st.image(timing.primary_image.location, use_column_width=True)
                # Fill remaining columns if any
                for j in range(len(timing_list), num_columns):
                    with grid_timing[j]:
                        st.empty()

            if view == 'main':
                switch1, switch2 = st.columns([1, 1])
                with switch1:
                    shot_adjustment_button(shot)
                with switch2:
                    shot_animation_button(shot)

                with st.expander("Details & settings:", expanded=False):
                    update_shot_name(shot.uuid)
                    move_shot_buttons(shot, "side")
                    delete_shot_button(shot.uuid)
                    duplicate_shot_button(shot.uuid, position="timeline_view")
                    if shot.main_clip:
                        create_video_download_button(shot.main_clip.location, tag="main_clip")

        if (idx + 1) % items_per_row == 0 or idx == len(shot_list) - 1:
            st.markdown("***")
        if idx == len(shot_list) - 1:
            with grid[(idx + 1) % items_per_row]:
                st.markdown("### Add new shot")
                add_new_shot_element(shot, data_repo)

def add_new_shot_element(shot, data_repo):
    new_shot_name = st.text_input("Shot Name:", max_chars=25)

    if st.button("Add new shot", type="primary", key=f"add_shot_btn_{shot.uuid}"):
        new_shot = add_new_shot(shot.project.uuid)
        if new_shot_name != "":
            data_repo.update_shot(uuid=new_shot.uuid, name=new_shot_name)
        st.rerun()
