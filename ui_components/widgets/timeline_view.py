from io import BytesIO
import os
import shutil
from typing import List
import uuid
from zipfile import ZipFile
import time
from ui_components.methods.file_methods import save_or_host_file
from ui_components.methods.file_methods import add_file_to_shortlist
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
from utils.state_refresh import refresh_app


def timeline_view(shot_uuid, stage, view="sidebar"):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    shot_list = data_repo.get_shot_list(shot.project.uuid)
    project_uuid = shot.project.uuid

    if view == "main":
        _, header_col_2 = st.columns([5.5, 1.5])
        items_per_row = 4
        shot_list_for_display = shot_list
    else:  # sidebar view
        items_per_row = 4  # Changed to 4 per row for sidebar view
        shot_list_for_display = shot_list[::-1]  # Set shot_list_for_display into a reverse
        # End of  Selection

    # Pagination setup for sidebar view
    if view == "sidebar":
        shot_list = data_repo.get_shot_list(project_uuid)

        # Initialize selected images list in session state if not present
        if "selected_images" not in st.session_state:
            st.session_state["selected_images"] = []

        if len(st.session_state["selected_images"]) == 0:
            st.info("Select images on the right to add them to a shot or shortlist.")
        else:
            # Display selected images and provide action buttons
            h1, h2, h3 = st.columns([2, 1, 1])
            with h1:
                if len(st.session_state["selected_images"]) == 1:
                    st.info(f"You have selected {len(st.session_state['selected_images'])} image.")
                else:
                    st.info(f"You have selected {len(st.session_state['selected_images'])} images.")
            with h3:
                if st.button("Clear all selected"):
                    st.session_state["selected_images"] = []
                    refresh_app()
            with h2:
                if st.button("Add all to shortlist"):
                    for uuid in st.session_state["selected_images"]:
                        add_file_to_shortlist(uuid)
                    time.sleep(0.3)
                    st.session_state["selected_images"] = []  # Clear selected images after adding
                    refresh_app()

        add_new_shot_element(shot, data_repo)

        st.markdown("***")

        # Add search bar
        search_query = st_memory.text_input("Search shots:", key=f"shot_search_bar_{project_uuid}")

        # Filter shots based on search query
        if search_query:
            shot_list_for_display = [shot for shot in shot_list if search_query.lower() in shot.name.lower()]
        else:
            shot_list_for_display = shot_list[::-1]  # Reverse the list if no search query

        # Update shot_names after filtering
        shot_names = [s.name for s in shot_list_for_display]
        shot_names.append("**Create New Shot**")

        total_pages = (len(shot_list_for_display) + items_per_row - 1) // items_per_row
        if total_pages > 1:
            page = st_memory.radio(
                "Select Page:",
                list(range(1, total_pages + 1)),
                horizontal=True,
                key=f"page_selector_{project_uuid}",
            )
        else:
            page = 1

        start_index = (page - 1) * items_per_row
        end_index = min(start_index + items_per_row, len(shot_list_for_display))

        # Slice the list for display on the current page
        shot_list_for_display = shot_list_for_display[start_index:end_index]

    for idx, shot in enumerate(shot_list_for_display):
        timing_list: List[InternalFrameTimingObject] = shot.timing_list
        if idx % items_per_row == 0:
            if view == "main":
                grid = st.columns(items_per_row)
            else:
                # Ensure grid is only as large as the number of shots in the last segment
                grid = [st.container() for _ in range(min(items_per_row, len(shot_list_for_display) - idx))]

        with grid[idx % items_per_row]:
            st.info(f"##### {shot.name}")

            num_columns = 4  # Set to 4 images per row regardless of the number of images
            if timing_list:
                if view == "sidebar":
                    rows = (len(timing_list) + num_columns - 1) // num_columns  # Calculate needed rows
                    cols = num_columns
                    grid_timing = [st.columns(cols) for _ in range(rows)]
                    for i, timing in enumerate(timing_list):
                        row = i // cols
                        col = i % cols
                        with grid_timing[row][col]:
                            if timing.primary_image and timing.primary_image.location:
                                st.image(timing.primary_image.location, use_column_width=True)
                else:  # main view
                    max_images = 11  # Reduced by 1 to make room for the "+" indicator
                    rows = 3
                    cols = 4
                    grid_timing = [st.columns(cols) for _ in range(rows)]
                    for i in range(min(max_images, len(timing_list))):
                        row = i // cols
                        col = i % cols
                        with grid_timing[row][col]:
                            timing = timing_list[i]
                            if timing.primary_image and timing.primary_image.location:
                                st.image(timing.primary_image.location, use_column_width=True)

                    # Add "+ more" info on the last column of the last row
                    with grid_timing[-1][-1]:  # Last column of the last row
                        if len(timing_list) > max_images:
                            st.info(f"\+ {len(timing_list) - max_images}", icon=None)
                        elif len(timing_list) == max_images:
                            st.info("+", icon=None)
            else:
                st.warning("No images in shot.")  # Warning if no images are present

            switch1, switch2 = st.columns([1, 1])
            with switch1:
                shot_adjustment_button(shot)
            with switch2:
                shot_animation_button(shot)
            if view == "main":
                with st.expander("Details & settings:", expanded=False):
                    update_shot_name(shot.uuid)
                    move_shot_buttons(shot, "side")
                    delete_shot_button(shot.uuid)
                    duplicate_shot_button(shot.uuid, position="timeline_view")

            elif view == "sidebar":
                if st.session_state["selected_images"]:

                    def add_selected_images_to_shot(shot, shot_list, data_repo):
                        shot_names = [s.name for s in shot_list]
                        shot_number = shot_names.index(shot.name)
                        st.session_state["last_shot_number"] = shot_number
                        for uuid in st.session_state["selected_images"]:
                            image = data_repo.get_file_from_uuid(uuid).location
                            if image:
                                add_key_frame(
                                    image,
                                    shot.uuid,
                                    len(data_repo.get_timing_list_from_shot(shot.uuid)),
                                    refresh_state=False,
                                )
                        st.session_state["selected_images"] = []  # Clear selected images after adding
                        refresh_app()

                    st.button(
                        f"Add {len(st.session_state['selected_images'])} selected images to this shot",
                        use_container_width=True,
                        type="primary",
                        key=f"add_to_shot_{shot.uuid}",
                        on_click=add_selected_images_to_shot,
                        args=(shot, shot_list, data_repo),
                    )
            st.markdown("***")

        if (idx + 1) % items_per_row == 0 or idx == len(shot_list_for_display) - 1:
            st.markdown("***")

        if view == "main" and idx == len(shot_list_for_display) - 1:
            with grid[(idx + 1) % items_per_row]:
                st.markdown("###### Add new shot")
                add_new_shot_element(shot, data_repo, show_image_uploader=True)


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

    if st.button("Add new shot", type="secondary", key=f"add_shot_btn_{shot.uuid}"):
        st.session_state["auto_refresh"] = False
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
        st.session_state["auto_refresh"] = True
        refresh_app()
