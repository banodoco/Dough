import shutil
import streamlit as st
import os
import time
from ui_components.widgets.attach_audio_element import attach_audio_element
from PIL import Image

from utils.common_utils import get_current_user_uuid
from utils.data_repo.data_repo import DataRepo
from utils.state_refresh import refresh_app


def project_settings_page(project_uuid):
    data_repo = DataRepo()
    st.markdown("#### Project Settings")
    st.markdown("***")

    with st.expander("📋 Project name", expanded=True):
        project = data_repo.get_project_from_uuid(project_uuid)
        new_name = st.text_input("Enter new name:", project.name)
        if st.button("Save", key="project_name"):
            data_repo.update_project(uuid=project_uuid, name=new_name)
            refresh_app()
    project_settings = data_repo.get_project_setting(project_uuid)

    frame_sizes = ["512x512", "768x512", "512x768", "512x896", "896x512", "512x1024", "1024x512"]
    current_size = f"{project_settings.width}x{project_settings.height}"
    current_index = frame_sizes.index(current_size) if current_size in frame_sizes else 0

    with st.expander("🖼️ Frame Size", expanded=True):

        v1, v2, v3 = st.columns([4, 4, 2])
        with v1:
            st.write("Current Size = ", project_settings.width, "x", project_settings.height)

            custom_frame_size = st.checkbox("Enter custom frame size", value=False)
            err = False
            if not custom_frame_size:
                frame_size = st.radio(
                    "Select frame size:",
                    options=frame_sizes,
                    index=current_index,
                    key="frame_size",
                    horizontal=True,
                )
                width, height = map(int, frame_size.split("x"))
            else:
                st.info(
                    "This is an experimental feature. There might be some issues - particularly with image generation."
                )
                width = st.text_input("Width", value=512)
                height = st.text_input("Height", value=512)
                try:
                    width, height = int(width), int(height)
                    err = False
                except Exception as e:
                    st.error("Please input integer values")
                    err = True

            if not err:
                img = Image.new("RGB", (width, height), color=(73, 109, 137))
                st.image(img, width=70)

                if st.button("Save"):
                    st.success("Frame size updated successfully")
                    time.sleep(0.3)
                    data_repo.update_project_setting(project_uuid, width=width)
                    data_repo.update_project_setting(project_uuid, height=height)
                    refresh_app()

    st.write("")
    st.write("")
    st.write("")
    delete_proj = st.checkbox("I confirm to delete this project entirely", value=False)
    if st.button("Delete Project", disabled=(not delete_proj)):
        project_list = data_repo.get_all_project_list(user_id=get_current_user_uuid())
        if project_list and len(project_list) > 1:
            data_repo.update_project(uuid=project_uuid, is_disabled=True)
            st.success("Project deleted successfully")
            st.session_state["index_of_project_name"] = 0
        else:
            st.error("You can't delete the only available project")

        time.sleep(0.7)
        refresh_app()
