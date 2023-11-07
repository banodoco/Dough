import shutil
import streamlit as st
import os
import time
from ui_components.widgets.attach_audio_element import attach_audio_element

from utils.data_repo.data_repo import DataRepo


def project_settings_page(project_uuid):
    data_repo = DataRepo()

    project_settings = data_repo.get_project_setting(project_uuid)
    attach_audio_element(project_uuid, True)

    with st.expander("Frame Size", expanded=True):
        st.write("Current Size = ",
                 project_settings.width, "x", project_settings.height)
        width = st.selectbox("Select video width", options=[
                             "512", "683", "704", "1024"], key="video_width")
        height = st.selectbox("Select video height", options=[
                              "512", "704", "1024"], key="video_height")
        if st.button("Save"):
            data_repo.update_project_setting(project_uuid, width=width)
            data_repo.update_project_setting(project_uuid, height=height)
            st.rerun()
