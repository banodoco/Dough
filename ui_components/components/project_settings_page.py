import shutil
import streamlit as st
import os
import time
from ui_components.widgets.attach_audio_element import attach_audio_element
from PIL import Image

from utils.data_repo.data_repo import DataRepo


def project_settings_page(project_uuid):
    data_repo = DataRepo()
    st.subheader("Project Settings")
    project_settings = data_repo.get_project_setting(project_uuid)

    

    frame_sizes = ["512x512", "768x512", "512x768"]
    current_size = f"{project_settings.width}x{project_settings.height}"
    current_index = frame_sizes.index(current_size) if current_size in frame_sizes else 0

    with st.expander("Frame Size", expanded=True):
        v1, v2, v3 = st.columns([4, 4, 2])
        with v1:
            st.write("Current Size = ", project_settings.width, "x", project_settings.height)
            
            frame_size = st.radio("Select frame size:", options=frame_sizes, index=current_index, key="frame_size", horizontal=True)
            width, height = map(int, frame_size.split('x'))
            
              
            img = Image.new('RGB', (width, height), color = (73, 109, 137))
            st.image(img, width=70)

            if st.button("Save"):
                data_repo.update_project_setting(project_uuid, width=width)
                data_repo.update_project_setting(project_uuid, height=height)
                st.experimental_rerun()      
        
    st.write("")
    attach_audio_element(project_uuid, True)