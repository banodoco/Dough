from io import BytesIO
import io
import tempfile
import uuid
import requests
import streamlit as st
from banodoco_settings import create_new_project
from shared.constants import SERVER, InternalFileType, ServerType
from ui_components.constants import AUDIO_FILE
from ui_components.models import InternalFileObject
from ui_components.common_methods import save_audio_file,create_timings_row_at_frame_number, save_uploaded_image
from utils.common_utils import create_working_assets, get_current_user, get_current_user_uuid, save_or_host_file, save_or_host_file_bytes
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.video import resize_video
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import os
from PIL import Image

import utils.local_storage.local_storage as local_storage


def new_project_page():

    data_repo = DataRepo()

    a1, a2 = st.columns(2)
    with a1:
        new_project_name = st.text_input("Project name:", value="")
    with a2:
        st.write("")
    
    img1, img2, img3 = st.columns([3, 1.5, 1])

    with img1:    
        starting_image = st.file_uploader("Choose a starting image:", key="starting_image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

    with img2:
        if starting_image is not None:
            image = Image.open(starting_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)            
            width, height = image.size    
            st.write("")
            
    with img3:  
        if starting_image is not None:
            st.success(f"The dimensions of the image are {width} x {height}")      
    
    b1, b2, b3 = st.columns(3)

    frame_sizes = ["512", "704", "768", "896", "1024"]
    
    with b1:
        width = int(st.selectbox("Select video width:", options=frame_sizes, key="video_width"))
    with b2:
        height = int(st.selectbox("Select video height:", options=frame_sizes, key="video_height"))
    with b3:
        st.info("Uploaded images will be resized to the selected dimensions.")

    audio_options = ["No audio", "Attach new audio"]
    
    audio = st.radio("Audio:", audio_options, key="audio", horizontal=True)

    if audio == "Attach new audio":
        d1, d2 = st.columns([4, 5])
        with d1:
            uploaded_audio = st.file_uploader("Choose a audio file:")
        with d2:
            st.write("")
            st.write("")
            st.info(
                "Make sure that this audio is around the same length as your video.")
    else:
        uploaded_audio = None

    st.write("")

    if st.button("Create New Project"):
        if not new_project_name:
            st.error("Please enter a project name")
        else:
            new_project_name = new_project_name.replace(" ", "_")
            current_user = data_repo.get_first_active_user()            
            new_project = create_new_project(current_user, new_project_name, width, height, "Images", "Interpolation")
            new_timing = create_timings_row_at_frame_number(new_project.uuid, 0)
            if starting_image:
                save_uploaded_image(data_repo, starting_image, new_project.uuid, new_timing.uuid, "source")
                save_uploaded_image(data_repo, starting_image, new_project.uuid, new_timing.uuid, "styled")
            if uploaded_audio:
                if save_audio_file(data_repo, uploaded_audio, new_project.uuid):
                    st.success("Audio file saved and attached successfully.")
                else:
                    st.error("Failed to save and attach the audio file.")            
            st.session_state["project_uuid"] = new_project.uuid
            video_list = data_repo.get_all_file_list(file_type=InternalFileType.VIDEO.value)  #[f for f in os.listdir("videos") if not f.startswith('.')]                        
            st.session_state["index_of_project_name"] = len(video_list) - 1
            st.session_state["section"] = "Open Project"   
            st.session_state['change_section'] = True      
            st.success("Project created successfully!")
            time.sleep(1)   
            st.experimental_rerun()
