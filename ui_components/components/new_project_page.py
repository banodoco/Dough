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
from utils.common_utils import create_working_assets, get_current_user, get_current_user_uuid, reset_project_state, save_or_host_file, save_or_host_file_bytes
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.video import resize_video
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import os
from PIL import Image

import utils.local_storage.local_storage as local_storage


def new_project_page():

    # Initialize data repository
    data_repo = DataRepo()
    
    # Define multicolumn layout
    project_column, filler_column = st.columns(2)
    
    # Prompt user for project naming within project_column
    with project_column:
        new_project_name = st.text_input("Project name:", value="")

    # Define multicolumn layout for images
    image_column, image_display_column, img_info_column = st.columns([3, 1.5, 1])

    # Prompt user for starting image within image_column
    with image_column:    
        starting_image = st.file_uploader("Choose a starting image:", key="starting_image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
    
    # Display starting image within image_display_column if available
    with image_display_column:
        if starting_image is not None: 
            try:
                image = Image.open(starting_image)
                st.image(image, caption='Uploaded Image.', use_column_width=True)            
                img_width, img_height = image.size      
            except Exception as e:
                st.error(f"Failed to open the image due to {str(e)}")
    # Display image information within img_info_column if starting image exists            
    with img_info_column:  
        if starting_image is not None:
            st.success(f"The dimensions of the image are {img_width} x {img_height}")

    # Prompt user for video dimension specifications
    video_width_column, video_height_column, video_info_column = st.columns(3)

    frame_sizes = ["512", "704", "768", "896", "1024"]
    with video_width_column:
        width = int(st.selectbox("Select video width:", options=frame_sizes, key="video_width"))
    with video_height_column:
        height = int(st.selectbox("Select video height:", options=frame_sizes, key="video_height"))
    with video_info_column:
        st.info("Uploaded images will be resized to the selected dimensions.")
    
    # Prompt user for audio preferences
    audio = st.radio("Audio:", ["No audio", "Attach new audio"], key="audio", horizontal=True)

    # Display audio upload option if user selects "Attach new audio"
    if audio == "Attach new audio":
        audio_upload_column, audio_info_column = st.columns([4, 5])
        with audio_upload_column:
            uploaded_audio = st.file_uploader("Choose an audio file:")
        with audio_info_column:
            st.write("")
            st.write("")
            st.info("Make sure that this audio is around the same length as your video.")
    else:
        uploaded_audio = None

    st.write("")

    if st.button("Create New Project"):
        # Add checks for project name existence and format
        if not new_project_name:
            st.error("Please enter a project name.")
        else:
            new_project_name = new_project_name.replace(" ", "_")
            current_user = data_repo.get_first_active_user()

            try:
                new_project = create_new_project(current_user, new_project_name, width, height, "Images", "Interpolation")
                new_timing = create_timings_row_at_frame_number(new_project.uuid, 0)
            except Exception as e:
                st.error(f"Failed to create the new project due to {str(e)}")

            if starting_image:
                try:
                    save_uploaded_image(starting_image, new_project.uuid, new_timing.uuid, "source")
                    save_uploaded_image(starting_image, new_project.uuid, new_timing.uuid, "styled")
                except Exception as e:
                    st.error(f"Failed to save the uploaded image due to {str(e)}")

            # remvoing the initial frame which moved to the 1st position
            initial_frame = data_repo.get_timing_from_frame_number(new_project.uuid, 1)
            data_repo.delete_timing_from_uuid(initial_frame.uuid)
            
            if uploaded_audio:
                try:
                    if save_audio_file(uploaded_audio, new_project.uuid):
                        st.success("Audio file saved and attached successfully.")
                    else:
                        st.error("Failed to save and attach the audio file.")  
                except Exception as e:
                    st.error(f"Failed to save the uploaded audio due to {str(e)}")

            reset_project_state()

            st.session_state["project_uuid"] = new_project.uuid
            project_list = data_repo.get_all_project_list(user_id=get_current_user_uuid())            
            st.session_state["index_of_project_name"] = len(project_list) - 1
            st.session_state["section"] = "Open Project"
            st.session_state['change_section'] = True 
            st.success("Project created successfully!")
            time.sleep(1)     
            st.experimental_rerun()