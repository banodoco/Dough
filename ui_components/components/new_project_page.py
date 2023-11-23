import streamlit as st
from banodoco_settings import create_new_project
from ui_components.methods.common_methods import save_audio_file,create_frame_inside_shot, save_and_promote_image
from utils.common_utils import get_current_user_uuid, reset_project_state
from utils.data_repo.data_repo import DataRepo
import time
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
    v1, v2, v3 = st.columns([4,1,7])
        
    frame_sizes = ["512x512", "768x512", "512x768"]
    with v1:
        frame_size = st.radio("Select frame size:", options=frame_sizes, key="frame_size",horizontal=True)
        if frame_size == "512x512":
            width = 512
            height = 512
        elif frame_size == "768x512":
            width = 768
            height = 512
        elif frame_size == "512x768":
            width = 512
            height = 768
    with v2:        
        img = Image.new('RGB', (width, height), color = (73, 109, 137))
        st.image(img, use_column_width=True)        
        # st.info("Uploaded images will be resized to the selected dimensions.")


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

            new_project, shot = create_new_project(current_user, new_project_name, width, height)
            new_timing = create_frame_inside_shot(shot.uuid, 0)
            
            if starting_image:
                try:
                    save_and_promote_image(starting_image, shot.uuid, new_timing.uuid, "source")
                    save_and_promote_image(starting_image, shot.uuid, new_timing.uuid, "styled")
                except Exception as e:
                    st.error(f"Failed to save the uploaded image due to {str(e)}")

            # remvoing the initial frame which moved to the 1st position 
            # (since creating new project also creates a frame)
            shot = data_repo.get_shot_from_number(new_project.uuid, 1)
            initial_frame = data_repo.get_timing_from_frame_number(shot.uuid, 0)
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
            st.session_state["main_view_type"] = "Creative Process"
            st.session_state['app_settings'] = 0 
            st.success("Project created successfully!")
            time.sleep(1)     
            st.rerun()