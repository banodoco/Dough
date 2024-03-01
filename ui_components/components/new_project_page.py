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
    
    # title
    st.markdown("#### New Project")
    st.markdown("***")
    # Define multicolumn layout
    project_column, _ = st.columns([1,3])
    
    # Prompt user for project naming within project_column
    with project_column:
        new_project_name = st.text_input("Project name:", value="")        

    # Prompt user for video dimension specifications
    v1, v2, v3 = st.columns([6,3,12])
        
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

    with v1:
        audio = st.radio("Audio:", ["No audio", "Attach new audio"], key="audio", horizontal=True)

        # Display audio upload option if user selects "Attach new audio"
        if audio == "Attach new audio":
      
            uploaded_audio = st.file_uploader("Choose an audio file:")
        
        else:
            uploaded_audio = None

    st.write("")

    if st.button("Create New Project"):
        # Add checks for project name existence and format
        if not new_project_name:
            st.error("Please enter a project name.")
        else:
            current_user = data_repo.get_first_active_user()
            new_project, shot = create_new_project(current_user, new_project_name, width, height)
            new_timing = create_frame_inside_shot(shot.uuid, 0)
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

    st.markdown("***")