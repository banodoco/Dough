import json
import time
from typing import List
import os
import zipfile
import shutil
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
from shared.constants import AppSubPage, InferenceParamType
from ui_components.constants import WorkflowStageType
from ui_components.methods.file_methods import generate_pil_image
from streamlit_option_menu import option_menu
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from ui_components.widgets.add_key_frame_element import add_key_frame,add_key_frame_section
from ui_components.widgets.common_element import duplicate_shot_button
from ui_components.widgets.frame_movement_widgets import change_frame_shot, delete_frame_button, jump_to_single_frame_view_button, move_frame_back_button, move_frame_forward_button, replace_image_widget
from utils.common_utils import refresh_app
from utils.data_repo.data_repo import DataRepo
from utils import st_memory

def shot_keyframe_element(shot_uuid, items_per_row, column=None,position="Timeline",**kwargs):
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    
    if "open_shot" not in st.session_state:
        st.session_state["open_shot"] = None
    

    timing_list: List[InternalFrameTimingObject] = shot.timing_list
                
    if position == "Timeline":

        header_col_0, header_col_1, header_col_2, header_col_3 = st.columns([2,1,1.5,0.5])
            
        with header_col_0:
            update_shot_name(shot.uuid)                 
                           
        # with header_col_1:   
        #     update_shot_duration(shot.uuid)

        with header_col_2:
            st.write("")
            shot_adjustment_button(shot, show_label=True)
        with header_col_3:           
            st.write("")                     
            shot_animation_button(shot, show_label=True)   

    else:
        with column:
            col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

            with col1:
                delete_frames_toggle = st_memory.toggle("Delete Frames", value=True, key="delete_frames_toggle")
            with col2:
                copy_frame_toggle = st_memory.toggle("Copy Frame", value=True, key="copy_frame_toggle")
            with col3:
                move_frames_toggle = st_memory.toggle("Move Frames", value=True, key="move_frames_toggle")
            with col4:
                change_shot_toggle = st_memory.toggle("Change Shot", value=False, key="change_shot_toggle")                
            with col5:
                shift_frame_toggle = st_memory.toggle("Shift Frames", value=False, key="shift_frame_toggle")
                                            
    st.markdown("***")

    for i in range(0, len(timing_list) + 1, items_per_row):
        with st.container():
            grid = st.columns(items_per_row)
            for j in range(items_per_row):
                idx = i + j
                if idx <= len(timing_list):
                    with grid[j]:
                        if idx == len(timing_list):
                            if position != "Timeline":

                                st.info("**Add new frame(s) to shot**")
                                add_key_frame_section(shot_uuid, False)                           
             
                        else:
                            timing = timing_list[idx]
                            if timing.primary_image and timing.primary_image.location:
                                st.image(timing.primary_image.location, use_column_width=True)
                            else:                        
                                st.warning("No primary image present.")       
                                jump_to_single_frame_view_button(idx + 1, timing_list, f"jump_to_{idx + 1}",uuid=shot.uuid)
                            if position != "Timeline":
                                timeline_view_buttons(idx, shot_uuid, copy_frame_toggle, move_frames_toggle,delete_frames_toggle, change_shot_toggle, shift_frame_toggle)
            if (i < len(timing_list) - 1) or (st.session_state["open_shot"] == shot.uuid) or (len(timing_list) % items_per_row != 0 and st.session_state["open_shot"] != shot.uuid) or len(timing_list) % items_per_row == 0:
                st.markdown("***")
    # st.markdown("***")

    if position == "Timeline":      
        # st.markdown("***")      
        bottom1, bottom2, bottom3, bottom4,_ = st.columns([1,1,1,1,2])
        with bottom1:            
            delete_shot_button(shot.uuid)
                            
        with bottom2:            
            duplicate_shot_button(shot.uuid)     
                    
        with bottom3:
            move_shot_buttons(shot, "up")


def move_shot_buttons(shot, direction):
    data_repo = DataRepo()
    move1, move2 = st.columns(2)

    if direction == "side":
        arrow_up = "⬅️"
        arrow_down = "➡️"
    else:  # direction == "up"
        arrow_up = "⬆️"
        arrow_down = "⬇️"

    with move1:
        if st.button(arrow_up, key=f'shot_up_movement_{shot.uuid}', help="This will move the shot up", use_container_width=True):
            if shot.shot_idx > 1:
                data_repo.update_shot(uuid=shot.uuid, shot_idx=shot.shot_idx-1)
            else:
                st.error("This is the first shot.")
                time.sleep(0.3)
            st.rerun()

    with move2:
        if st.button(arrow_down, key=f'shot_down_movement_{shot.uuid}', help="This will move the shot down", use_container_width=True):
            shot_list = data_repo.get_shot_list(shot.project.uuid)
            if shot.shot_idx < len(shot_list):
                data_repo.update_shot(uuid=shot.uuid, shot_idx=shot.shot_idx+1)
            else:
                st.error("This is the last shot.")
                time.sleep(0.3)
            st.rerun()

def download_all_images(shot_uuid):
    #@peter4piyush, you may neeed to do this in a different way to interact properly with the db etc.
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = shot.timing_list

    # Create a directory for the images
    if not os.path.exists(shot.uuid):
        os.makedirs(shot.uuid)

    # Download and save each image
    for idx, timing in enumerate(timing_list):
        if timing.primary_image and timing.primary_image.location:
            location = timing.primary_image.location
            if location.startswith('http'):
                # Remote image
                response = requests.get(location)
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(shot.uuid, f"{idx}.png"))
            else:
                # Local image
                shutil.copy(location, os.path.join(shot.uuid, f"{idx}.png"))

    # Create a zip file
    with zipfile.ZipFile(f"{shot.uuid}.zip", "w") as zipf:
        for file in os.listdir(shot.uuid):
            zipf.write(os.path.join(shot.uuid, file), arcname=file)

    # Read the zip file in binary mode
    with open(f"{shot.uuid}.zip", "rb") as file:
        data = file.read()

    # Delete the directory and zip file
    for file in os.listdir(shot.uuid):
        os.remove(os.path.join(shot.uuid, file))
    os.rmdir(shot.uuid)
    os.remove(f"{shot.uuid}.zip")

    return data

def delete_shot_button(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    confirm_delete = st.checkbox("Confirm deletion",key=f"confirm_delete_{shot.uuid}") 
    help_text = "Check the box above to enable the delete button." if not confirm_delete else "This will this shot and all the frames and videos within."
    if st.button("Delete shot", disabled=(not confirm_delete), help=help_text, key=f"delete_btn_{shot.uuid}", use_container_width=True):
        if st.session_state['shot_uuid'] == str(shot.uuid):
            shot_list = data_repo.get_shot_list(shot.project.uuid)
            for s in shot_list:
                if str(s.uuid) != shot.uuid:
                    st.session_state['shot_uuid'] = s.uuid
        
        data_repo.delete_shot(shot.uuid)
        st.success("Shot deleted successfully")
        time.sleep(0.3)
        st.rerun()

def update_shot_name(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    name = st.text_input("Name:", value=shot.name, max_chars=25, key=f"shot_name_{shot_uuid}")
    if name != shot.name:
        data_repo.update_shot(uuid=shot.uuid, name=name)
        st.session_state['shot_name'] = name
        st.success("Name updated!")
        time.sleep(0.3)
        st.rerun()

def update_shot_duration(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    duration = st.number_input("Duration:", value=shot.duration, key=f"shot_duration_{shot_uuid}")
    if duration != shot.duration:
        data_repo.update_shot(uuid=shot.uuid, duration=duration)
        st.success("Duration updated!")
        time.sleep(0.3)
        st.rerun()

def create_video_download_button(video_location, tag="temp"):
    # Extract the file name from the video location
    file_name = os.path.basename(video_location)

    if video_location.startswith('http'):  # cloud file
        response = requests.get(video_location)
        st.download_button(
            label="Download video",
            data=response.content,
            file_name=file_name,
            mime='video/mp4',
            key=tag + str(file_name),
            use_container_width=True
        )
    else:  # local file
        with open(video_location, 'rb') as file:
            st.download_button(
                label="Download video",
                data=file,
                file_name=file_name,
                mime='video/mp4',
                key=tag + str(file_name),
                use_container_width=True
            )

def shot_adjustment_button(shot, show_label=False):
    button_label = "Shot Adjustment 🔧" if show_label else "🔧"
    if st.button(button_label, key=f"jump_to_shot_adjustment_{shot.uuid}", help=f"Adjust '{shot.name}'", use_container_width=True):
        st.session_state["shot_uuid"] = shot.uuid
        st.session_state['current_frame_sidebar_selector'] = 0
        st.session_state['current_subpage'] = AppSubPage.ADJUST_SHOT.value
        st.session_state['selected_page_idx'] = 1
        st.session_state['shot_view_index'] = 1  
        st.rerun() 

def shot_animation_button(shot, show_label=False):
    button_label = "Shot Animation 🎞️" if show_label else "🎞️"
    if st.button(button_label, key=f"jump_to_shot_animation_{shot.uuid}", help=f"Animate '{shot.name}'", use_container_width=True):
        st.session_state["shot_uuid"] = shot.uuid
        st.session_state['current_subpage'] = AppSubPage.ANIMATE_SHOT.value
        st.session_state['selected_page_idx'] = 2
        st.session_state['shot_view_index'] = 0
        st.rerun() 



def shift_frame_to_position(timing_uuid, target_position):
    '''
    Shifts the frame to the specified target position within the list of frames.
    
    Note: target_position is expected to be 1-based for user convenience (e.g., position 1 is the first position).
    '''
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_list = data_repo.get_timing_list_from_shot(timing.shot.uuid)

    # Adjusting target_position to 0-based indexing for internal logic
    target_position -= 1

    current_position = timing.aux_frame_index
    total_frames = len(timing_list)

    # Check if the target position is valid
    if target_position < 0 or target_position >= total_frames:
        st.error("Invalid target position")
        time.sleep(0.5)
        return
    
    # Check if the frame is already at the target position
    if current_position == target_position:
        st.error("That's already your position")
        time.sleep(0.5)
        return

    # Update the position of the current frame
    data_repo.update_specific_timing(timing.uuid, aux_frame_index=target_position)


def shift_frame_button(idx,shot):
    timing_list: List[InternalFrameTimingObject] = shot.timing_list
    col1, col2 = st.columns([1,1])
    with col1:
        position_to_shift_to = st.number_input("Shift to position:", value=timing_list[idx].aux_frame_index+1, key=f"shift_to_position_{timing_list[idx].uuid}",min_value=1, max_value=len(timing_list))
    with col2:
        st.write("")
        if st.button("Shift", key=f"shift_frame_{timing_list[idx].uuid}", use_container_width=True):
            shift_frame_to_position(timing_list[idx].uuid, position_to_shift_to)
            st.rerun()
            

def timeline_view_buttons(idx, shot_uuid, copy_frame_toggle, move_frames_toggle, delete_frames_toggle, change_shot_toggle, shift_frame_toggle):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = shot.timing_list

    
    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
    
    if move_frames_toggle:
        with btn1:                                            
            move_frame_back_button(timing_list[idx].uuid, "side-to-side")
        with btn2:   
            move_frame_forward_button(timing_list[idx].uuid, "side-to-side")
    
    if copy_frame_toggle:
        with btn3:
            if st.button("🔁", key=f"copy_frame_{timing_list[idx].uuid}", use_container_width=True):
                pil_image = generate_pil_image(timing_list[idx].primary_image.location)
                add_key_frame(pil_image, False, st.session_state['shot_uuid'], timing_list[idx].aux_frame_index+1, refresh_state=False)
                refresh_app(maintain_state=True)

    if delete_frames_toggle:
        with btn4:
            delete_frame_button(timing_list[idx].uuid)
    
    if change_shot_toggle:
        change_frame_shot(timing_list[idx].uuid, "side-to-side")
    
    jump_to_single_frame_view_button(idx + 1, timing_list, 'timeline_btn_'+str(timing_list[idx].uuid))        

    if shift_frame_toggle:
        shift_frame_button(idx,shot)
