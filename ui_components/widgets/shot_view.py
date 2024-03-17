import base64
import json
import time
from typing import List
import os
import zipfile
import shutil
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import streamlit as st
import uuid
import random
from shared.constants import AppSubPage, InferenceParamType
from ui_components.constants import WorkflowStageType
from ui_components.methods.file_methods import generate_pil_image, get_file_bytes_and_extension, get_file_size
from streamlit_option_menu import option_menu
from shared.constants import InternalFileType
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from ui_components.widgets.add_key_frame_element import add_key_frame,add_key_frame_section
from ui_components.widgets.common_element import duplicate_shot_button
from ui_components.methods.common_methods import apply_coord_transformations, apply_image_transformations
from ui_components.widgets.frame_movement_widgets import change_frame_shot, delete_frame_button, jump_to_single_frame_view_button, move_frame_back_button, move_frame_forward_button, replace_image_widget,delete_frame
from utils.common_utils import refresh_app
from utils.data_repo.data_repo import DataRepo
from ui_components.methods.file_methods import save_or_host_file
from utils import st_memory
from ui_components.widgets.image_zoom_widgets import reset_zoom_element, save_zoomed_image, zoom_inputs

def shot_keyframe_element(shot_uuid, items_per_row, column=None, position="Timeline", **kwargs):
    from ui_components.widgets.variant_comparison_grid import image_variant_details
    
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    project_uuid = shot.project.uuid

    timing_list: List[InternalFrameTimingObject] = shot.timing_list
    with column:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            move_frame_mode = st_memory.toggle("Enter Frame Changer™ mode", value=False, key=f"move_frame_mode_{shot.uuid}")
            if st.session_state[f"move_frame_mode_{shot.uuid}"]:
                st.warning("You are in 'move frame' mode. You have to press 'Save' below to save the changes.")
                if st.button("Save", key=f"save_move_frame_{shot.uuid}", help="Save the changes made in 'move frame' mode", use_container_width=True):                                        
                    update_shot_frames(shot_uuid, timing_list)
                    st.rerun()
                if f"shot_data_{shot_uuid}" not in st.session_state:
                    st.session_state[f"shot_data_{shot_uuid}"] = None
                if st.session_state[f"shot_data_{shot_uuid}"] is None:
                    shot_data = [{
                        "uuid": timing.uuid,
                        "image_location": timing.primary_image.location if timing.primary_image and timing.primary_image.location else None,
                        "position": idx
                    } for idx, timing in enumerate(timing_list)]
                    st.session_state[f"shot_data_{shot_uuid}"] = pd.DataFrame(shot_data)
            else:
                st.session_state[f"shot_data_{shot_uuid}"] = None

                                                 
    st.markdown("***")
    if move_frame_mode:
        with column:
            if f'list_to_move_{shot.uuid}' not in st.session_state:
                st.session_state[f'list_to_move_{shot.uuid}'] = []
            with col2:
                frame_to_move_to = st.selectbox("Bulk move frames to:", [f"{i + 1}" for i in range(len(timing_list))], key=f"frame_to_move_to_{shot.uuid}")
            with col3:
                if st.session_state[f"move_frame_mode_{shot.uuid}"]:

                    if st.session_state[f'list_to_move_{shot.uuid}'] == []:
                        st.write("")
                        st.info("No frames selected to move. Select them below.")
                    else:                
                        st.info(f"Selected frames to move: {st.session_state[f'list_to_move_{shot.uuid}']}")
                if st.session_state[f'list_to_move_{shot.uuid}'] != []:
                    if st.button("Remove all selected", key=f"remove_all_selected_{shot.uuid}", help="Remove all selected frames to move"):
                        st.session_state[f'list_to_move_{shot.uuid}'] = []
                        st.rerun()
                    
                    
            with col2:
                if st.session_state[f'list_to_move_{shot.uuid}'] != []:
                    if st.button("Move selected", key=f"move_frame_to_{shot.uuid}", help="Move the frame to the selected position", use_container_width=True):
                        # order list to move in ascending order
                        list_to_move = sorted(st.session_state[f'list_to_move_{shot.uuid}'])

                        st.session_state[f"shot_data_{shot_uuid}"] = move_temp_frames_to_positions(st.session_state[f"shot_data_{shot_uuid}"], list_to_move, int(frame_to_move_to)-1)
                        st.session_state[f'list_to_move_{shot.uuid}'] = []
                        st.rerun()
                    if st.button("Delete selected", key=f"delete_frame_to_{shot.uuid}", help="Delete the selected frames"):
                        
                        st.session_state[f"shot_data_{shot_uuid}"] = bulk_delete_temp_frames(st.session_state[f"shot_data_{shot_uuid}"], st.session_state[f'list_to_move_{shot.uuid}'])
                        st.session_state[f'list_to_move_{shot.uuid}'] = []
                        st.rerun()
                else:
                    st.button("Move selected", key=f"move_frame_to_{shot.uuid}", use_container_width=True,disabled=True, help="No frames selected to move.")
                
            
        for i in range(0, len(st.session_state[f"shot_data_{shot_uuid}"]), items_per_row):
            with st.container():
                grid = st.columns(items_per_row)
                for j in range(items_per_row):                    
                    idx = i + j
                    if idx < len(st.session_state[f"shot_data_{shot_uuid}"]):  # Ensure idx does not exceed the length of shot_df
                        
                        with grid[j % items_per_row]:  # Use modulo for column indexingr
                            
                            row = st.session_state[f"shot_data_{shot_uuid}"].loc[idx]
                            
                            if row['image_location']:
                                st.caption(f"Frame {idx + 1}")
                                st.image(row['image_location'], use_column_width=True)
                            else:
                                st.warning("No primary image present.")

                            btn1, btn2, btn3, btn4, btn5 = st.columns([1, 1, 1, 1, 1.25])
                            
                            with btn1:        
                                if st.button("⬅️", key=f"move_frame_back_{idx}", help="Move frame back", use_container_width=True):
                                    st.session_state[f"shot_data_{shot_uuid}"] = move_temp_frame(st.session_state[f"shot_data_{shot_uuid}"], idx, 'backward')           
                                    st.rerun()
                            with btn2:
                                if st.button("➡️", key=f"move_frame_forward_{idx}", help="Move frame forward", use_container_width=True):
                                    st.session_state[f"shot_data_{shot_uuid}"] = move_temp_frame(st.session_state[f"shot_data_{shot_uuid}"], idx, 'forward')
                                    st.rerun()
                            with btn3:
                                if st.button("🔁", key=f"copy_frame_{idx}", use_container_width=True):
                                    st.session_state[f"shot_data_{shot_uuid}"] = copy_temp_frame(st.session_state[f"shot_data_{shot_uuid}"], idx)
                                    st.rerun()
                            with btn4:
                                if st.button("❌", key=f"delete_frame_{idx}", use_container_width=True):
                                    st.session_state[f"shot_data_{shot_uuid}"] = delete_temp_frame(st.session_state[f"shot_data_{shot_uuid}"], idx)
                                    st.rerun()
                            with btn5:
                                if idx not in st.session_state[f'list_to_move_{shot.uuid}']:
                                    if st.button("Select", key=f"select_frame_{idx}", use_container_width=True):
                                        st.session_state[f'list_to_move_{shot.uuid}'].append(idx)
                                        st.rerun()
                                else:
                                    if st.button("Deselect", key=f"deselect_frame_{idx}", use_container_width=True,type='primary'):
                                        st.session_state[f'list_to_move_{shot.uuid}'].remove(idx)
                                        st.rerun()

                            header1, header2 = st.columns([1, 1.5])
                            with header1:
                                st_memory.toggle("Open Zoom", key=f"open_zoom_{shot.uuid}_{idx}", value=False)
                             
                            if st.session_state[f"open_zoom_{shot.uuid}_{idx}"]:
                                with header2:
                                    if st.button("Reset",use_container_width=True):
                                        reset_zoom_element()
                                        st.rerun()

                                # close all other zooms
                                for i in range(0, len(st.session_state[f"shot_data_{shot_uuid}"])):
                                    if i != idx:
                                        st.session_state[f"open_zoom_{shot.uuid}_{i}"] = False

                                input_image = generate_pil_image(st.session_state[f"shot_data_{shot_uuid}"].loc[idx]['image_location'])

                                if 'zoom_level_input' not in st.session_state:
                                    st.session_state['zoom_level_input'] = 100
                                    st.session_state['rotation_angle_input'] = 0
                                    st.session_state['x_shift'] = 0
                                    st.session_state['y_shift'] = 0
                                    st.session_state['flip_vertically'] = False
                                    st.session_state['flip_horizontally'] = False              
                                
                                zoom_inputs(horizontal=True)

                                st.caption("Output Image:")

                                output_image = apply_image_transformations(
                                    input_image,
                                    st.session_state['zoom_level_input'],
                                    st.session_state['rotation_angle_input'],
                                    st.session_state['x_shift'] ,
                                    st.session_state['y_shift'] ,
                                    st.session_state['flip_vertically'] ,
                                    st.session_state['flip_horizontally']
                                )

                                st.image(output_image, use_column_width=True)
                                
                               
                                if st.button("Save", key=f"save_zoom_{idx}", help="Save the changes made in 'move frame' mode",type="primary",use_container_width=True):
                                    # make file_name into a random uuid using uuid
                                    file_name = f"{uuid.uuid4()}.png"
                                    
                                    save_location = f"videos/{project_uuid}/assets/frames/2_character_pipeline_completed/{file_name}"
                                    hosted_url = save_or_host_file(output_image, save_location)
                                    file_data = {
                                        "name": file_name,
                                        "type": InternalFileType.IMAGE.value,
                                        "project_id": project_uuid
                                    }

                                    if hosted_url:
                                        file_data.update({'hosted_url': hosted_url})
                                        location = hosted_url
                                    else:
                                        file_data.update({'local_path': save_location})
                                        location = save_location

                                    st.session_state[f"shot_data_{shot_uuid}"].loc[idx, 'image_location'] = location
                                    st.session_state[f'open_zoom_{shot.uuid}_{idx}'] = False
                                    st.rerun()
                                                                                       
                st.markdown("***")
        bottom1, bottom2 = st.columns([1, 1])
        with bottom1:
            st.warning("You are in 'move frame' mode. You have to press 'Save' below to save the changes.")
            if st.button("Save", key=f"save_move_frame_{shot.uuid}_bottom", help="Save the changes made in 'move frame' mode", use_container_width=True):                
                update_shot_frames(shot_uuid, timing_list)
                st.rerun()
        st.markdown("***")

    else:
        for i in range(0, len(timing_list) + 1, items_per_row):
            with st.container():
                grid = st.columns(items_per_row)
                for j in range(items_per_row):
                    idx = i + j
                    if idx <= len(timing_list):
                        with grid[j]:
                            if idx == len(timing_list):
                                if position != "Timeline":
                                    add_key_frame_section(shot_uuid)
                            else:
                                timing = timing_list[idx]
                                if timing.primary_image and timing.primary_image.location:
                                    st.image(timing.primary_image.location, use_column_width=True)                                                    
                                    jump_to_single_frame_view_button(idx + 1, timing_list, f"jump_to_{idx + 1}",uuid=shot.uuid)
                                else:                        
                                    st.warning("No primary image present.")

                st.markdown("***")


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

    # if get_file_size(video_location) > 5:
    if st.button("Prepare video for download", use_container_width=True, key=tag + str(file_name)):
        file_bytes, file_ext = get_file_bytes_and_extension(video_location)
        # file_bytes = base64.b64encode(file_bytes).decode('utf-8')
        st.download_button(
            label="Download video",
            data=file_bytes,
            file_name=file_name,
            mime='video/mp4',
            key=tag + str(file_name) + "_download_gen",
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




def move_temp_frame(df, current_position, direction):
    if direction == 'forward' and current_position < len(df) - 1:
        df.loc[current_position, 'position'], df.loc[current_position + 1, 'position'] = df.loc[current_position + 1, 'position'], df.loc[current_position, 'position']
    elif direction == 'backward' and current_position > 0:
        df.loc[current_position, 'position'], df.loc[current_position - 1, 'position'] = df.loc[current_position - 1, 'position'], df.loc[current_position, 'position']
    return df.sort_values('position').reset_index(drop=True)

# Function to copy a frame in the dataframe
def copy_temp_frame(df, position_to_copy):
    new_row = df.loc[position_to_copy].copy()
    new_row['uuid'] = f"Copy_of_{new_row['uuid']}"
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # make the position current frame + 1 and all the frames after it + 1
    df.loc[position_to_copy + 1:, 'position'] = df.loc[position_to_copy + 1:, 'position'] + 1
    
    
    return df.sort_values('position').reset_index(drop=True)

# Function to delete a frame in the dataframe
def delete_temp_frame(df, position_to_delete):
    df = df.drop(position_to_delete).reset_index(drop=True)
    df['position'] = range(len(df))
    return df

def bulk_delete_temp_frames(df, positions_to_delete):
    # Ensure positions_to_delete is a list
    if not isinstance(positions_to_delete, list):
        positions_to_delete = [positions_to_delete]
    
    # Drop the rows from their positions
    df = df.drop(positions_to_delete).reset_index(drop=True)
    
    # Correct the 'position' column to reflect the new order
    df['position'] = range(len(df))
    
    return df

def update_shot_frames(shot_uuid, timing_list):
    
    # Ensure the move frame mode is turned off
    st.session_state[f"move_frame_mode_{shot_uuid}"] = False

    # Delete all existing frames first
    for timing in timing_list:
        delete_frame(timing.uuid)

    # Initialize the progress bar
    progress_bar = st.progress(0)

    # Calculate the total number of items for accurate progress update
    total_items = len(st.session_state[f"shot_data_{shot_uuid}"])

    # List of happy emojis
    random_list_of_emojis = ["🎉", "🎊", "🎈", "🎁", "🎀", "🎆", "🎇", "🧨", "🪅"]

    # Add frames again and update progress
    for idx, (index, row) in enumerate(st.session_state[f"shot_data_{shot_uuid}"].iterrows()):
        selected_image_location = row['image_location']
        # def add_key_frame(selected_image: Union[Image.Image, InternalFileObject], shot_uuid, target_frame_position=None, refresh_state=True, update_cur_frame_idx=True):
        add_key_frame(selected_image_location, shot_uuid, refresh_state=False)

        # Update the progress bar
        progress = (idx + 1) / total_items
        random_emoji = random.choice(random_list_of_emojis)
        st.caption(f"Saving frame {idx + 1} of {total_items} {random_emoji}")
        progress_bar.progress(progress)
    
    st.session_state[f"shot_data_{shot_uuid}"] = None



def move_temp_frames_to_positions(df, current_positions, new_start_position):
    # Ensure current_positions is a list
    if not isinstance(current_positions, list):
        current_positions = [current_positions]
    
    # Sort current_positions to handle them in order
    current_positions = sorted(current_positions)
    
    # Extract rows to move
    rows_to_move = df.iloc[current_positions]
    
    # Drop the rows from their current positions
    df = df.drop(df.index[current_positions]).reset_index(drop=True)
    
    # Calculate new positions considering the removals
    new_positions = range(new_start_position, new_start_position + len(rows_to_move))
    
    # Split the DataFrame into parts before and after the new start position
    df_before = df.iloc[:new_start_position]
    df_after = df.iloc[new_start_position:]
    
    # Reassemble the DataFrame with rows inserted at their new positions
    df = pd.concat([df_before, rows_to_move, df_after]).reset_index(drop=True)
    
    # Correct the 'position' column to reflect the new order
    df['position'] = range(len(df))
    
    return df