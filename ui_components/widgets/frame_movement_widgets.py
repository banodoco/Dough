import time
import streamlit as st
from ui_components.methods.common_methods import add_image_variant, promote_image_variant, save_uploaded_image, update_clip_duration_of_all_timing_frames
from ui_components.models import InternalFrameTimingObject
from utils.constants import ImageStage

from utils.data_repo.data_repo import DataRepo

def change_frame_position_input(timing_uuid, src):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)

    min_value = 1
    max_value = len(timing_list)

    new_position = st.number_input("Move to new position:", min_value=min_value, max_value=max_value,
                                   value=timing.aux_frame_index + 1, step=1, key=f"new_position_{timing.uuid}_{src}")
    
    if st.button('Update Position',key=f"change_frame_position_{timing.uuid}_{src}"): 
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=new_position - 1)
        st.rerun()
        

def move_frame(direction, timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    if direction == "Up":
        if timing.aux_frame_index == 0:
            st.error("This is the first frame")       
            time.sleep(1)     
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index - 1)
    elif direction == "Down":
        timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)
        if timing.aux_frame_index == len(timing_list) - 1:
            st.error("This is the last frame")
            time.sleep(1)
            return
        
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=timing.aux_frame_index + 1)

def move_frame_back_button(timing_uuid, orientation):
    direction = "Up"
    if orientation == "side-to-side":
        arrow = "⬅️"        
    else:  # up-down
        arrow = "⬆️"        
    if st.button(arrow, key=f"move_frame_back_{timing_uuid}", help="Move frame back"):
        move_frame(direction, timing_uuid)
        st.rerun()


def move_frame_forward_button(timing_uuid, orientation):
    direction = "Down"
    if orientation == "side-to-side":
        arrow = "➡️"        
    else:  # up-down
        arrow = "⬇️"

    if st.button(arrow, key=f"move_frame_forward_{timing_uuid}", help="Move frame forward"):
        move_frame(direction, timing_uuid)
        st.rerun()


def delete_frame_button(timing_uuid, show_label=False):
    if show_label:
        label = "Delete Frame 🗑️"
    else:
        label = "🗑️"

    if st.button(label, key=f"delete_frame_{timing_uuid}", help="Delete frame"):
        delete_frame(timing_uuid)
        st.rerun()

def delete_frame(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    project_uuid = timing.project.uuid
    next_timing = data_repo.get_next_timing(timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)

    if len(timing_details) == 1:
        st.error("can't delete the only image present in the project")
        return

    if next_timing:
        data_repo.update_specific_timing(
            next_timing.uuid,
            interpolated_clip_list=None,
            preview_video_id=None,
            timed_clip_id=None
        )

    data_repo.delete_timing_from_uuid(timing.uuid)
    timing_details = data_repo.get_timing_list_from_project(project_uuid=project_uuid)
    
    # this is the last frame
    if not next_timing:
        st.session_state['current_frame_index'] = max(1, st.session_state['current_frame_index'] - 1)
        st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
    # this is the first frame or something in the middle
    else:
        st.session_state['current_frame_index'] = min(len(timing_details) - 1, st.session_state['current_frame_index'] + 1)
        st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
    
    update_clip_duration_of_all_timing_frames(project_uuid)

def replace_image_widget(timing_uuid, stage):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(timing.project.uuid)
                                        
    replace_with = st.radio("Replace with:", [
        "Uploaded Frame", "Other Frame"], horizontal=True, key=f"replace_with_what_{stage}")
    

    if replace_with == "Other Frame":
                    
        which_stage_to_use_for_replacement = st.radio("Select stage to use:", [
            ImageStage.MAIN_VARIANT.value, ImageStage.SOURCE_IMAGE.value], key=f"which_stage_to_use_for_replacement_{stage}", horizontal=True)
        which_image_to_use_for_replacement = st.number_input("Select image to use:", min_value=0, max_value=len(
            timing_details)-1, value=0, key=f"which_image_to_use_for_replacement_{stage}")
        
        if which_stage_to_use_for_replacement == ImageStage.SOURCE_IMAGE.value:                                    
            selected_image = timing_details[which_image_to_use_for_replacement].source_image
            
        
        elif which_stage_to_use_for_replacement == ImageStage.MAIN_VARIANT.value:
            selected_image = timing_details[which_image_to_use_for_replacement].primary_image
            
        
        st.image(selected_image.local_path, use_column_width=True)

        if st.button("Replace with selected frame", disabled=False,key=f"replace_with_selected_frame_{stage}"):
            if stage == "source":
                                            
                data_repo.update_specific_timing(timing.uuid, source_image_id=selected_image.uuid)                                        
                st.success("Replaced")
                time.sleep(1)
                st.rerun()
                
            else:
                number_of_image_variants = add_image_variant(
                    selected_image.uuid, timing.uuid)
                promote_image_variant(
                    timing.uuid, number_of_image_variants - 1)
                st.success("Replaced")
                time.sleep(1)
                st.rerun()
                                            
    elif replace_with == "Uploaded Frame":
        if stage == "source":
            uploaded_file = st.file_uploader("Upload Source Image", type=[
                "png", "jpeg"], accept_multiple_files=False)
            if st.button("Upload Source Image"):
                if uploaded_file:
                    timing = data_repo.get_timing_from_uuid(timing.uuid)
                    if save_uploaded_image(uploaded_file, timing.project.uuid, timing.uuid, "source"):
                        time.sleep(1.5)
                        st.rerun()
        else:
            replacement_frame = st.file_uploader("Upload a replacement frame here", type=[
                "png", "jpeg"], accept_multiple_files=False, key=f"replacement_frame_upload_{stage}")
            if st.button("Replace frame", disabled=False):
                images_for_model = []
                timing = data_repo.get_timing_from_uuid(timing.uuid)
                if replacement_frame:
                    saved_file = save_uploaded_image(replacement_frame, timing.project.uuid, timing.uuid, "styled")
                    if saved_file:
                        number_of_image_variants = add_image_variant(saved_file.uuid, timing.uuid)
                        promote_image_variant(
                            timing.uuid, number_of_image_variants - 1)
                        st.success("Replaced")
                        time.sleep(1)
                        st.rerun()

def jump_to_single_frame_view_button(display_number, timing_details):
    if st.button(f"Jump to #{display_number}"):
        st.session_state['prev_frame_index'] = display_number
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
        st.session_state['frame_styling_view_type'] = "Individual View"
        st.session_state['change_view_type'] = True
        st.rerun()

def change_position_input(timing_uuid, src):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    timing_list = data_repo.get_timing_list_from_project(project_uuid=timing.project.uuid)

    min_value = 1
    max_value = len(timing_list)

    new_position = st.number_input("Move to new position:", min_value=min_value, max_value=max_value,
                                   value=timing.aux_frame_index + 1, step=1, key=f"new_position_{timing.uuid}_{src}")
    
    if st.button('Update Position',key=f"change_frame_position_{timing.uuid}_{src}"): 
        data_repo.update_specific_timing(timing.uuid, aux_frame_index=new_position - 1)
        st.rerun()