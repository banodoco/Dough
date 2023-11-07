import time
from typing import Union
import streamlit as st
from ui_components.constants import CreativeProcessType, WorkflowStageType
from ui_components.models import InternalFileObject
from ui_components.widgets.image_zoom_widgets import zoom_inputs

from utils import st_memory

from utils.data_repo.data_repo import DataRepo

from utils.constants import ImageStage
from ui_components.methods.file_methods import generate_pil_image,save_or_host_file
from ui_components.methods.common_methods import apply_image_transformations, clone_styling_settings, create_frame_inside_shot, save_uploaded_image
from PIL import Image



def add_key_frame_section(shot_uuid, individual_view=True):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
    len_shot_timing_list = len(timing_list) if len(timing_list) > 0 else 0
    selected_image_location = ""
    source_of_starting_image = st.radio("Starting image:", ["None","Uploaded image", "Existing Frame"], key="source_of_starting_image")
    
    if source_of_starting_image == "Existing Frame":                
        image_idx = st.number_input("Which frame would you like to use?", min_value=1, max_value=max(1, len(timing_list)), value=len_shot_timing_list, step=1, key="image_idx")
        selected_image_location = timing_list[image_idx - 1].primary_image_location
    elif source_of_starting_image == "Uploaded image":            
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            file_location = f"videos/{shot.uuid}/assets/frames/1_selected/{uploaded_image.name}"
            selected_image_location = save_or_host_file(image, file_location)
            selected_image_location = selected_image_location or file_location
        else:
            selected_image_location = ""
        image_idx = len_shot_timing_list

    if individual_view:
        radio_text = "Inherit styling settings from the " + ("current frame?" if source_of_starting_image == "Uploaded image" else "selected frame")
        inherit_styling_settings = st_memory.radio(radio_text, ["Yes", "No"], key="inherit_styling_settings", horizontal=True)
        
        # apply_zoom_effects = st_memory.radio("Apply zoom effects to inputted image?", ["No","Yes"], key="apply_zoom_effects", horizontal=True)
        
        #if apply_zoom_effects == "Yes":
        #     zoom_inputs(position='new', horizontal=True)
    else:
        inherit_styling_settings = "Yes"
        apply_zoom_effects = "No"

    return selected_image_location, inherit_styling_settings

def display_selected_key_frame(selected_image_location, apply_zoom_effects):
    selected_image = None
    if selected_image_location:
        # if apply_zoom_effects == "Yes":
            # image_preview = generate_pil_image(selected_image_location)
            # selected_image = apply_image_transformations(image_preview, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'], st.session_state['flip_vertically'], st.session_state['flip_horizontally'])
        
        selected_image = generate_pil_image(selected_image_location)
        st.info("Starting Image:")                
        st.image(selected_image)
    else:
        st.error("No Starting Image Found")

    return selected_image

def add_key_frame_element(shot_uuid):
    add1, add2 = st.columns(2)
    with add1:
        selected_image_location, inherit_styling_settings  = add_key_frame_section(shot_uuid)
    with add2:
        selected_image = display_selected_key_frame(selected_image_location, False)
    
    return selected_image, inherit_styling_settings


def add_key_frame(selected_image: Union[Image.Image, InternalFileObject], inherit_styling_settings, shot_uuid, target_frame_position=None, refresh_state=True):
    '''
    either a pil image or a internalfileobject can be passed to this method, for adding it inside a shot
    '''
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    # checking if the shot has reached the max frame limit
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    project_settings = data_repo.get_project_setting(shot.project.uuid)
    if len(shot.timing_list) >= project_settings.max_frames_per_shot:
        st.error(f'Only {project_settings.max_frames_per_shot} frames allowed per shot')
        time.sleep(0.3)
        st.rerun()

    # creating frame inside the shot at target_frame_position
    len_shot_timing_list = len(timing_list) if len(timing_list) > 0 else 0
    target_frame_position = len_shot_timing_list if target_frame_position is None else target_frame_position
    target_aux_frame_index = min(len(timing_list), target_frame_position)
    _ = create_frame_inside_shot(shot_uuid, target_aux_frame_index)

    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
    # updating the newly created frame timing
    save_uploaded_image(selected_image, shot_uuid, timing_list[target_aux_frame_index].uuid, WorkflowStageType.SOURCE.value)
    save_uploaded_image(selected_image, shot_uuid, timing_list[target_aux_frame_index].uuid, WorkflowStageType.STYLED.value)

    if inherit_styling_settings == "Yes" and st.session_state['current_frame_index']:    
        clone_styling_settings(st.session_state['current_frame_index'] - 1, timing_list[target_aux_frame_index-1].uuid)

    if len(timing_list) == 1:
        st.session_state['current_frame_index'] = 1
        st.session_state['current_frame_uuid'] = timing_list[0].uuid
    else:
        st.session_state['prev_frame_index'] = min(len(timing_list), target_aux_frame_index + 1)
        st.session_state['current_frame_index'] = min(len(timing_list), target_aux_frame_index + 1)
        st.session_state['current_frame_uuid'] = timing_list[st.session_state['current_frame_index'] - 1].uuid

    st.session_state['page'] = CreativeProcessType.STYLING.value
    st.session_state['section_index'] = 0
    
    if refresh_state:
        st.rerun()