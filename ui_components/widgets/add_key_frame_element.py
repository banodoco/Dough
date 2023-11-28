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



import zipfile
import os

def add_key_frame_section(shot_uuid, individual_view=True):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)    
    selected_image_location = ""
    
    uploaded_image = st.file_uploader("Upload an image or zipped up images:", type=["png", "jpg", "jpeg","zip"], key=f"uploaded_image_{shot_uuid}", help="You can upload a single image or a zip file containing multiple images")
    
    if st.button(f"Add key frame(s)",type="primary",use_container_width=True, key=f"add_key_frame_btn_{shot_uuid}"):
        if uploaded_image.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_image, 'r') as zip_ref:
                zip_ref.extractall('temp_dir')
            for filename in os.listdir('temp_dir'):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                image = Image.open(os.path.join('temp_dir', filename))
                file_location = f"videos/{shot.uuid}/assets/frames/1_selected/{filename}"
                selected_image_location = save_or_host_file(image, file_location)
                selected_image_location = selected_image_location or file_location
                add_key_frame(selected_image_location, "No", shot_uuid,refresh_state=False)
        else:
            image = Image.open(uploaded_image)
            file_location = f"videos/{shot.uuid}/assets/frames/1_selected/{uploaded_image.name}"
            selected_image_location = save_or_host_file(image, file_location)
            selected_image_location = selected_image_location or file_location
            add_key_frame(selected_image_location, "No", shot_uuid)
        st.rerun()
    

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


def add_key_frame(selected_image: Union[Image.Image, InternalFileObject], inherit_styling_settings, shot_uuid, target_frame_position=None, refresh_state=True, update_cur_frame_idx=True):
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
    save_uploaded_image(selected_image, shot.project.uuid, timing_list[target_aux_frame_index].uuid, WorkflowStageType.SOURCE.value)
    save_uploaded_image(selected_image, shot.project.uuid, timing_list[target_aux_frame_index].uuid, WorkflowStageType.STYLED.value)

    if update_cur_frame_idx:
        # this part of code updates current_frame_index when a new keyframe is added
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