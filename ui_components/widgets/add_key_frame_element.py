import streamlit as st
from ui_components.constants import CreativeProcessType, WorkflowStageType
from ui_components.widgets.image_zoom_widgets import zoom_inputs

from utils import st_memory

from utils.data_repo.data_repo import DataRepo

from utils.constants import ImageStage
from ui_components.methods.file_methods import generate_pil_image,save_or_host_file
from ui_components.methods.common_methods import apply_image_transformations, clone_styling_settings, create_timings_row_at_frame_number, save_uploaded_image
from PIL import Image



def add_key_frame_element(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
    add1, add2 = st.columns(2)

    with add1:
        selected_image_location = ""
        image1,image2 = st.columns(2)
        with image1:
            source_of_starting_image = st.radio("Where would you like to get the starting image from?", [
                                                "Existing Frame", "Uploaded image"], key="source_of_starting_image")
        
        transformation_stage = None
        if source_of_starting_image == "Existing Frame":                
            with image2:
                transformation_stage = st.radio(
                                                label="Which stage would you like to use?",
                                                options=ImageStage.value_list(),
                                                key="transformation_stage-bottom",
                                                horizontal=True
                                            )
                image_idx = st.number_input(
                                            "Which frame would you like to use?", 
                                            min_value=1, 
                                            max_value=max(1, len(timing_list)), 
                                            value=st.session_state['current_frame_index'], 
                                            step=1, 
                                            key="image_idx"
                                        )
            if transformation_stage == ImageStage.SOURCE_IMAGE.value:
                if timing_list[image_idx - 1].source_image is not None and timing_list[image_idx - 1].source_image != "":
                    selected_image_location = timing_list[image_idx - 1].source_image.location
                else:
                    selected_image_location = ""
            elif transformation_stage == ImageStage.MAIN_VARIANT.value:
                selected_image_location = timing_list[image_idx - 1].primary_image_location
        elif source_of_starting_image == "Uploaded image":
            with image2:
                uploaded_image = st.file_uploader(
                    "Upload an image", type=["png", "jpg", "jpeg"])
                # FILE UPLOAD HANDLE--
                if uploaded_image is not None:
                    image = Image.open(uploaded_image)
                    file_location = f"videos/{shot.uuid}/assets/frames/1_selected/{uploaded_image.name}"
                    selected_image_location = save_or_host_file(image, file_location)
                    selected_image_location = selected_image_location or file_location
                else:
                    selected_image_location = ""
                image_idx = st.session_state['current_frame_index']

        
        how_long_after = st.slider(
            "How long after the current frame?", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        
        radio_text = "Inherit styling settings from the " + ("current frame?" if source_of_starting_image == "Uploaded image" else "selected frame")
        inherit_styling_settings = st_memory.radio(radio_text, ["Yes", "No"], \
                                                key="inherit_styling_settings", horizontal=True)
        
        apply_zoom_effects = st_memory.radio("Apply zoom effects to inputted image?", [
                                                        "No","Yes"], key="apply_zoom_effects", horizontal=True)
        
        if apply_zoom_effects == "Yes":
            zoom_inputs(position='new', horizontal=True)

    selected_image = None
    with add2:
        if selected_image_location:
            if apply_zoom_effects == "Yes":
                image_preview = generate_pil_image(selected_image_location)
                selected_image = apply_image_transformations(image_preview, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'])
            else:
                selected_image = generate_pil_image(selected_image_location)
            st.info("Starting Image:")                
            st.image(selected_image)
        else:
            st.error("No Starting Image Found")

    return selected_image, inherit_styling_settings, how_long_after, transformation_stage

def add_key_frame(selected_image, inherit_styling_settings, how_long_after, target_frame_position=None, refresh_state=True):
    data_repo = DataRepo()
    shot_uuid = st.session_state['shot_uuid']
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
    project_settings = data_repo.get_project_setting(shot_uuid)

    if len(timing_list) == 0:
        index_of_current_item = 1
    else:
        target_frame_position = st.session_state['current_frame_index'] if target_frame_position is None else target_frame_position
        index_of_current_item = min(len(timing_list), target_frame_position)

    if len(timing_list) == 0:
        key_frame_time = 0.0
    elif target_frame_position is not None:
        key_frame_time = float(timing_list[target_frame_position - 1].frame_time) + how_long_after
    elif index_of_current_item == len(timing_list):
        key_frame_time = float(timing_list[index_of_current_item - 1].frame_time) + how_long_after
    else:
        key_frame_time = (float(timing_list[index_of_current_item - 1].frame_time) + float(
            timing_list[index_of_current_item].frame_time)) / 2.0

    if len(timing_list) == 0:
        _ = create_timings_row_at_frame_number(shot_uuid, 0)
    else:
        _ = create_timings_row_at_frame_number(shot_uuid, index_of_current_item, frame_time=key_frame_time)

    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
    if selected_image:
        save_uploaded_image(selected_image, shot_uuid, timing_list[index_of_current_item].uuid, WorkflowStageType.SOURCE.value)
        save_uploaded_image(selected_image, shot_uuid, timing_list[index_of_current_item].uuid, WorkflowStageType.STYLED.value)

    if inherit_styling_settings == "Yes":    
        clone_styling_settings(index_of_current_item - 1, timing_list[index_of_current_item].uuid)

    timing_list[index_of_current_item].animation_style = project_settings.default_animation_style

    if len(timing_list) == 1:
        st.session_state['current_frame_index'] = 1
        st.session_state['current_frame_uuid'] = timing_list[0].uuid
    else:
        st.session_state['prev_frame_index'] = min(len(timing_list), index_of_current_item + 1)
        st.session_state['current_frame_index'] = min(len(timing_list), index_of_current_item + 1)
        st.session_state['current_frame_uuid'] = timing_list[st.session_state['current_frame_index'] - 1].uuid

    st.session_state['page'] = CreativeProcessType.STYLING.value
    st.session_state['section_index'] = 0
    
    if refresh_state:
        st.rerun()