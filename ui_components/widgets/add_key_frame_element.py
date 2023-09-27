import streamlit as st

from utils import st_memory

from utils.data_repo.data_repo import DataRepo

from utils.constants import ImageStage
from ui_components.methods.file_methods import generate_pil_image,save_or_host_file
from ui_components.methods.common_methods import apply_image_transformations,zoom_inputs
from PIL import Image



def add_key_frame_element(timing_details, project_uuid):
    data_repo = DataRepo()

    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    project_settings = data_repo.get_project_setting(project_uuid)

    add1, add2 = st.columns(2)

    with add1:

        selected_image_location = ""
        image1,image2 = st.columns(2)
        with image1:
            source_of_starting_image = st.radio("Where would you like to get the starting image from?", [
                                                "Previous frame", "Uploaded image"], key="source_of_starting_image")
        
        which_stage_for_starting_image = None
        if source_of_starting_image == "Previous frame":                
            with image2:
                which_stage_for_starting_image = st.radio("Which stage would you like to use?", [
                                                        ImageStage.MAIN_VARIANT.value, ImageStage.SOURCE_IMAGE.value], key="which_stage_for_starting_image", horizontal=True)
                which_number_for_starting_image = st.number_input("Which frame would you like to use?", min_value=1, max_value=
                                                            max(1, len(timing_details)), value=st.session_state['current_frame_index'], step=1, key="which_number_for_starting_image")
            if which_stage_for_starting_image == ImageStage.SOURCE_IMAGE.value:
                if timing_details[which_number_for_starting_image - 1].source_image != "":
                    selected_image_location = timing_details[which_number_for_starting_image - 1].source_image.location
                else:
                    selected_image_location = ""
            elif which_stage_for_starting_image == ImageStage.MAIN_VARIANT.value:
                selected_image_location = timing_details[which_number_for_starting_image - 1].primary_image_location
        elif source_of_starting_image == "Uploaded image":
            with image2:
                uploaded_image = st.file_uploader(
                    "Upload an image", type=["png", "jpg", "jpeg"])
                # FILE UPLOAD HANDLE--
                if uploaded_image is not None:
                    image = Image.open(uploaded_image)
                    file_location = f"videos/{project_uuid}/assets/frames/1_selected/{uploaded_image.name}"
                    selected_image_location = save_or_host_file(image, file_location)
                    selected_image_location = selected_image_location or file_location
                else:
                    selected_image_location = ""
                which_number_for_starting_image = st.session_state['current_frame_index']

        
        how_long_after = st.slider(
            "How long after the current frame?", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        
        radio_text = "Inherit styling settings from the " + ("current frame?" if source_of_starting_image == "Uploaded image" else "selected frame")
        inherit_styling_settings = st_memory.radio(radio_text, ["Yes", "No"], \
                                                key="inherit_styling_settings", horizontal=True, project_settings=project_settings)
        
        apply_zoom_effects = st_memory.radio("Apply zoom effects to inputted image?", [
                                                        "No","Yes"], key="apply_zoom_effects", horizontal=True, project_settings=project_settings)
        
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

    return selected_image, inherit_styling_settings, how_long_after, which_stage_for_starting_image