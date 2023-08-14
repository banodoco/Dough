

import streamlit as st
import time
from shared.constants import InternalFileType
from ui_components.widgets.frame_time_selector import single_frame_time_selector
from utils.data_repo.data_repo import DataRepo
from ui_components.constants import WorkflowStageType
from shared.file_upload.s3 import upload_file
from ui_components.common_methods import delete_frame, add_image_variant, promote_image_variant, save_uploaded_image,display_image


def frame_selector_widget():
    data_repo = DataRepo()
    
    time1, time2 = st.columns([1,1])

    timing_details = data_repo.get_timing_list_from_project(project_uuid=st.session_state["project_uuid"])
    with time1:
        if 'prev_frame_index' not in st.session_state:
            st.session_state['prev_frame_index'] = 1

        st.write(st.session_state['prev_frame_index'])
        st.write(st.session_state['current_frame_index'])
        st.session_state['current_frame_index'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 1, len(timing_details), value=st.session_state['prev_frame_index'], step=1, key="which_image_selector")
        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
        if st.session_state['prev_frame_index'] != st.session_state['current_frame_index']:
            st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
            st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
            st.session_state['reset_canvas'] = True
            st.session_state['frame_styling_view_type_index'] = 0
            st.session_state['frame_styling_view_type'] = "Individual View"
                                        
            st.experimental_rerun()       

    with time2:
        single_frame_time_selector(st.session_state['current_frame_uuid'], 'navbar')
    

    def replace_image_widget(stage=WorkflowStageType.STYLED.value):
                                            
        replace_with = st.radio("Replace with:", [
            "Uploaded Frame", "Other Frame"], horizontal=True, key=f"replace_with_what_{stage}")
        

        if replace_with == "Other Frame":
                        
            which_stage_to_use_for_replacement = st.radio("Select stage to use:", [
                "Styled Image", "Guidance Image"], key=f"which_stage_to_use_for_replacement_{stage}", horizontal=True)
            which_image_to_use_for_replacement = st.number_input("Select image to use:", min_value=0, max_value=len(
                timing_details)-1, value=0, key=f"which_image_to_use_for_replacement_{stage}")
            
            if which_stage_to_use_for_replacement == "Guidance Image":                                    
                selected_image = timing_details[which_image_to_use_for_replacement].source_image
                
            
            elif which_stage_to_use_for_replacement == "Styled Image":
                selected_image = timing_details[which_image_to_use_for_replacement].primary_image
                
            
            st.image(selected_image.local_path, use_column_width=True)

            if st.button("Replace with selected frame", disabled=False,key=f"replace_with_selected_frame_{stage}"):
                if stage == "source":
                                             
                    data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=selected_image.uuid)                                        
                    st.success("Replaced")
                    time.sleep(1)
                    st.experimental_rerun()
                    
                else:
                    number_of_image_variants = add_image_variant(
                        selected_image.uuid, st.session_state['current_frame_uuid'])
                    promote_image_variant(
                        st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                    st.success("Replaced")
                    time.sleep(1)
                    st.experimental_rerun()
                        
                        

        elif replace_with == "Uploaded Frame":
            if stage == "source":
                uploaded_file = st.file_uploader("Upload Source Image", type=[
                    "png", "jpeg"], accept_multiple_files=False)
                if st.button("Upload Source Image"):
                    if uploaded_file:
                        timing = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                        if save_uploaded_image(uploaded_file, timing.project.uuid, st.session_state['current_frame_uuid'], "source"):
                            time.sleep(1.5)
                            st.experimental_rerun()
            else:
                replacement_frame = st.file_uploader("Upload a replacement frame here", type=[
                    "png", "jpeg"], accept_multiple_files=False, key=f"replacement_frame_upload_{stage}")
                if st.button("Replace frame", disabled=False):
                    images_for_model = []
                    timing = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                    if replacement_frame:
                        saved_file = save_uploaded_image(replacement_frame, timing.project.uuid, st.session_state['current_frame_uuid'], "styled")
                        if saved_file:
                            number_of_image_variants = add_image_variant(saved_file.uuid, st.session_state['current_frame_uuid'])
                            promote_image_variant(
                                st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                            st.success("Replaced")
                            time.sleep(1)
                            st.experimental_rerun()


    image_1, image_2 = st.columns([1,1])
    with image_1:
        st.warning(f"Guidance Image:")
        display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value, clickable=False)
        with st.expander("Replace guidance image"):
            replace_image_widget(stage=WorkflowStageType.SOURCE.value)
    with image_2:
        st.success(f"Main Styled Image:")
        display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)
        with st.expander("Replace styled image"):
            replace_image_widget(stage=WorkflowStageType.STYLED.value)
    
    st.markdown("***")
    
    if st.button("Delete key frame"):
        delete_frame(st.session_state['current_frame_uuid'])
        st.experimental_rerun()