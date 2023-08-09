import streamlit as st
from ui_components.common_methods import delete_frame, display_image, single_frame_time_changer
from ui_components.constants import WorkflowStageType

from utils.data_repo.data_repo import DataRepo

def frame_selector_widget():
    data_repo = DataRepo()
    
    time1, time2 = st.columns([1,1])

    timing_details = data_repo.get_timing_list_from_project(project_uuid=st.session_state["project_uuid"])
    with time1:
        if 'prev_frame_index' not in st.session_state:
            st.session_state['prev_frame_index'] = 0

        st.session_state['current_frame_index'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(timing_details)-1, value=st.session_state['prev_frame_index'], step=1, key="which_image_selector")
        if st.session_state['prev_frame_index'] != st.session_state['current_frame_index']:
            st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
            st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index']].uuid
            st.session_state['reset_canvas'] = True
            st.session_state['frame_styling_view_type_index'] = 0
            st.session_state['frame_styling_view_type'] = "Individual View"
                                        
            st.experimental_rerun()       

    with time2:
        single_frame_time_changer(st.session_state['current_frame_uuid'])

    with st.expander("Notes:"):
            
        notes = st.text_area("Frame Notes:", value=timing_details[st.session_state['current_frame_index']].notes, height=100, key="notes")

    if notes != timing_details[st.session_state['current_frame_index']].notes:
        data_repo.update_specific_timing(st.session_state['current_frame_uuid'], notes=notes)
        st.experimental_rerun()
    
    if st.session_state['page'] == "Guidance":
        image_1_size = 2
        image_2_size = 1.5
    elif st.session_state['page'] == "Styling":
        image_1_size = 1.5
        image_2_size = 2
    elif st.session_state['page'] == "Motion":
        image_1_size = 1.5
        image_2_size = 1.5

    image_1, image_2 = st.columns([image_1_size,image_2_size])
    with image_1:
        st.caption(f"Guidance Image for Frame #{st.session_state['current_frame_index']}:")
        display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value, clickable=False)
    with image_2:
        st.caption(f"Main Styled Image for Frame #{st.session_state['current_frame_index']}:")
        display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)
    st.markdown("***")
    

    
    if st.button("Delete key frame"):
        delete_frame(st.session_state['current_frame_uuid'])
        st.experimental_rerun()