import streamlit as st
from ui_components.widgets.frame_movement_widgets import delete_frame, replace_image_widget
from ui_components.widgets.frame_time_selector import single_frame_time_selector
from ui_components.widgets.image_carousal import display_image
from utils.data_repo.data_repo import DataRepo
from ui_components.constants import WorkflowStageType


def frame_selector_widget():
    data_repo = DataRepo()
    
    time1, time2 = st.columns([1,1])

    timing_details = data_repo.get_timing_list_from_project(project_uuid=st.session_state["project_uuid"])
    len_timing_details = len(timing_details) if len(timing_details) > 0 else 1.0
    st.progress(st.session_state['current_frame_index'] / len_timing_details)
    with time1:
        if 'prev_frame_index' not in st.session_state:
            st.session_state['prev_frame_index'] = 1

        st.session_state['current_frame_index'] = st.number_input(f"Key frame # (out of {len(timing_details)})", 1, 
                                                                  len(timing_details), value=st.session_state['prev_frame_index'], 
                                                                  step=1, key="which_image_selector")
        
        update_current_frame_index(st.session_state['current_frame_index'])

    with time2:
        single_frame_time_selector(st.session_state['current_frame_uuid'], 'navbar', shift_frames=False)
    
    with st.expander(f"🖼️ Frame #{st.session_state['current_frame_index']} Details"):
        a1, a2 = st.columns([1,1])
        with a1:
            st.warning(f"Guidance Image:")
            display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value, clickable=False)

    with st.expander(f"🖼️ Frame #{st.session_state['current_frame_index']} Details"):
        a1, a2 = st.columns([1,1])
        with a1:
            st.warning(f"Guidance Image:")
            display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value, clickable=False)

        with a2:
            st.success(f"Main Styled Image:")
            display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)

        st.markdown("---")
        
        b1, b2 = st.columns([1,1])
        with b1:
            st.caption("Replace guidance image")
            replace_image_widget(st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value)

        with b2:
            st.caption("Replace styled image")
            replace_image_widget(st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value)
            
        st.markdown("---")
        
        if st.button("Delete key frame"):
            delete_frame(st.session_state['current_frame_uuid'])
            st.rerun()


def update_current_frame_index(index):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid=st.session_state["project_uuid"])

    st.session_state['current_frame_uuid'] = timing_details[index - 1].uuid
        
    if st.session_state['prev_frame_index'] != index:
        st.session_state['prev_frame_index'] = index
        st.session_state['current_frame_uuid'] = timing_details[index - 1].uuid
        st.session_state['reset_canvas'] = True
        st.session_state['frame_styling_view_type_index'] = 0
        st.session_state['frame_styling_view_type'] = "Individual View"
                                    
        st.rerun()       
