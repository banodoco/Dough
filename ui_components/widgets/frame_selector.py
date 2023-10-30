import streamlit as st
from ui_components.widgets.frame_movement_widgets import delete_frame, replace_image_widget
from ui_components.widgets.image_carousal import display_image
from utils.data_repo.data_repo import DataRepo
from ui_components.constants import WorkflowStageType


def frame_selector_widget():
    data_repo = DataRepo()
    time1, time2 = st.columns([1,1])

    timing_list = data_repo.get_timing_list_from_shot(shot_uuid=st.session_state["shot_uuid"])
    shot = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
    shot_list = data_repo.get_shot_list(shot.project.uuid)
    len_timing_list = len(timing_list) if len(timing_list) > 0 else 1.0
    st.progress(st.session_state['current_frame_index'] / len_timing_list)

    with time1:
        if 'prev_shot_index' not in st.session_state:
            st.session_state['prev_shot_index'] = shot.shot_idx

        st.session_state['current_shot_index'] = st.number_input(f"Shot # (out of {len(shot_list)})", 1, 
                                                                  len(shot_list), value=st.session_state['prev_shot_index'], 
                                                                  step=1, key="current_shot_sidebar_selector")
        
        update_current_shot_index(st.session_state['current_shot_index'])

    with time2:
        if 'prev_frame_index' not in st.session_state:
            st.session_state['prev_frame_index'] = 1

        st.session_state['current_frame_index'] = st.number_input(f"Key frame # (out of {len(timing_list)})", 1, 
                                                                  len(timing_list), value=st.session_state['prev_frame_index'], 
                                                                  step=1, key="current_frame_sidebar_selector")
        
        update_current_frame_index(st.session_state['current_frame_index'])
    
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
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid=st.session_state["shot_uuid"])

    st.session_state['current_frame_uuid'] = timing_list[index - 1].uuid
        
    if st.session_state['prev_frame_index'] != index:
        st.session_state['prev_frame_index'] = index
        st.session_state['current_frame_uuid'] = timing_list[index - 1].uuid
        st.session_state['reset_canvas'] = True
        st.session_state['frame_styling_view_type_index'] = 0
        st.session_state['frame_styling_view_type'] = "Individual View"
                                    
        st.rerun()


def update_current_shot_index(index):
    data_repo = DataRepo()
    shot_list = data_repo.get_shot_list(project_uuid=st.session_state["project_uuid"])

    st.session_state['shot_uuid'] = shot_list[index - 1].uuid
        
    if st.session_state['prev_shot_index'] != index:
        st.session_state['prev_shot_index'] = index
        st.session_state['shot_uuid'] = shot_list[index - 1].uuid
        st.session_state['reset_canvas'] = True
        st.session_state['frame_styling_view_type_index'] = 0
        st.session_state['frame_styling_view_type'] = "Individual View"
                                    
        st.rerun()       
