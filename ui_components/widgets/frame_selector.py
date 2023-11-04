from typing import List
import streamlit as st
from ui_components.widgets.frame_movement_widgets import delete_frame_button, replace_image_widget,jump_to_single_frame_view_button
from ui_components.widgets.image_carousal import display_image
from ui_components.widgets.shot_view import update_shot_name,update_shot_duration, delete_shot_button
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from utils.data_repo.data_repo import DataRepo
from ui_components.constants import WorkflowStageType
from utils import st_memory



def frame_selector_widget():
    data_repo = DataRepo()
    time1, time2 = st.columns([1,1])

    timing_list = data_repo.get_timing_list_from_shot(st.session_state["shot_uuid"])
    shot = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
    shot_list = data_repo.get_shot_list(shot.project.uuid)
    len_timing_list = len(timing_list) if len(timing_list) > 0 else 1.0


    with time1:
        if 'prev_shot_index' not in st.session_state:
            st.session_state['prev_shot_index'] = shot.shot_idx

        shot_names = [s.name for s in shot_list]
        shot_name = st.selectbox('Shot name:', shot_names, key="current_shot_sidebar_selector",index=shot_names.index(shot.name))
        # find shot index based on shot name
        st.session_state['current_shot_index'] = shot_names.index(shot_name) + 1

        if shot_name != shot.name:
            st.session_state["shot_uuid"] = shot_list[shot_names.index(shot_name)].uuid
            st.rerun()
        
        if not ('current_shot_index' in st.session_state and st.session_state['current_shot_index']):
            st.session_state['current_shot_index'] = shot_names.index(shot_name) + 1
            update_current_shot_index(st.session_state['current_shot_index'])
        
        

    if st.session_state['page'] == "Key Frames":
        st.progress(st.session_state['current_frame_index'] / len_timing_list)
    elif st.session_state['page'] == "Shots":
        st.progress(st.session_state['current_shot_index'] / len(shot_list))
    if st.session_state['page'] == "Key Frames":
        with time2:
            if 'prev_frame_index' not in st.session_state:
                st.session_state['prev_frame_index'] = 1

            st.session_state['current_frame_index'] = st.number_input(f"Key frame # (out of {len(timing_list)})", 1, 
                                                                    len(timing_list), value=st.session_state['prev_frame_index'], 
                                                                    step=1, key="current_frame_sidebar_selector")
            
            update_current_frame_index(st.session_state['current_frame_index'])
        
        with st.expander(f"🖼️ Frame #{st.session_state['current_frame_index']} Details", expanded=True):
            if st_memory.toggle("Open", value=True, key="frame_toggle"):
                a1, a2 = st.columns([3,2])
                with a1:
                    st.success(f"Main Key Frame:")
                    display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)


                    # st.warning(f"Guidance Image:")
                    # display_image(st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value, clickable=False)
                with a2:
                    st.caption("Replace styled image")
                    replace_image_widget(st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value)
                
                st.markdown("---")

                st.info("In Context:")
                shot_list = data_repo.get_shot_list(shot.project.uuid)
                shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])

                # shot = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
                timing_list: List[InternalFrameTimingObject] = shot.timing_list

                display_shot_frames(timing_list, False)

                st.markdown("---")

                delete_frame_button(st.session_state['current_frame_uuid'])

                

    else:
        shot_list = data_repo.get_shot_list(shot.project.uuid)
        shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
        with st.expander(f"🎬 {shot.name} Details",expanded=True):
            if st_memory.toggle("Open", value=True, key="shot_details_toggle"):
                a1,a2 = st.columns([2,2])
                with a1:
                    update_shot_name(shot, data_repo)
                with a2:
                    update_shot_duration(shot, data_repo)
                
                st.markdown("---")

                timing_list: List[InternalFrameTimingObject] = shot.timing_list

                display_shot_frames(timing_list, True)

                st.markdown("---")

                delete_shot_button(shot, data_repo)

def update_current_frame_index(index):
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(st.session_state["shot_uuid"])

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
    shot_list = data_repo.get_shot_list(st.session_state["project_uuid"])

    st.session_state['shot_uuid'] = shot_list[index - 1].uuid
        
    if st.session_state['prev_shot_index'] != index:
        st.session_state['prev_shot_index'] = index
        st.session_state['shot_uuid'] = shot_list[index - 1].uuid
        st.session_state['reset_canvas'] = True
        st.session_state['frame_styling_view_type_index'] = 0
        st.session_state['frame_styling_view_type'] = "Individual View"
                                    
        st.rerun()       


def display_shot_frames(timing_list: List[InternalFrameTimingObject], show_button: bool):
    if timing_list and len(timing_list):
        items_per_row = 3
        for i in range(0, len(timing_list), items_per_row):
            with st.container():
                grid = st.columns(items_per_row)
                for j in range(items_per_row):
                    idx = i + j
                    if idx < len(timing_list):
                        timing = timing_list[idx]
                        with grid[j]:
                            if timing.primary_image and timing.primary_image.location:
                                st.image(timing.primary_image.location, use_column_width=True)
                                # Show button if show_button is True
                                if show_button:
                                    # Call jump_to_single_frame_view_button function
                                    jump_to_single_frame_view_button(idx + 1, timing_list, f"jump_to_{idx + 1}")
                            else:
                                st.warning("No primary image present")
            st.markdown("***")
    else:
        st.warning("No keyframes present")