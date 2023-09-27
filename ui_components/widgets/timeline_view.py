import streamlit as st
from ui_components.methods.common_methods import delete_frame, jump_to_single_frame_view_button, move_frame
from ui_components.widgets.frame_time_selector import single_frame_time_selector, single_frame_time_duration_setter
from ui_components.widgets.image_carousal import display_image
from utils.data_repo.data_repo import DataRepo
from ui_components.widgets.frame_clip_generation_elements import update_animation_style_element
from ui_components.constants import WorkflowStageType

def timeline_view_buttons(i, j, timing_details, items_per_row):
    if items_per_row > 6:
        jump_to_single_frame_view_button(i + j + 1, timing_details)
        st.markdown("***")
        btn1, btn2, btn3 = st.columns([1, 1, 1])
        with btn1:                                            
            st.button("⬅️", key=f"move_frame_back_{i + j + 1}", help="Move frame back")
        with btn2:   
            st.button("➡️", key=f"move_frame_forward_{i + j + 1}", help="Move frame forward")
        with btn3:
            st.button("🗑️", key=f"delete_frame_{i + j + 1}", help="Delete frame")
    else:
        btn1, btn2, btn3, btn4 = st.columns([1.7, 1, 1, 1])
        with btn1:                                 
            jump_to_single_frame_view_button(i + j + 1, timing_details)
        with btn2:
            st.button("⬅️", key=f"move_frame_back_{i + j + 1}", help="Move frame back")
        with btn3:
            st.button("➡️", key=f"move_frame_forward_{i + j + 1}", help="Move frame forward")
        with btn4:
            st.button("🗑️", key=f"delete_frame_{i + j + 1}", help="Delete frame")


def timeline_view(shift_frames_setting, project_uuid, items_per_row, expand_all, stage='Styling'):
    data_repo = DataRepo()
    timing = data_repo.get_timing_list_from_project(project_uuid)[0]
    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    for i in range(0, len(timing_details), items_per_row):  # Step of items_per_row for grid
        grid = st.columns(items_per_row)  # Create items_per_row columns for grid
        for j in range(items_per_row):
            if i + j < len(timing_details):  # Check if index is within range
                with grid[j]:
                    display_number = i + j + 1                                
                    if stage == 'Styling':
                        display_image(timing_uuid=timing_details[i + j].uuid, stage=WorkflowStageType.STYLED.value, clickable=False)
                    elif stage == 'Motion':
                        if timing.timed_clip:
                            st.video(timing.timed_clip.location)
                        else:
                            st.error("No video found for this frame.")
                    with st.expander(f'Frame {display_number}', expanded=expand_all):                                                                        
                        single_frame_time_selector(timing_details[i + j].uuid, 'motion', shift_frames=shift_frames_setting)                                    
                        single_frame_time_duration_setter(timing_details[i + j].uuid, 'motion', shift_frames=shift_frames_setting)
                        update_animation_style_element(timing_details[i + j].uuid)
                        timeline_view_buttons(i, j, timing_details, items_per_row)
                        # if move_frame_back button is clicked
                        if st.session_state[f"move_frame_back_{i + j + 1}"]:                                        
                            move_frame("Up", timing_details[i + j].uuid)      
                            st.experimental_rerun()
                        if st.session_state[f"move_frame_forward_{i + j + 1}"]:
                            move_frame("Down", timing_details[i + j].uuid)                                      
                            st.experimental_rerun()
                        if st.session_state[f"delete_frame_{i + j + 1}"]:
                            delete_frame(timing_details[i + j].uuid)
                            st.experimental_rerun()
                                                                
                    st.markdown("***")