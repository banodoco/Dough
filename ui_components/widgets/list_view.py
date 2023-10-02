import streamlit as st
from ui_components.constants import WorkflowStageType
from utils.data_repo.data_repo import DataRepo
from ui_components.widgets.frame_time_selector import single_frame_time_selector, single_frame_time_duration_setter
from ui_components.widgets.image_carousal import display_image
from ui_components.methods.common_methods import delete_frame, move_frame,jump_to_single_frame_view_button
import math
from utils.data_repo.data_repo import DataRepo
from ui_components.methods.common_methods import delete_frame
from ui_components.widgets.frame_clip_generation_elements import current_individual_clip_element, current_preview_video_element, update_animation_style_element
from ui_components.widgets.frame_time_selector import single_frame_time_selector, single_frame_time_duration_setter
from ui_components.widgets.image_carousal import display_image
from ui_components.widgets.frame_clip_generation_elements import current_individual_clip_element, current_preview_video_element
from utils.data_repo.data_repo import DataRepo

def list_view_set_up(timing_details,project_uuid):
    data_repo = DataRepo()
    
    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1

    if not('index_of_current_page' in st.session_state and st.session_state['index_of_current_page']):
        st.session_state['index_of_current_page'] = 1

    items_per_page = 10
    num_pages = math.ceil(len(timing_details) / items_per_page) + 1

    return num_pages, items_per_page

def page_toggle(num_pages, items_per_page, project_uuid, position):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    st.session_state['current_page'] = st.radio(f"Select page:", options=range(
        1, num_pages), horizontal=True, index=st.session_state['index_of_current_page'] - 1, key=f"page_selection_radio_{position}")
    if st.session_state['current_page'] != st.session_state['index_of_current_page']:
        st.session_state['index_of_current_page'] = st.session_state['current_page']
        st.rerun()

    start_index = (st.session_state['current_page'] - 1) * items_per_page         
    end_index = min(start_index + items_per_page,len(timing_details))

    return start_index, end_index

def styling_list_view(start_index, end_index, shift_frames_setting, project_uuid):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    for i in range(start_index, end_index):
        display_number = i + 1
        st.subheader(f"Frame {display_number}")
        image1, image2, image3 = st.columns([2, 3, 2])

        with image1:
            display_image(timing_uuid=timing_details[i].uuid, stage=WorkflowStageType.SOURCE.value, clickable=False)

        with image2:
            display_image(timing_uuid=timing_details[i].uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

        with image3:
            time1, time2 = st.columns([1, 1])
            with time1:
                single_frame_time_selector(timing_details[i].uuid, 'sidebar', shift_frames=shift_frames_setting)
                single_frame_time_duration_setter(timing_details[i].uuid,'sidebar',shift_frames=shift_frames_setting)

            with time2:
                st.write("") 


            jump_to_single_frame_view_button(display_number,timing_details)

            st.markdown("---")
            btn1, btn2, btn3 = st.columns([2, 1, 1])
            with btn1:
                if st.button("Delete this keyframe", key=f'{i}'):
                    delete_frame(timing_details[i].uuid)
                    st.rerun()
            with btn2:
                if st.button("⬆️", key=f"Promote {display_number}"):
                    move_frame("Up", timing_details[i].uuid)
                    st.rerun()
            with btn3:
                if st.button("⬇️", key=f"Demote {display_number}"):
                    move_frame("Down", timing_details[i].uuid)
                    st.rerun()

        st.markdown("***")

def motion_list_view(start_index, end_index, shift_frames_setting, project_uuid):
    data_repo = DataRepo()
    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    num_timing_details = len(timing_details)
    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    for idx in range(start_index, end_index):
        st.header(f"Frame {idx+1}")
        timing1, timing2, timing3 = st.columns([1, 1, 1])

        with timing1:
            frame1, frame2, frame3 = st.columns([2, 1, 2])
            with frame1:
                if timing_details[idx].primary_image_location:
                    st.image(timing_details[idx].primary_image_location)
            with frame2:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.info("     ➜")
            with frame3:
                if idx+1 < num_timing_details and timing_details[idx+1].primary_image_location:
                    st.image(timing_details[idx+1].primary_image_location)
                elif idx+1 == num_timing_details:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.markdown("<h1 style='text-align: center; color: black; font-family: Arial; font-size: 50px; font-weight: bold;'>FIN</h1>", unsafe_allow_html=True)

            single_frame_time_selector(timing_details[idx].uuid, 'motion', shift_frames=shift_frames_setting)
            single_frame_time_duration_setter(timing_details[idx].uuid, 'motion', shift_frames=shift_frames_setting)
            update_animation_style_element(timing_details[idx].uuid)

        if timing_details[idx].aux_frame_index != len(timing_details) - 1:
            with timing2:
                current_individual_clip_element(timing_details[idx].uuid)
            with timing3:
                current_preview_video_element(timing_details[idx].uuid)

        st.markdown("***")