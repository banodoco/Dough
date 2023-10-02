from typing import List
import streamlit as st

from ui_components.models import InternalFrameTimingObject
from utils.data_repo.data_repo import DataRepo


def back_and_forward_buttons():
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        st.session_state['current_frame_uuid'])
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    smallbutton0, smallbutton1, smallbutton2, smallbutton3, smallbutton4 = st.columns([
                                                                                      2, 2, 2, 2, 2])

    display_idx = st.session_state['current_frame_index']
    with smallbutton0:
        if display_idx > 2:
            if st.button(f"{display_idx-2} ⏮️", key=f"Previous Previous Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] - 2
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.rerun()
    with smallbutton1:
        # if it's not the first image
        if display_idx != 1:
            if st.button(f"{display_idx-1} ⏪", key=f"Previous Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] - 1
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.rerun()

    with smallbutton2:
        st.button(f"{display_idx} 📍", disabled=True)
    with smallbutton3:
        # if it's not the last image
        if display_idx != len(timing_details):
            if st.button(f"{display_idx+1} ⏩", key=f"Next Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 1
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.rerun()
    with smallbutton4:
        if display_idx <= len(timing_details)-2:
            if st.button(f"{display_idx+2} ⏭️", key=f"Next Next Image for {display_idx}"):
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 2
                st.session_state['prev_frame_index'] = st.session_state['current_frame_index']
                st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                st.rerun()
