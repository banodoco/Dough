from typing import List
from ui_components.methods.common_methods import update_clip_duration_of_all_timing_frames
from ui_components.models import InternalFrameTimingObject
from utils.data_repo.data_repo import DataRepo
import streamlit as st

def shift_subsequent_frames(timing, time_delta):
    data_repo = DataRepo()
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    if time_delta > 0:
        for a in range(timing.aux_frame_index + 1, len(timing_details)):
            frame = timing_details[a]
            # shift them by the difference between the new frame time and the old frame time
            new_frame_time = frame.frame_time + time_delta
            data_repo.update_specific_timing(frame.uuid, frame_time=new_frame_time, timed_clip_id=None)

def update_frame_duration(timing_uuid, frame_duration, next_timing, shift_frames):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    
    if next_timing:
        # Calculate time_delta before updating next_timing.frame_time
        time_delta = frame_duration - (next_timing.frame_time - timing.frame_time)

        next_timing.frame_time = timing.frame_time + frame_duration
        data_repo.update_specific_timing(next_timing.uuid, frame_time=next_timing.frame_time, timed_clip_id=None)

        if shift_frames:
            shift_subsequent_frames(timing, time_delta)
    
    # updating clip_duration
    update_clip_duration_of_all_timing_frames(timing.project.uuid)

    st.rerun()

def update_frame_time(timing_uuid, frame_time, shift_frames):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    data_repo.update_specific_timing(timing_uuid, frame_time=frame_time, timed_clip_id=None)

    if shift_frames:
        next_timing = data_repo.get_next_timing(timing_uuid)
        if next_timing is not None:
            time_delta = (frame_time + timing.clip_duration) - next_timing.frame_time
            shift_subsequent_frames(timing, time_delta)
    
    # updating clip_duration
    update_clip_duration_of_all_timing_frames(timing.project.uuid)

    st.rerun()


def single_frame_time_duration_setter(timing_uuid, src, shift_frames=True):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    next_timing = data_repo.get_next_timing(timing_uuid)
    
    # Calculate clip_duration
    if next_timing:
        clip_duration = next_timing.frame_time - timing.frame_time
    else:
        clip_duration = 0.0  # or some default value

    max_value = 100.0 if shift_frames else clip_duration
    
    disable_duration_input = False if next_timing else True
    help_text = None if shift_frames else "You currently won't shift subsequent frames - to do this, go to the List View and turn on Shift Frames."
    frame_duration = st.number_input("Duration:", min_value=0.0, max_value=max_value,
                                     value=clip_duration, step=0.1, key=f"frame_duration_{timing.aux_frame_index}_{src}", 
                                     disabled=disable_duration_input, help=help_text)
    
    if frame_duration != clip_duration:
        update_frame_duration(timing_uuid, frame_duration, next_timing, shift_frames)



def single_frame_time_selector(timing_uuid, src, shift_frames=True):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    timing_list: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(timing.project.uuid)
    prev_timing = None
    if timing.aux_frame_index > 0:
        prev_timing_uuid = timing_list[timing.aux_frame_index - 1].uuid
        prev_timing = data_repo.get_timing_from_uuid(prev_timing_uuid)

    min_value = prev_timing.frame_time if prev_timing else 0.0

    disabled_time_change = True if timing.aux_frame_index == 0 else False

    next_timing = data_repo.get_next_timing(timing_uuid)
    if next_timing:
        max_value = 100.0 if shift_frames else next_timing.frame_time
    else:
        max_value = timing.frame_time + 100  # Allow up to 100 seconds more if it's the last item

    help_text = None if shift_frames else "You currently won't shift subsequent frames - to do this, go to the List View and turn on Shift Frames."
    frame_time = st.number_input("Time:", min_value=min_value, max_value=max_value,
                                     value=timing.frame_time, step=0.1, key=f"frame_time_{timing.aux_frame_index}_{src}",disabled=disabled_time_change, help=help_text)
    if frame_time != timing.frame_time:
        update_frame_time(timing_uuid, frame_time, shift_frames)



