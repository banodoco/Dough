from typing import List
from ui_components.models import InternalFrameTimingObject
from utils.data_repo.data_repo import DataRepo
import streamlit as st


def single_frame_time_selector(timing_uuid, src):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    # src is required to create unique widget key
    frame_time = st.number_input("Frame time (secs):", min_value=0.0, max_value=100.0,
                                 value=timing.frame_time, step=0.1, key=f"frame_time_{timing.aux_frame_index}_{src}")

    if frame_time != timing.frame_time:
        data_repo.update_specific_timing(timing_uuid, frame_time=frame_time)
        if timing.aux_frame_index != 0:
            prev_timing = data_repo.get_prev_timing(timing_uuid)
            data_repo.update_specific_timing(
                prev_timing.uuid, timed_clip_id=None)

        data_repo.update_specific_timing(timing_uuid, timed_clip_id=None)

        # if the frame time of this frame is more than the frame time of the next frame,
        # then we need to update the next frame's frame time, and all the frames after that
        # - shift them by the difference between the new frame time and the old frame time

        next_timing = data_repo.get_next_timing(timing_uuid)
        if next_timing and frame_time > next_timing.frame_time:
            for a in range(timing.aux_frame_index, len(timing_details)):
                frame = timing_details[a]
                this_frame_time = frame.frame_time
                # shift them by the difference between the new frame time and the old frame time
                new_frame_time = this_frame_time + \
                    (frame_time - timing.frame_time)
                data_repo.update_specific_timing(
                    frame.uuid, frame_time=new_frame_time)
                data_repo.update_specific_timing(
                    frame.uuid, timed_clip_id=None)
        st.experimental_rerun()
