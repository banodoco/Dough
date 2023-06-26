from typing import List
import streamlit as st
import time

from ui_components.common_methods import add_image_variant, promote_image_variant
from ui_components.models import InternalFrameTimingObject, InternalProjectObject
from utils.data_repo.data_repo import DataRepo


def batch_action_page(project_uuid):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(project_uuid)
    timing_details: List[
        InternalFrameTimingObject
    ] = data_repo.get_timing_list_from_project(project_uuid)

    st.markdown("***")

    st.markdown("#### Make extracted key frames into completed key frames")
    st.write(
        "This will move all the extracted key frames to completed key frames - good for if you don't want to make any changes to the key frames"
    )
    if st.button("Move initial key frames to completed key frames"):
        for timing_frame in timing_details:
            add_image_variant(timing_frame.source_image.uuid, timing_frame.uuid)
            promote_image_variant(timing_frame.uuid, len(timing_details) - 1)
        st.success("All initial key frames moved to completed key frames")

    st.markdown("***")

    st.markdown("#### Remove all existing timings")
    st.write("This will remove all the timings and key frames from the project")
    if st.button("Remove Existing Timings"):
        data_repo.remove_existing_timing(project.uuid)

    st.markdown("***")

    st.markdown("#### Bulk adjust the timings")
    st.write(
        "This will adjust the timings of all the key frames by the number of seconds you enter below"
    )
    bulk_adjustment = st.number_input(
        "What multiple would you like to adjust the timings by?", value=1.0
    )
    if st.button("Adjust Timings"):
        for timing_frame in timing_details:
            new_frame_time = float(timing_frame.frame_time) * bulk_adjustment
            data_repo.update_specific_timing(
                timing_frame.uuid, frame_time=new_frame_time
            )

        st.success("Timings adjusted successfully!")
        time.sleep(1)
        st.experimental_rerun()

    st.markdown("***")
    st.markdown("#### Remove all variants other than main")
    st.write("This will remove all the variants of the key frames except the main one")
    if st.button("Remove all variants"):
        for timing_frame in timing_details:
            data_repo.update_specific_timing(timing_frame.uuid, alternative_images=None)
            data_repo.remove_primay_frame(timing_frame.uuid)

        st.success("All variants removed successfully!")
        time.sleep(1)
        st.experimental_rerun()
