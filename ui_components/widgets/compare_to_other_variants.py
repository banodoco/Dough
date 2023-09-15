import json
import streamlit as st
import uuid
from typing import List
from utils.data_repo.data_repo import DataRepo


def compare_to_other_variants(timing_details, project_uuid, data_repo, stage="Motion"):

    main_video = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    alternative_videos = ['https://www.youtube.com/watch?v=kib6uXQsxBA','https://www.youtube.com/watch?v=ehWD5kG4xws','https://www.youtube.com/watch?v=zkTf0LmDqKI']
    primary_video_variant_index = 0

    timing = data_repo.get_timing_from_uuid(
                    st.session_state['current_frame_uuid'])
    variants = timing.alternative_images_list
    mainimages1, mainimages2 = st.columns([1, 1])
    aboveimage1, aboveimage2, aboveimage3 = st.columns([1, 0.25, 0.75])
    
    which_variant = None

    with aboveimage1:
        st.info(f"Current variant = {timing_details[st.session_state['current_frame_index'] - 1].primary_variant_index + 1}")

    with aboveimage2:
        show_more_than_10_variants = st.checkbox("Show >10 variants", key="show_more_than_10_variants")

    with aboveimage3:
        number_of_variants = len(alternative_videos) if stage == "Motion" else len(variants)

        if show_more_than_10_variants is True:
            current_variant = primary_video_variant_index if stage == "Motion" else int(
                timing_details[st.session_state['current_frame_index'] - 1].primary_variant_index)
            which_variant = st.radio(f'Main variant = {current_variant + 1}', range(1, 
                number_of_variants + 1), index=number_of_variants-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_index']}")
        else:
            last_ten_variants = range(
                max(1, number_of_variants - 10), number_of_variants + 1)
            current_variant = primary_video_variant_index if stage == "Motion" else int(
                timing_details[st.session_state['current_frame_index'] - 1].primary_variant_index)
            which_variant = st.radio(f'Main variant = {current_variant + 1}', last_ten_variants, index=len(
                last_ten_variants)-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_index']}")

    with mainimages1:
        project_settings = data_repo.get_project_setting(project_uuid)
        st.success("**Main variant**")
        if stage == "Motion":
            st.video(main_video, format='mp4', start_time=0)
        else:
            if len(timing_details[st.session_state['current_frame_index'] - 1].alternative_images_list):
                st.image(timing_details[st.session_state['current_frame_index'] - 1].primary_image_location,
                            use_column_width=True)
            else:
                st.error("No variants found for this frame")

    with mainimages2:
        if stage == "Motion":
            if which_variant - 1 == current_variant:
                st.success("**Main variant**")
            else:
                st.info(f"**Variant #{which_variant}**")
            
            st.video(alternative_videos[which_variant- 1], format='mp4', start_time=0)
        else:
            if len(timing_details[st.session_state['current_frame_index'] - 1].alternative_images_list):
                if which_variant - 1 == current_variant:
                    st.success("**Main variant**")
                else:
                    st.info(f"**Variant #{which_variant}**")
                
                st.image(variants[which_variant- 1].location,
                            use_column_width=True)

        if which_variant- 1 != current_variant:
            if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image"):
                promote_image_variant(
                    st.session_state['current_frame_uuid'], which_variant - 1)
                time.sleep(0.5)
                st.experimental_rerun()