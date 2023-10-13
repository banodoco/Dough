import time
import json
import streamlit as st
import uuid
from typing import List
from ui_components.constants import CreativeProcessType
from ui_components.methods.common_methods import promote_image_variant
from utils.data_repo.data_repo import DataRepo


def variant_comparison_element(timing_uuid, stage=CreativeProcessType.MOTION.value):
    data_repo = DataRepo()

    timing = data_repo.get_timing_from_uuid(timing_uuid)
    variants = timing.alternative_images_list
    mainimages1, mainimages2 = st.columns([1, 1])
    aboveimage1, aboveimage2, aboveimage3 = st.columns([1, 0.25, 0.75])
    
    which_variant = 1
    number_of_variants = 0

    with aboveimage1:
        st.info(f"Current variant = {timing.primary_variant_index + 1}")

    with aboveimage2:
        show_more_than_10_variants = st.checkbox("Show >10 variants", key="show_more_than_10_variants")

    with aboveimage3:
        number_of_variants = len(timing.interpolated_clip_list) if stage == CreativeProcessType.MOTION.value else len(variants)

        if number_of_variants:
            if show_more_than_10_variants is True:
                current_variant = timing.primary_interpolated_video_index if stage == CreativeProcessType.MOTION.value else int(
                    timing.primary_variant_index)
                which_variant = st.radio(f'Main variant = {current_variant + 1}', range(1, 
                    number_of_variants + 1), index=number_of_variants-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_index']}")
            else:
                last_ten_variants = range(
                    max(1, number_of_variants - 10), number_of_variants + 1)
                current_variant = timing.primary_interpolated_video_index if stage == CreativeProcessType.MOTION.value else int(
                    timing.primary_variant_index)
                which_variant = st.radio(f'Main variant = {current_variant + 1}', last_ten_variants, index=len(
                    last_ten_variants)-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_index']}")

    with mainimages1:
        st.success("**Main variant**")
        if stage == CreativeProcessType.MOTION.value:
            st.video(timing.timed_clip.location, format='mp4', start_time=0) if timing.timed_clip else st.error("No video present")
        else:
            if len(timing.alternative_images_list):
                st.image(timing.primary_image_location, use_column_width=True)
            else:
                st.error("No variants found for this frame")

    with mainimages2:
        if stage == CreativeProcessType.MOTION.value:
            if number_of_variants:
                if not (timing.interpolated_clip_list and len(timing.interpolated_clip_list)):
                    st.error("No variant for this frame")
                    
                if which_variant - 1 == current_variant:
                    st.success("**Main variant**")
                else:
                    st.info(f"**Variant #{which_variant}**")
                
                st.video(timing.interpolated_clip_list[which_variant - 1].location, format='mp4', start_time=0) if \
                    (timing.interpolated_clip_list and len(timing.interpolated_clip_list)) else st.error("No video present")
            else:
                st.error("No variants found for this frame")
        else:
            if len(timing.alternative_images_list):
                if which_variant - 1 == current_variant:
                    st.success("**Main variant**")
                else:
                    st.info(f"**Variant #{which_variant}**")
                
                st.image(variants[which_variant - 1].location,
                            use_column_width=True)

        if number_of_variants:
            if which_variant - 1 != current_variant:
                if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image"):
                    if stage == CreativeProcessType.MOTION.value:
                        promote_video_variant(timing.uuid, which_variant - 1)
                    else:
                        promote_image_variant(timing.uuid, which_variant - 1)
                    time.sleep(0.5)
                    st.rerun()