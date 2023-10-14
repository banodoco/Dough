import time
import streamlit as st
from streamlit_image_comparison import image_comparison
from ui_components.constants import CreativeProcessType, WorkflowStageType
from ui_components.methods.common_methods import promote_image_variant, promote_video_variant
from ui_components.methods.video_methods import create_or_get_single_preview_video
from ui_components.widgets.image_carousal import display_image
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


def compare_to_previous_and_next_frame(project_uuid, timing_details):
    data_repo = DataRepo()
    mainimages1, mainimages2, mainimages3 = st.columns([1, 1, 1])

    with mainimages1:
        if st.session_state['current_frame_index'] - 2 >= 0:
            previous_image = data_repo.get_timing_from_frame_number(project_uuid, frame_number=st.session_state['current_frame_index'] - 2)
            st.info(f"Previous image:")
            display_image(
                timing_uuid=previous_image.uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

            if st.button(f"Preview Interpolation From #{st.session_state['current_frame_index']-1} to #{st.session_state['current_frame_index']}", key=f"Preview Interpolation From #{st.session_state['current_frame_index']-1} to #{st.session_state['current_frame_index']}", use_container_width=True):
                prev_frame_timing = data_repo.get_prev_timing(st.session_state['current_frame_uuid'])
                create_or_get_single_preview_video(prev_frame_timing.uuid)
                prev_frame_timing = data_repo.get_timing_from_uuid(prev_frame_timing.uuid)
                if prev_frame_timing.preview_video:
                    st.video(prev_frame_timing.preview_video.location)

    with mainimages2:
        st.success(f"Current image:")
        display_image(
            timing_uuid=st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)

    with mainimages3:
        if st.session_state['current_frame_index'] + 1 <= len(timing_details):
            next_image = data_repo.get_timing_from_frame_number(project_uuid, frame_number=st.session_state['current_frame_index'])
            st.info(f"Next image")
            display_image(timing_uuid=next_image.uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

            if st.button(f"Preview Interpolation From #{st.session_state['current_frame_index']} to #{st.session_state['current_frame_index']+1}", key=f"Preview Interpolation From #{st.session_state['current_frame_index']} to #{st.session_state['current_frame_index']+1}", use_container_width=True):
                create_or_get_single_preview_video(st.session_state['current_frame_uuid'])
                current_frame = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                st.video(current_frame.timed_clip.location)


def compare_to_source_frame(timing_details):
    if timing_details[st.session_state['current_frame_index']- 1].primary_image:
        img2 = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
    else:
        img2 = 'https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'
    
    img1 = timing_details[st.session_state['current_frame_index'] - 1].source_image.location if timing_details[st.session_state['current_frame_index'] - 1].source_image else 'https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'
    
    image_comparison(starting_position=50,
                        img1=img1,
                        img2=img2, make_responsive=False, label1=WorkflowStageType.SOURCE.value, label2=WorkflowStageType.STYLED.value)