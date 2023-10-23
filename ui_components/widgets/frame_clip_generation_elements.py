import uuid
import streamlit as st
from shared.constants import AnimationStyleType
from ui_components.methods.file_methods import convert_bytes_to_file
from ui_components.methods.video_methods import create_full_preview_video, create_single_interpolated_clip, update_speed_of_video_clip
from ui_components.models import InternalFrameTimingObject
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.interpolator import VideoInterpolator


# get audio_bytes of correct duration for a given frame
def current_individual_clip_element(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    idx = timing.aux_frame_index

    st.info(f"Individual Clip for #{idx+1}:")
    if timing.timed_clip:
        st.video(timing.timed_clip.location)
                        
        if timing.interpolation_steps is not None:
            if VideoInterpolator.calculate_dynamic_interpolations_steps(timing.clip_duration) > timing.interpolation_steps:
                st.error("Low Resolution")
                if st.button("Generate Full Resolution Clip", key=f"generate_full_resolution_video_{idx}"):                                    
                    create_single_interpolated_clip(timing.uuid, 'full')
                    st.rerun()
            else:
                st.success("Full Resolution")
    else:
        st.error('''
        **----------------------------------------**
        
        ---------
        
        ==================

        **No Individual Clip Created Yet**
        
        ==================

        ---------

        **----------------------------------------**


        ''')
        gen1, gen2 = st.columns([1, 1])

        with gen1:
            if st.button("Generate Low-Resolution Clip", key=f"generate_preview_video_{idx}"):
                create_single_interpolated_clip(timing.uuid, 'preview')
                st.rerun()
        with gen2:
            if st.button("Generate Full Resolution Clip", key=f"generate_full_resolution_video_{idx}"):
                create_single_interpolated_clip(timing.uuid, 'full')
                st.rerun()


def update_animation_style_element(timing_uuid, horizontal=True):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    idx = timing.aux_frame_index

    animation_styles = AnimationStyleType.value_list()

    if f"animation_style_index_{idx}" not in st.session_state:
        st.session_state[f"animation_style_index_{idx}"] = animation_styles.index(
            timing.animation_style)
        st.session_state[f"animation_style_{idx}"] = timing.animation_style

    st.session_state[f"animation_style_{idx}"] = st.radio(
        "Animation style:", animation_styles, index=st.session_state[f"animation_style_index_{idx}"], key=f"animation_style_radio_{idx}", help="This is for the morph from the current frame to the next one.", horizontal=horizontal)

    if st.session_state[f"animation_style_{idx}"] != timing.animation_style:
        st.session_state[f"animation_style_index_{idx}"] = animation_styles.index(st.session_state[f"animation_style_{idx}"])
        timing.animation_style = st.session_state[f"animation_style_{idx}"]
        st.rerun()