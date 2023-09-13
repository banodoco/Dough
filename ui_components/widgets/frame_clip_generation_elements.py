import uuid
import streamlit as st
from shared.constants import AnimationStyleType
from ui_components.methods.file_methods import convert_bytes_to_file
from ui_components.methods.video_methods import create_full_preview_video, update_speed_of_video_clip
from ui_components.models import InternalFrameTimingObject
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.interpolator import VideoInterpolator


# get audio_bytes of correct duration for a given frame
def current_individual_clip_element(timing_uuid):
    def generate_individual_clip(timing_uuid, quality):
        data_repo = DataRepo()
        timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
        next_timing: InternalFrameTimingObject = data_repo.get_next_timing(timing_uuid)

        if quality == 'full':
            interpolation_steps = VideoInterpolator.calculate_dynamic_interpolations_steps(timing.clip_duration)
        elif quality == 'preview':
            interpolation_steps = 3

        timing.interpolated_steps = interpolation_steps
        img_list = [timing.source_image.location, next_timing.source_image.location]
        settings = {"interpolation_steps": timing.interpolation_steps}
        video_bytes = VideoInterpolator.create_interpolated_clip(
            img_list,
            timing.animation_style,
            settings
        )

        video_location = "videos/" + timing.project.name + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".mp4"
        video = convert_bytes_to_file(
            video_location,
            "video/mp4",
            video_bytes,
            timing.project.uuid
        )

        data_repo.add_interpolated_clip(timing_uuid, interpolated_clip_id=video.uuid, clip_settings=settings)
        output_video = update_speed_of_video_clip(timing.interpolated_clip, timing_uuid)
        data_repo.update_specific_timing(timing_uuid, timed_clip_id=output_video.uuid)
        return output_video
    
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
                    generate_individual_clip(timing.uuid, 'full')
                    st.experimental_rerun()
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
                generate_individual_clip(timing.uuid, 'preview')
                st.experimental_rerun()
        with gen2:
            if st.button("Generate Full Resolution Clip", key=f"generate_full_resolution_video_{idx}"):
                generate_individual_clip(timing.uuid, 'full')
                st.experimental_rerun()


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
        st.experimental_rerun()


def current_preview_video_element(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    idx = timing.aux_frame_index
    st.info("Preview Video in Context:")

    preview_video_1, preview_video_2 = st.columns([2.5, 1])
    
    with preview_video_1:
        if timing.preview_video:
            st.video(timing.preview_video.location)
        else:
            st.error('''
            **----------------------------------------**
            
            ---------
            
            ==================

            **No Preview Video Created Yet**
            
            ==================

            ---------

            **----------------------------------------**
            ''')
    
    with preview_video_2:
        
        if st.button("Generate New Preview Video", key=f"generate_preview_{idx}"):
            preview_video = create_full_preview_video(
                timing.uuid, 1.0)
            data_repo.update_specific_timing(
                timing.uuid, preview_video_id=preview_video.uuid)
            st.experimental_rerun()