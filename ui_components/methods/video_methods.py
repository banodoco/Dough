import os
import random
import string
import tempfile
import time
from typing import List
import ffmpeg
import streamlit as st
from moviepy.editor import concatenate_videoclips, VideoFileClip, AudioFileClip

from backend.models import InternalFileObject
from shared.constants import InferenceType, InternalFileTag
from shared.file_upload.s3 import is_s3_image_url
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.interpolator import VideoInterpolator
from utils.media_processor.video import VideoProcessor


def create_single_interpolated_clip(shot_uuid, quality, settings={}, variant_count=1):
    '''
    - this includes all the animation styles [direct morphing, interpolation, image to video]
    - this stores the newly created video in the interpolated_clip_list and promotes them to
    timed_clip (if it's not already present)
    '''

    from ui_components.methods.common_methods import process_inference_output
    from shared.constants import QUEUE_INFERENCE_QUERIES

    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    if quality == 'full':
        interpolation_steps = VideoInterpolator.calculate_dynamic_interpolations_steps(shot.duration)
    elif quality == 'preview':
        interpolation_steps = 3

    img_list = [t.primary_image.location for t in timing_list]
    settings.update(interpolation_steps=interpolation_steps)

    # res is an array of tuples (video_bytes, log)
    res = VideoInterpolator.create_interpolated_clip(
        img_list,
        settings['animation_style'],
        settings,
        variant_count,
        QUEUE_INFERENCE_QUERIES
    )

    for (output, log) in res:
        inference_data = {
            "inference_type": InferenceType.FRAME_INTERPOLATION.value,
            "output": output,
            "log_uuid": log.uuid,
            "settings": settings,
            "shot_uuid": str(shot_uuid)
        }

        process_inference_output(**inference_data)

def update_speed_of_video_clip(video_file: InternalFileObject, duration) -> InternalFileObject:
    from ui_components.methods.file_methods import generate_temp_file, convert_bytes_to_file

    temp_video_file = None
    if video_file.hosted_url and is_s3_image_url(video_file.hosted_url):
        temp_video_file = generate_temp_file(video_file.hosted_url, '.mp4')
    
    location_of_video = temp_video_file.name if temp_video_file else video_file.local_path
    
    new_file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16)) + ".mp4"
    new_file_location = "videos/" + str(video_file.project.uuid) + "/assets/videos/1_final/" + str(new_file_name)

    video_bytes = VideoProcessor.update_video_speed(
        location_of_video,
        duration
    )

    video_file = convert_bytes_to_file(
        new_file_location,
        "video/mp4",
        video_bytes,
        video_file.project.uuid
    )

    if temp_video_file:
        os.remove(temp_video_file.name)

    return video_file

def add_audio_to_video_slice(video_file, audio_bytes):
    video_location = video_file.local_path
    # Save the audio bytes to a temporary file
    audio_file = "temp_audio.wav"
    with open(audio_file, 'wb') as f:
        f.write(audio_bytes.getvalue())

    # Create an input video stream
    video_stream = ffmpeg.input(video_location)

    # Create an input audio stream
    audio_stream = ffmpeg.input(audio_file)

    # Add the audio stream to the video stream
    output_stream = ffmpeg.output(video_stream, audio_stream, "output_with_audio.mp4",
                                  vcodec='copy', acodec='aac', strict='experimental')

    # Run the ffmpeg command
    output_stream.run()

    # Remove the original video file and the temporary audio file
    os.remove(video_location)
    os.remove(audio_file)

    # TODO: handle online update in this case
    # Rename the output file to have the same name as the original video file
    os.rename("output_with_audio.mp4", video_location)


def render_video(final_video_name, project_uuid, file_tag=InternalFileTag.GENERATED_VIDEO.value):
    '''
    combines the main variant of all the shots to form the final video. no processing happens in this, only
    simple combination
    '''
    from ui_components.methods.file_methods import convert_bytes_to_file, generate_temp_file

    data_repo = DataRepo()

    if not final_video_name:
        st.error("Please enter a video name")
        time.sleep(0.3)
        return

    video_list = []
    temp_file_list = []

    # combining all the main_clip of shots in finalclip, and keeping track of temp video files
    # in temp_file_list
    shot_list: List[InternalShotObject] = data_repo.get_shot_list_from_project(project_uuid)
    for shot in shot_list:
        if not shot.main_clip:
            st.error("Please generate all videos")
            time.sleep(0.3)
            return
        
        temp_video_file = None
        if shot.main_clip.hosted_url:
            temp_video_file = generate_temp_file(shot.main_clip.hosted_url, '.mp4')
            temp_file_list.append(temp_video_file)

        file_path = temp_video_file.name if temp_video_file else shot.main_clip.local_path
        video_list.append(file_path)

    finalclip = concatenate_videoclips([VideoFileClip(v) for v in video_list])

    # attaching audio to finalclip
    project_settings = data_repo.get_project_settings_from_uuid(project_uuid)
    output_video_file = f"videos/{project_uuid}/assets/videos/2_completed/{final_video_name}.mp4"
    if project_settings.audio:
        temp_audio_file = None
        if 'http' in project_settings.audio.location:
            temp_audio_file = generate_temp_file(project_settings.audio.location, '.mp4')
            temp_file_list.append(temp_audio_file)

        audio_location = temp_audio_file.name if temp_audio_file else project_settings.audio.location
        
        audio_clip = AudioFileClip(audio_location)
        finalclip = finalclip.set_audio(audio_clip)

    # writing the video to the temp file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    finalclip.write_videofile(
        temp_video_file.name,
        fps=60,  # or 60 if your original video is 60fps
        audio_bitrate="128k",
        bitrate="5000k",
        codec="libx264",
        audio_codec="aac"
    )

    temp_video_file.close()
    video_bytes = None
    with open(temp_video_file.name, "rb") as f:
        video_bytes = f.read()

    _ = convert_bytes_to_file(
        file_location_to_save=output_video_file,
        mime_type="video/mp4",
        file_bytes=video_bytes,
        project_uuid=project_uuid,
        inference_log_id=None,
        filename=final_video_name,
        tag=file_tag
    )

    for file in temp_file_list:
        os.remove(file.name)
