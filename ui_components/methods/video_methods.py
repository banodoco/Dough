import os
import random
import string
import tempfile
import time
from typing import List
import ffmpeg
import streamlit as st
import uuid
from moviepy.editor import concatenate_videoclips, TextClip, VideoFileClip, vfx, AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from backend.models import InternalFileObject
from shared.constants import AnimationToolType, InferenceType, InternalFileTag
from shared.file_upload.s3 import is_s3_image_url
from ui_components.constants import VideoQuality
from ui_components.methods.common_methods import process_inference_output
from ui_components.methods.file_methods import convert_bytes_to_file, generate_temp_file
from ui_components.models import InternalFrameTimingObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.interpolator import VideoInterpolator
from utils.media_processor.video import VideoProcessor


# NOTE: interpolated_clip_uuid signals which clip to promote to timed clip (this is the main variant)
# this function returns the 'single' preview_clip, which is basically timed_clip with the frame number
def create_or_get_single_preview_video(timing_uuid, interpolated_clip_uuid=None):
    from ui_components.methods.file_methods import generate_temp_file
    from ui_components.methods.common_methods import get_audio_bytes_for_slice

    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_details: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    if not len(timing.interpolated_clip_list):
        timing.interpolation_steps = 3
        next_timing = data_repo.get_next_timing(timing.uuid)
        img_list = [timing.source_image.location, next_timing.source_image.location]
        res = VideoInterpolator.video_through_frame_interpolation(img_list, {"interpolation_steps": timing.interpolation_steps})
        
        output_url, log = res[0]

        inference_data = {
            "inference_type": InferenceType.SINGLE_PREVIEW_VIDEO.value,
            "file_location_to_save": "videos/" + timing.project.uuid + "/assets/videos" + (str(uuid.uuid4())) + ".mp4",
            "mime_type": "video/mp4",
            "output": output_url,
            "project_uuid": timing.project.uuid,
            "log_uuid": log.uuid,
            "timing_uuid": timing_uuid
        }

        process_inference_output(**inference_data)
        

    if not timing.timed_clip:
        interpolated_clip = data_repo.get_file_from_uuid(interpolated_clip_uuid) if interpolated_clip_uuid \
                                else timing.interpolated_clip_list[0]
        
        output_video = update_speed_of_video_clip(interpolated_clip, timing_uuid)
        data_repo.update_specific_timing(timing_uuid, timed_clip_id=output_video.uuid)

    if not timing.preview_video:
        timing = data_repo.get_timing_from_uuid(timing_uuid)
        timed_clip = timing.timed_clip
        
        temp_video_file = None
        if timed_clip.hosted_url and is_s3_image_url(timed_clip.hosted_url):
            temp_video_file = generate_temp_file(timed_clip.hosted_url, '.mp4')

        file_path = temp_video_file.name if temp_video_file else timed_clip.local_path
        clip = VideoFileClip(file_path)
        
        if temp_video_file:
            os.remove(temp_video_file.name)

        number_text = TextClip(str(timing.aux_frame_index),
                               fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(
            number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(
            ('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position(
            (number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)
        clip_with_number = CompositeVideoClip([clip, number_background, number_text])

        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
        clip_with_number.write_videofile(filename=temp_output_file.name, codec='libx264', audio_codec='aac')

        if temp_output_file:
            video_bytes = None
            with open(file_path, 'rb') as f:
                video_bytes = f.read()

            preview_video = convert_bytes_to_file(
                file_location_to_save="videos/" + str(timing.project.uuid) + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".png",
                mime_type="video/mp4",
                file_bytes=video_bytes,
                project_uuid=timing.project.uuid,
                inference_log_id=None
            )

            data_repo.update_specific_timing(timing_uuid, preview_video_id=preview_video.uuid)
            os.remove(temp_output_file.name)

        # preview has the correct length (equal to the time difference between the current and the next frame)
        # which the interpolated video may or maynot have
        # clip_duration = calculate_desired_duration_of_individual_clip(timing_uuid)
        # data_repo.update_specific_timing(timing_uuid, clip_duration=clip_duration)

    # adding audio if the audio file is present
    if project_details.audio:
        audio_bytes = get_audio_bytes_for_slice(timing_uuid)
        add_audio_to_video_slice(timing.preview_video, audio_bytes)

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    return timing.preview_video

# this includes all the animation styles [direct morphing, interpolation, image to video]
def create_single_interpolated_clip(timing_uuid, quality, settings={}, variant_count=1):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    next_timing: InternalFrameTimingObject = data_repo.get_next_timing(timing_uuid)

    if not next_timing:
        st.error('This is the last image. Please add more images to create interpolated clip.')
        return None

    if quality == 'full':
        interpolation_steps = VideoInterpolator.calculate_dynamic_interpolations_steps(timing.clip_duration)
    elif quality == 'preview':
        interpolation_steps = 3

    timing.interpolated_steps = interpolation_steps
    img_list = [timing.primary_image.location, next_timing.primary_image.location]
    settings.update(interpolation_steps=timing.interpolation_steps)

    # res is an array of tuples (video_bytes, log)
    res = VideoInterpolator.create_interpolated_clip(
        img_list,
        timing.animation_style,
        settings,
        variant_count
    )

    for (output, log) in res:
        inference_data = {
            "inference_type": InferenceType.FRAME_INTERPOLATION.value,
            "output": output,
            "log_uuid": log.uuid,
            "settings": settings,
            "timing_uuid": timing_uuid
        }

        process_inference_output(**inference_data)


# preview_clips have frame numbers on them. Preview clip is generated from index-2 to index+2 frames
def create_full_preview_video(timing_uuid, speed=1) -> InternalFileObject:
    from ui_components.methods.file_methods import save_or_host_file_bytes, convert_bytes_to_file, generate_temp_file

    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    index_of_item = timing.aux_frame_index

    num_timing_details = len(timing_details)
    clips = []

    temp_file_list = []

    for i in range(index_of_item - 2, index_of_item + 3):
        if i < 0 or i >= num_timing_details-1:
            continue

        primary_variant_location = timing_details[i].primary_image_location
        print(f"primary_variant_location for i={i}: {primary_variant_location}")

        if not primary_variant_location:
            break

        preview_video = create_or_get_single_preview_video(timing_details[i].uuid)

        clip = VideoFileClip(preview_video.location)

        number_text = TextClip(str(i), fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(
            number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(
            ('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position(
            (number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)

        clip_with_number = CompositeVideoClip(
            [clip, number_background, number_text])

        # remove existing preview video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
        temp_file_list.append(temp_file)
        clip_with_number.write_videofile(temp_file.name, codec='libx264', bitrate='3000k')
        video_bytes = None
        with open(temp_file.name, 'rb') as f:
            video_bytes = f.read()

        hosted_url = save_or_host_file_bytes(video_bytes, preview_video.local_path)
        if hosted_url:
            data_repo.update_file(preview_video.uuid, hosted_url=hosted_url)
        
        clips.append(preview_video)

    print(clips)
    video_clips = []

    for v in clips:
        path = v.location
        if 'http' in path:
            temp_file = generate_temp_file(path)
            temp_file_list.append(temp_file)
            path = temp_file.name
        
        video_clips.append(VideoFileClip(path))

    # video_clips = [VideoFileClip(v.location) for v in clips]
    combined_clip = concatenate_videoclips(video_clips)
    output_filename = str(uuid.uuid4()) + ".mp4"
    video_location = f"videos/{timing.project.uuid}/assets/videos/1_final/{output_filename}"

    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
    combined_clip = combined_clip.fx(vfx.speedx, speed)
    combined_clip.write_videofile(temp_output_file.name)

    video_bytes = None
    with open(temp_output_file.name, 'rb') as f:
        video_bytes = f.read()
    
    video_file = convert_bytes_to_file(
        video_location,
        "video/mp4",
        video_bytes,
        timing.project.uuid
    )

    for file in temp_file_list:
        os.remove(file.name)

    return video_file

def update_speed_of_video_clip(video_file: InternalFileObject, timing_uuid) -> InternalFileObject:
    from ui_components.methods.file_methods import generate_temp_file, convert_bytes_to_file

    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    desired_duration = timing.clip_duration
    animation_style = timing.animation_style

    temp_video_file = None
    if video_file.hosted_url and is_s3_image_url(video_file.hosted_url):
        temp_video_file = generate_temp_file(video_file.hosted_url, '.mp4')
    
    location_of_video = temp_video_file.name if temp_video_file else video_file.local_path
    
    new_file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16)) + ".mp4"
    new_file_location = "videos/" + str(timing.project.uuid) + "/assets/videos/1_final/" + str(new_file_name)

    video_bytes = VideoProcessor.update_video_speed(
        location_of_video,
        animation_style,
        desired_duration
    )

    video_file = convert_bytes_to_file(
        new_file_location,
        "video/mp4",
        video_bytes,
        timing.project.uuid
    )

    if temp_video_file:
        os.remove(temp_video_file.name)

    return video_file


def calculate_desired_duration_of_individual_clip(timing_uuid):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    length_of_list = len(timing_details)

    # last frame
    if timing.aux_frame_index == length_of_list - 1:
        time_of_frame = timing.frame_time
        total_duration_of_frame = 0.0   # can be changed
    else:
        time_of_frame = timing.frame_time
        time_of_next_frame = data_repo.get_next_timing(timing_uuid).frame_time
        total_duration_of_frame = float(
            time_of_next_frame) - float(time_of_frame)

    return total_duration_of_frame


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


# final video rendering of all the frames involved (it bascially combines all the timed clips)
def render_video(final_video_name, project_uuid, quality, file_tag=InternalFileTag.GENERATED_VIDEO.value):
    from ui_components.methods.common_methods import update_clip_duration_of_all_timing_frames
    from ui_components.methods.file_methods import convert_bytes_to_file, generate_temp_file

    data_repo = DataRepo()

    if not final_video_name:
        st.error("Please enter a video name")
        return

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)

    update_clip_duration_of_all_timing_frames(project_uuid)

    total_number_of_videos = len(timing_details) - 1

    # creating timed clip for every frame
    for i in range(0, total_number_of_videos):
        index_of_current_item = i
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            project_uuid, i)

        timing = timing_details[i]

        # updating the interpolation steps
        if quality == VideoQuality.HIGH.value:
            # data_repo.update_specific_timing(current_timing.uuid, timed_clip_id=None)
            interpolation_steps = VideoInterpolator.calculate_dynamic_interpolations_steps(
                timing_details[index_of_current_item].clip_duration)
            timing.interpolation_steps = interpolation_steps
        else:
            if not timing.interpolation_steps or timing.interpolation_steps < 3:
                data_repo.update_specific_timing(
                    current_timing.uuid, interpolation_steps=3)

        # creating timed clips if not already present
        if not timing.timed_clip:
            # creating an interpolated clip if not already present
            if not len(timing.interpolated_clip_list):
                next_timing = data_repo.get_next_timing(current_timing.uuid)
                settings = {
                    "animation_tool": current_timing.animation_tool,
                    "interpolation_steps": current_timing.interpolation_steps
                }

                res = VideoInterpolator.create_interpolated_clip(
                    img_location_list=[current_timing.source_image.location, next_timing.source_image.location],
                    animation_style=current_timing.animation_style,
                    settings=settings,
                    interpolation_steps=current_timing.interpolation_steps
                )

                video_bytes, log = res[0]

                file_location = "videos/" + current_timing.project.name + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".mp4"
                video_file = convert_bytes_to_file(
                    file_location_to_save=file_location,
                    mime_type="video/mp4",
                    file_bytes=video_bytes,
                    project_uuid=current_timing.project.uuid,
                    inference_log_id=log.uuid
                )

                data_repo.add_interpolated_clip(
                    current_timing.uuid, interpolated_clip_id=video_file.uuid)
            else:
                video_file = timing.interpolated_clip_list[0]
            
            # add timed clip
            output_video = update_speed_of_video_clip(video_file, current_timing.uuid)
            data_repo.update_specific_timing(current_timing.uuid, timed_clip_id=output_video.uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(project_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(project_uuid)
    total_number_of_videos = len(timing_details) - 2

    video_list = []
    temp_file_list = []

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    # joining all the timed clips
    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            project_uuid, index_of_current_item)
        if index_of_current_item <= total_number_of_videos:
            temp_video_file = None
            if current_timing.timed_clip.hosted_url:
                temp_video_file = generate_temp_file(current_timing.timed_clip.hosted_url, '.mp4')
                temp_file_list.append(temp_video_file)

            file_path = temp_video_file.name if temp_video_file else current_timing.timed_clip.local_path
        
            video_list.append(file_path)

    video_clip_list = [VideoFileClip(v) for v in video_list]
    finalclip = concatenate_videoclips(video_clip_list)

    output_video_file = f"videos/{timing.project.uuid}/assets/videos/2_completed/{final_video_name}.mp4"
    if project_settings.audio:
        temp_audio_file = None
        if 'http' in project_settings.audio.location:
            temp_audio_file = generate_temp_file(project_settings.audio.location, '.mp4')
            temp_file_list.append(temp_audio_file)

        audio_location = temp_audio_file.name if temp_audio_file else project_settings.audio.location
        
        audio_clip = AudioFileClip(audio_location)
        finalclip = finalclip.set_audio(audio_clip)

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
