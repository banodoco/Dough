import os
import random
import string
import tempfile
from typing import List
import ffmpeg
import streamlit as st
import uuid
from moviepy.editor import concatenate_videoclips, TextClip, VideoFileClip, vfx, AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from backend.models import InternalFileObject
from shared.constants import InternalFileTag
from shared.file_upload.s3 import is_s3_image_url
from ui_components.constants import VideoQuality
from ui_components.models import InternalFrameTimingObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.interpolator import VideoInterpolator
from utils.media_processor.video import VideoProcessor


# returns the timed_clip, which is the interpolated video with correct length
def create_or_get_single_preview_video(timing_uuid):
    from ui_components.methods.file_methods import generate_temp_file, save_or_host_file_bytes
    from ui_components.methods.common_methods import get_audio_bytes_for_slice

    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_details: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)

    if not timing.interpolated_clip:
        data_repo.update_specific_timing(timing_uuid, interpolation_steps=3)
        interpolated_video: InternalFileObject = VideoInterpolator.video_through_frame_interpolation(
            timing_uuid)
        data_repo.update_specific_timing(
            timing_uuid, interpolated_clip_id=interpolated_video.uuid)

    if not timing.timed_clip:
        timing = data_repo.get_timing_from_uuid(timing_uuid)
        
        temp_video_file = None
        if timing.interpolated_clip.hosted_url:
            temp_video_file = generate_temp_file(timing.interpolated_clip.hosted_url, '.mp4')

        file_path = temp_video_file.name if temp_video_file else timing.interpolated_clip.local_path
        clip = VideoFileClip(file_path)
            
        number_text = TextClip(str(timing.aux_frame_index),
                               fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(
            number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(
            ('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position(
            (number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)
        clip_with_number = CompositeVideoClip([clip, number_background, number_text])

        clip_with_number.write_videofile(filename=file_path, codec='libx264', audio_codec='aac')

        if temp_video_file:
            video_bytes = None
            with open(file_path, 'rb') as f:
                video_bytes = f.read()

            hosted_url = save_or_host_file_bytes(video_bytes, timing.interpolated_clip.local_path)
            data_repo.update_file(timing.interpolated_clip.uuid, hosted_url=hosted_url)

            os.remove(temp_video_file.name)

        # timed_clip has the correct length (equal to the time difference between the current and the next frame)
        # which the interpolated video may or maynot have
        clip_duration = calculate_desired_duration_of_individual_clip(timing_uuid)
        data_repo.update_specific_timing(timing_uuid, clip_duration=clip_duration)

        timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
        output_video = update_speed_of_video_clip(timing.interpolated_clip, timing_uuid)
        data_repo.update_specific_timing(timing_uuid, timed_clip_id=output_video.uuid)

    # adding audio if the audio file is present
    if project_details.audio:
        audio_bytes = get_audio_bytes_for_slice(timing_uuid)
        add_audio_to_video_slice(timing.timed_clip, audio_bytes)

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    return timing.timed_clip


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



def render_video(final_video_name, project_uuid, quality, file_tag=InternalFileTag.GENERATED_VIDEO.value):
    from ui_components.methods.common_methods import update_clip_duration_of_all_timing_frames
    from ui_components.methods.file_methods import convert_bytes_to_file, generate_temp_file

    data_repo = DataRepo()

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)

    update_clip_duration_of_all_timing_frames(project_uuid)

    total_number_of_videos = len(timing_details) - 1

    for i in range(0, total_number_of_videos):
        index_of_current_item = i
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            project_uuid, i)

        timing = timing_details[i]
        if quality == VideoQuality.HIGH.value:
            data_repo.update_specific_timing(
                current_timing.uuid, timed_clip_id=None)
            interpolation_steps = VideoInterpolator.calculate_dynamic_interpolations_steps(
                timing_details[index_of_current_item].clip_duration)
            if not timing.interpolation_steps or timing.interpolation_steps < interpolation_steps:
                data_repo.update_specific_timing(
                    current_timing.uuid, interpolation_steps=interpolation_steps, interpolated_clip_id=None)
        else:
            if not timing.interpolation_steps or timing.interpolation_steps < 3:
                data_repo.update_specific_timing(
                    current_timing.uuid, interpolation_steps=3)

        if not timing.interpolated_clip:
            next_timing = data_repo.get_next_timing(current_timing.uuid)
            video_bytes = VideoInterpolator.create_interpolated_clip(
                img_location_list=[current_timing.source_image.location, next_timing.source_image.location],
                interpolation_steps=current_timing.interpolation_steps
            )

            file_location = "videos/" + current_timing.project.name + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".mp4"
            video_file = convert_bytes_to_file(
                file_location,
                "video/mp4",
                video_bytes,
                current_timing.project.uuid
            )

            data_repo.update_specific_timing(
                current_timing.uuid, interpolated_clip_id=video_file.uuid)

    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.project.uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)
    total_number_of_videos = len(timing_details) - 2

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        timing = timing_details[index_of_current_item]
        current_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(
            timing.project.uuid, index_of_current_item)
        if index_of_current_item <= total_number_of_videos:
            if not current_timing.timed_clip:
                desired_duration = current_timing.clip_duration
                location_of_input_video_file = current_timing.interpolated_clip

                output_video = update_speed_of_video_clip(
                    location_of_input_video_file, timing.uuid)

                if quality == VideoQuality.PREVIEW.value:
                    print("")
                    '''
                    clip = VideoFileClip(location_of_output_video)

                    number_text = TextClip(str(index_of_current_item), fontsize=24, color='white')
                    number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(number_text.w + 10, number_text.h + 10))
                    number_background = number_background.set_position(('right', 'bottom')).set_duration(clip.duration)
                    number_text = number_text.set_position((number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)

                    clip_with_number = CompositeVideoClip([clip, number_background, number_text])

                    # remove existing preview video
                    os.remove(location_of_output_video)
                    clip_with_number.write_videofile(location_of_output_video, codec='libx264', bitrate='3000k')
                    '''

                data_repo.update_specific_timing(
                    current_timing.uuid, timed_clip_id=output_video.uuid)

    video_list = []
    temp_file_list = []

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        timing.project.uuid)

    # TODO: CORRECT-CODE
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
        output_video_file,
        "video/mp4",
        video_bytes,
        project_uuid,
        filename=final_video_name,
        tag=file_tag
    )

    for file in temp_file_list:
        os.remove(file.name)
