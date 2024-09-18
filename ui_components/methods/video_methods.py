import os
import random
import string
import subprocess
import sys
import tempfile
import time
from typing import List
import uuid
import json
import ffmpeg
import pkg_resources
import streamlit as st
from moviepy.editor import (
    concatenate_videoclips,
    concatenate_audioclips,
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
)
from pydub import AudioSegment

from shared.constants import (
    GPU_INFERENCE_ENABLED_KEY,
    QUEUE_INFERENCE_QUERIES,
    FileTransformationType,
    InferenceLogTag,
    InferenceType,
    InternalFileTag,
    ConfigManager
)
from shared.file_upload.s3 import is_s3_image_url
from ui_components.constants import ShotMetaData
from ui_components.methods.animation_style_methods import get_generation_settings_from_log
from ui_components.methods.file_methods import save_or_host_file_bytes
from ui_components.models import InternalFileObject, InternalFrameTimingObject, InternalShotObject
from utils.common_utils import padded_integer
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.video import VideoProcessor
from utils.ml_processor.constants import ML_MODEL
from utils.ml_processor.ml_interface import get_ml_client


def upscale_video(file_uuid, styling_model, upscale_factor, promote_to_main_variant):
    from ui_components.methods.common_methods import process_inference_output
    from shared.constants import QUEUE_INFERENCE_QUERIES

    data_repo = DataRepo()
    config_manager = ConfigManager()
    gpu_enabled = config_manager.get(GPU_INFERENCE_ENABLED_KEY, False)
    video_file: InternalFileObject = data_repo.get_file_from_uuid(uuid=file_uuid)
    shot_meta_data, data_type = get_generation_settings_from_log(video_file.inference_log.uuid)

    # hacky fix to prevent conflicting opencv versions
    if gpu_enabled:
        try:
            pkg_resources.require("opencv-python-headless==4.8.0.74")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "opencv-python-headless==4.8.0.74",
                        "ffmpeg-python",
                    ]
                )
                print("Packages installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing packages: {e}")

    query_obj = MLQueryObject(
        timing_uuid=None,
        model_uuid=None,
        guidance_scale=8,
        seed=-1,
        num_inference_steps=25,
        strength=0.5,
        adapter_type=None,
        prompt="upscale",
        negative_prompt="",
        height=512,  # these are dummy values, not used
        width=512,
        data={
            "model": styling_model,
            "upscale_factor": upscale_factor,
            "shot_data": {
                ShotMetaData.MOTION_DATA.value: json.dumps(shot_meta_data)
            },  # adding parent's generation detail in the upscaled version (change the key according to the model)
        },
        file_data={"video_file": {"uuid": video_file.uuid, "dest": "input/"}},
        relation_data=json.dumps(
            [
                {
                    "type": "file",
                    "id": video_file.uuid,
                    "transformation_type": FileTransformationType.UPSCALE.value,
                }
            ]
        ),
    )

    ml_client = get_ml_client()
    output, log = ml_client.predict_model_output_standardized(
        ML_MODEL.video_upscaler, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
    )
    if log:
        # TODO: this is a hackish way of adding the input images of the base shot in the upscaled version
        # change it so that the origin data is added in the log before it is created
        origin_log = data_repo.get_inference_log_from_uuid(video_file.inference_log.uuid)
        shot_data = json.loads(origin_log.input_params)
        file_uuid_list = (
            shot_data.get("origin_data", json.dumps({})).get("settings", {}).get("file_uuid_list", [])
        )

        inference_data = {
            "inference_type": InferenceType.FRAME_INTERPOLATION.value,
            "output": output,
            "log_uuid": log.uuid,
            "settings": {
                "promote_to_main_variant": promote_to_main_variant,
                "file_uuid_list": file_uuid_list,
            },
            "shot_uuid": str(video_file.origin_shot_uuid),
            "inference_tag": InferenceLogTag.UPSCALED_VIDEO.value,
        }

        process_inference_output(**inference_data)


def update_speed_of_video_clip(video_file: InternalFileObject, duration) -> InternalFileObject:
    from ui_components.methods.file_methods import generate_temp_file, convert_bytes_to_file

    temp_video_file = None
    if video_file.hosted_url and is_s3_image_url(video_file.hosted_url):
        temp_video_file = generate_temp_file(video_file.hosted_url, ".mp4")

    location_of_video = temp_video_file.name if temp_video_file else video_file.local_path

    new_file_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=16)) + ".mp4"
    new_file_location = (
        "videos/" + str(video_file.project.uuid) + "/assets/videos/completed/" + str(new_file_name)
    )

    video_bytes = VideoProcessor.update_video_speed(location_of_video, duration)

    hosted_url = save_or_host_file_bytes(video_bytes, new_file_location, ".mp4")
    data_repo = DataRepo()
    if hosted_url:
        data_repo.update_file(video_file.uuid, hosted_url=hosted_url)
    else:
        data_repo.update_file(video_file.uuid, local_path=new_file_location)

    if temp_video_file:
        os.remove(temp_video_file.name)

    return video_file


def add_audio_to_video_slice(video_file, audio_bytes):
    video_location = video_file.local_path
    # Save the audio bytes to a temporary file
    audio_file = "temp_audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes.getvalue())

    # Create an input video stream
    video_stream = ffmpeg.input(video_location)

    # Create an input audio stream
    audio_stream = ffmpeg.input(audio_file)

    # Add the audio stream to the video stream
    output_stream = ffmpeg.output(
        video_stream,
        audio_stream,
        "output_with_audio.mp4",
        vcodec="copy",
        acodec="aac",
        strict="experimental",
    )

    # Run the ffmpeg command
    output_stream.run()

    # Remove the original video file and the temporary audio file
    os.remove(video_location)
    os.remove(audio_file)

    # TODO: handle online update in this case
    # Rename the output file to have the same name as the original video file
    os.rename("output_with_audio.mp4", video_location)


def sync_audio_and_duration(video_file: InternalFileObject, shot_uuid, audio_sync_required=False):
    """
    audio_sync_required: this ensures that entire video clip is filled with proper audio
    """
    from ui_components.methods.file_methods import convert_bytes_to_file, generate_temp_file

    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    shot_list = data_repo.get_shot_list(shot.project.uuid)
    project_settings = data_repo.get_project_setting(shot.project.uuid)

    # --------- update video duration
    # output_video = update_speed_of_video_clip(video_file, shot.duration)
    output_video = video_file

    # --------- add audio
    temp_file_list = []
    temp_audio_file = None

    if not project_settings.audio:
        print("no audio to sync")
        if not audio_sync_required:
            return output_video
        else:
            return None

    if "http" in project_settings.audio.location:
        temp_audio_file = generate_temp_file(project_settings.audio.location, ".mp4")
        temp_file_list.append(temp_audio_file)

    audio_location = temp_audio_file.name if temp_audio_file else project_settings.audio.location

    temp_video_file = None
    if output_video.hosted_url:
        temp_video_file = generate_temp_file(output_video.hosted_url, ".mp4")
        temp_file_list.append(temp_video_file)

    video_location = temp_video_file.name if temp_video_file else output_video.local_path

    audio_clip = AudioFileClip(audio_location)
    video_clip = VideoFileClip(video_location)

    start_timestamp = 0
    for i in range(1, shot.shot_idx):
        start_timestamp += round(shot_list[i - 1].duration, 2)

    trimmed_audio_clip = None
    audio_len_overlap = 0  # length of audio that can be added to the video clip
    if audio_clip.duration >= start_timestamp:
        audio_len_overlap = round(min(video_clip.duration, audio_clip.duration - start_timestamp), 2)

    # audio doesn't fit the video clip
    if audio_len_overlap < video_clip.duration and audio_sync_required:
        return None

    if audio_len_overlap:
        trimmed_audio_clip = audio_clip.subclip(start_timestamp, start_timestamp + audio_len_overlap)
        trimmed_audio_clip_duration = round(trimmed_audio_clip.duration, 2)
        if trimmed_audio_clip_duration < video_clip.duration:
            video_with_sound = video_clip.subclip(0, trimmed_audio_clip_duration)
            video_with_sound = video_with_sound.copy()
            video_without_sound = video_clip.subclip(trimmed_audio_clip_duration)
            video_without_sound = video_without_sound.copy()
            video_with_sound = video_with_sound.set_audio(trimmed_audio_clip)
            video_clip = concatenate_videoclips([video_with_sound, video_without_sound])
        else:
            video_clip = video_clip.set_audio(trimmed_audio_clip)
    else:
        for file in temp_file_list:
            os.remove(file.name)

        return output_video

    # writing the video to the temp file
    output_temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_clip.write_videofile(output_temp_video_file.name, codec="libx264", audio=True, audio_codec="aac")

    output_temp_video_file.close()
    video_bytes = None
    with open(output_temp_video_file.name, "rb") as f:
        video_bytes = f.read()

    temp_file_list.append(output_temp_video_file)
    unique_name = str(uuid.uuid4())
    output_video_file = f"videos/{shot.project.uuid}/assets/videos/completed/{unique_name}.mp4"
    hosted_url = save_or_host_file_bytes(video_bytes, output_video_file, ext=".mp4")
    if hosted_url:
        data_repo.update_file(output_video.uuid, hosted_url=hosted_url)
    else:
        data_repo.update_file(output_video.uuid, local_path=output_video_file)

    for file in temp_file_list:
        os.remove(file.name)

    output_video = data_repo.get_file_from_uuid(output_video.uuid)
    _ = data_repo.get_shot_list(shot.project.uuid, invalidate_cache=True)
    return output_video


def render_video(final_video_name, project_uuid, file_tag=InternalFileTag.GENERATED_VIDEO.value):
    """
    combines the main variant of all the shots to form the final video. no processing happens in this, only
    simple combination
    """
    from ui_components.methods.file_methods import convert_bytes_to_file, generate_temp_file

    data_repo = DataRepo()

    if not final_video_name:
        st.error("Please enter a video name")
        time.sleep(0.3)
        return False

    video_list = []
    temp_file_list = []

    # combining all the main_clip of shots in finalclip, and keeping track of temp video files
    # in temp_file_list
    shot_list: List[InternalShotObject] = data_repo.get_shot_list(project_uuid)
    for shot in shot_list:
        if not shot.main_clip:
            st.error("Please generate all videos")
            time.sleep(0.7)
            return False

        shot_video = sync_audio_and_duration(shot.main_clip, shot.uuid, audio_sync_required=False)
        if not shot_video:
            st.error("Audio sync failed. Length mismatch")
            time.sleep(0.7)
            return False

        data_repo.update_shot(uuid=shot.uuid, main_clip_id=shot_video.uuid)
        data_repo.add_interpolated_clip(shot.uuid, interpolated_clip_id=shot_video.uuid)

        temp_video_file = None
        if shot_video.hosted_url:
            temp_video_file = generate_temp_file(shot_video.hosted_url, ".mp4")
            temp_file_list.append(temp_video_file)

        file_path = temp_video_file.name if temp_video_file else shot_video.local_path
        video_list.append(file_path)

    finalclip = concatenate_videoclips([VideoFileClip(v) for v in video_list])

    # writing the video to the temp file
    output_video_file = f"videos/{project_uuid}/assets/videos/2_completed/{final_video_name}.mp4"
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    finalclip.write_videofile(
        temp_video_file.name,
        fps=60,  # or 60 if your original video is 60fps
        audio_bitrate="128k",
        bitrate="5000k",
        codec="libx264",
        audio_codec="aac",
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
        tag=file_tag,
    )

    for file in temp_file_list:
        os.remove(file.name)

    return True
