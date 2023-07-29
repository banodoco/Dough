from io import BytesIO
import tempfile
import uuid
import requests
import streamlit as st
from shared.constants import SERVER, InternalFileType, ServerType
from ui_components.constants import AUDIO_FILE
from ui_components.models import InternalFileObject
from utils.common_methods import create_working_assets, get_current_user_uuid, save_or_host_file
from utils.data_repo.data_repo import DataRepo
from utils.media_processor.video import resize_video
from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import os

import utils.local_storage.local_storage as local_storage


def new_project_page():
    data_repo = DataRepo()

    a1, a2 = st.columns(2)
    with a1:
        new_project_name = st.text_input("Project name:", value="")
    with a2:
        st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        width = int(st.selectbox("Select video width:", options=[
                    "512", "683", "704", "768", "896", "1024"], key="video_width"))

    with b2:
        height = int(st.selectbox("Select video height:", options=[
                     "512", "683", "704", "768", "1024"], key="video_height"))
    with b3:
        st.info("We recommend a small size + then scaling up afterwards.")

    guidance_type = st.radio("Select guidance type:", options=[
                             "Drawing", "Images", "Video"], help="You can always change this later.", key="guidance_type", horizontal=True)
    audio_options = ["No audio", "Attach new audio"]
    if guidance_type == "Video":
        c1, c2 = st.columns(2)
        with c1:
            uploaded_video = st.file_uploader("Choose a video file:")
        with c2:
            st.write("")
            st.write("")
            if uploaded_video is not None:
                audio_options.append("Keep audio from original video")
            st.info("This video will be resized to match the dimensions above.")
        if uploaded_video is not None:
            resize_this_video = st.checkbox(
                "Resize video to match video dimensions above", value=True)
    else:
        uploaded_video = None
        resize_this_video = False

    audio = st.radio("Audio:", audio_options, key="audio", horizontal=True)
    if uploaded_video is None:
        st.info("You can also keep the audio from your original video - just upload the video above and the option will appear.")

    default_animation_style = st.radio("Select default animation style:", options=[
                                       "Interpolation", "Direct Morphing"], help="You can always change this later.", key="default_animation_style", horizontal=True)

    if audio == "Attach new audio":
        d1, d2 = st.columns([4, 5])
        with d1:
            uploaded_audio = st.file_uploader("Choose a audio file:")
        with d2:
            st.write("")
            st.write("")
            st.info(
                "Make sure that this audio is around the same length as your video.")

    st.write("")
    if st.button("Create New Project"):
        new_project_name = new_project_name.replace(" ", "_")
        create_working_assets(new_project_name)

        current_user_uuid = get_current_user_uuid()
        new_project = data_repo.create_project(name=new_project_name, user_id=current_user_uuid)
        
        data_repo.update_project_setting(new_project.uuid, width=width)
        data_repo.update_project_setting(new_project.uuid, height=height)
        data_repo.update_project_setting(new_project.uuid, guidance_type=guidance_type)
        data_repo.update_project_setting(new_project.uuid, default_animation_style=default_animation_style)

        if uploaded_video is not None:
            video_path = f'videos/{new_project_name}/assets/resources/input_videos/{uploaded_video.name}'
            hosted_url = save_or_host_file(uploaded_video, video_path)
            
            file_data = {
                "name": str(uuid.uuid4()) + ".png",
                "type": InternalFileType.VIDEO.value,
                "project_id": new_project.uuid
            }

            if hosted_url:
                file_data.update({"hosted_url": hosted_url})
            else:
                file_data.update({"local_path": video_path})

            video_file: InternalFileObject = data_repo.create_file(**file_data)
            data_repo.update_project_setting(new_project.uuid, input_video_uuid=video_file.uuid)

            if resize_this_video == True:
                resize_video(input_video_uuid=video_file.uuid, width=width, height=height)

            if audio == "Keep audio from original video":
                audio_file_path = f'videos/{new_project_name}/assets/resources/audio/extracted_audio.mp3'

                video_path = video_file.location
                temp_file = None
                if video_path.contains('http'):
                    response = requests.get(video_path)
                    if not response.ok:
                        raise ValueError(f"Could not download video from URL: {video_path}")

                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
                    temp_file.write(response.content)
                    temp_file.close()
                    video_path = temp_file.name

                clip = VideoFileClip(video_path)
                uploaded_url = None
                if SERVER != ServerType.DEVELOPMENT.value:
                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", mode='wb')
                    clip.audio.write_audiofile(temp_audio_file)
                    file_bytes = BytesIO()

                    with open(temp_file.name, 'rb') as f:
                        file_bytes.write(f.read())
                    file_bytes.seek(0)
                    uploaded_url = data_repo.upload_file(file_bytes)
                    os.remove(temp_audio_file.name)
                    if temp_file:
                        os.remove(temp_file.name)
                else:
                    clip.audio.write_audiofile(audio_file_path)

                file_data = {
                    "name": str(uuid.uuid4()) + ".png",
                    "type": InternalFileType.AUDIO.value,
                    "project_id": new_project.uuid
                }

                if uploaded_url:
                    file_data.update({"hosted_url": uploaded_url})
                else:
                    file_data.update({"local_path": audio_file_path})

                audio_file: InternalFileObject = data_repo.create_file(**file_data)
                data_repo.update_project_setting(new_project.uuid, audio_uuid=audio_file.uuid)

        if audio == "Attach new audio":
            if uploaded_audio is not None:
                uploaded_file_path = f"videos/{new_project_name}/assets/resources/audio/{uploaded_audio.name}"
                hosted_url = save_or_host_file(uploaded_audio, uploaded_file_path)
                
                file_data = {
                    "name": str(uuid.uuid4()) + ".png",
                    "type": InternalFileType.AUDIO.value,
                    "project_id": new_project.uuid 
                }

                if hosted_url:
                    file_data.update({"hosted_url": hosted_url})
                else:
                    file_data.update({"local_path": uploaded_file_path})

                audio_file: InternalFileObject = data_repo.create_file(**file_data)
                data_repo.update_project_setting(new_project.uuid, audio_uuid=audio_file.uuid)

        st.session_state["project_uuid"] = new_project_name
        st.session_state["project_uuid"] = new_project.uuid

        video_list = data_repo.get_all_file_list(file_type=InternalFileType.VIDEO.value)  #[f for f in os.listdir("videos") if not f.startswith('.')]
        
        index = -1
        for video in video_list:
            if video.name == new_project_name:
                index = video_list.index(video)
                break

        st.session_state["index_of_project_name"] = index
        st.session_state["section"] = "Open Project"   
        st.session_state['change_section'] = True      
        st.success("Project created successfully!")
        time.sleep(1)   
        st.experimental_rerun()
