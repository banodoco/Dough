import streamlit as st
from ui_components.common_methods import save_audio_file
from ui_components.models import InternalProjectObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo


def attach_audio_element(project_uuid, expanded):
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(
        uuid=project_uuid)
    project_setting: InternalSettingObject = data_repo.get_project_setting(project_uuid)

    with st.expander("Audio"):
        uploaded_file = st.file_uploader("Attach audio", type=[
                                         "mp3"], help="This will attach this audio when you render a video")
        if st.button("Upload and attach new audio"):
            if uploaded_file:
                save_audio_file(uploaded_file, project_uuid)
                st.experimental_rerun()
            else:
                st.warning('No file selected')

        if project_setting.audio:
            # TODO: store "extracted_audio.mp3" in a constant
            if project_setting.audio.name == "extracted_audio.mp3":
                st.info("You have attached the audio from the video you uploaded.")

            if project_setting.audio.location:
                st.audio(project_setting.audio.location)

