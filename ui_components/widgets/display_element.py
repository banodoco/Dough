from typing import Union
import streamlit as st
from ui_components.methods.file_methods import get_file_size
from ui_components.models import InternalFileObject


def individual_video_display_element(file: Union[InternalFileObject, str]):
    file_location = file.location if file and not isinstance(file, str) and file.location else file
    if file_location:
        st.video(file_location, format='mp4', start_time=0) if get_file_size(file_location) < 5 else st.info("Video file too large to display")
    else: 
        st.error("No video present")