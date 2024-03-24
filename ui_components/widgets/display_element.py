from typing import Union
import streamlit as st
from ui_components.methods.file_methods import get_file_size
from ui_components.models import InternalFileObject
from utils.local_storage.local_storage import read_from_motion_lora_local_db


def individual_video_display_element(file: Union[InternalFileObject, str]):
    file_location = file.location if file and not isinstance(file, str) and file.location else file
    if file_location:
        st.video(file_location, format='mp4', start_time=0) if get_file_size(file_location) < 5 else st.info("Video file too large to display")
    else: 
        st.error("No video present")
        

def display_motion_lora(motion_lora, lora_file_dict = {}):
    filename_video_dict = read_from_motion_lora_local_db()
    
    if motion_lora and motion_lora in filename_video_dict and filename_video_dict[motion_lora]:
        st.image(filename_video_dict[motion_lora])
    elif motion_lora in lora_file_dict:
        loras = [ele.split("/")[-1] for ele in lora_file_dict.keys()]
        try:
            idx = loras.index(motion_lora)
            if lora_file_dict[list(lora_file_dict.keys())[idx]]:
                st.image(lora_file_dict[list(lora_file_dict.keys())[idx]])                            
        except ValueError:
            st.write("")


