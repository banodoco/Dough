import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip


def video_cropping_element(shot_uuid):
    st.title("Video Cropper")

    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    video_url = st.text_input("...or enter a video URL")

    if video_file or video_url:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        if video_file:
            tfile.write(video_file.read())
            video_path = tfile.name
        else:
            video_path = video_url

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()

        start_time = st.slider("Start Time", 0.0, float(duration), 0.0, 0.1)
        end_time = st.slider("End Time", 0.0, float(duration), float(duration), 0.1)

        starting1, starting2 = st.columns(2)
        with starting1:
            starting_frame_number = int(start_time * fps)
            display_frame(video_path, starting_frame_number)
        with starting2:
            ending_frame_number = int(end_time * fps)
            display_frame(video_path, ending_frame_number)

        if st.button("Save New Video"):
            with st.spinner("Processing..."):
                clip = VideoFileClip(video_path).subclip(start_time, end_time)
                output_file = video_path.split(".")[0] + "_cropped.mp4"
                clip.write_videofile(output_file)
                st.success("Saved as {}".format(output_file))


def display_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame)
    cap.release()
