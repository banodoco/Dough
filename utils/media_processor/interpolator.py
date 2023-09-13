import os
import cv2
import streamlit as st
import requests as r
import numpy as np
from shared.constants import AnimationStyleType
from ui_components.methods.file_methods import generate_temp_file

from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import REPLICATE_MODEL


class VideoInterpolator:
    @staticmethod
    def calculate_dynamic_interpolations_steps(clip_duration):
        if clip_duration < 0.17:
            interpolation_steps = 2
        elif clip_duration < 0.3:
            interpolation_steps = 3
        elif clip_duration < 0.57:
            interpolation_steps = 4
        elif clip_duration < 1.1:
            interpolation_steps = 5
        elif clip_duration < 2.17:
            interpolation_steps = 6
        elif clip_duration < 4.3:
            interpolation_steps = 7
        else:
            interpolation_steps = 8
            
        return interpolation_steps
    
    @staticmethod
    def create_interpolated_clip(img_location_list, animation_style, settings):
        data_repo = DataRepo()
        if not animation_style:
            project_setting = data_repo.get_project_setting(st.session_state["project_uuid"])
            animation_style = project_setting.default_animation_style

        if animation_style == AnimationStyleType.INTERPOLATION.value:
            output_video_bytes = VideoInterpolator.video_through_frame_interpolation(
                img_location_list,
                settings
            )

        elif animation_style == AnimationStyleType.DIRECT_MORPHING.value:
            output_video_bytes = VideoInterpolator.video_through_direct_morphing(
                img_location_list,
                settings
                )

        return output_video_bytes

    # returns a video bytes generated through interpolating frames between the given list of frames
    @staticmethod
    def video_through_frame_interpolation(img_location_list, settings):
        # TODO: extend this for more than two images
        img1 = img_location_list[0]
        img2 = img_location_list[1]

        if not img1.startswith("http"):
            img1 = open(img1, "rb")

        if not img2.startswith("http"):
            img2 = open(img2, "rb")

        ml_client = get_ml_client()
        output = ml_client.predict_model_output(REPLICATE_MODEL.google_frame_interpolation, frame1=img1, frame2=img2,
                                                    times_to_interpolate=settings['interpolation_steps'])
        
        temp_output_file = generate_temp_file(output, '.mp4')
        video_bytes = None
        with open(temp_output_file.name, 'rb') as f:
            video_bytes = f.read()

        os.remove(temp_output_file.name)

        return video_bytes

    @staticmethod
    def video_through_direct_morphing(img_location_list, settings):
        def load_image(image_path_or_url):
            if image_path_or_url.startswith("http"):
                response = r.get(image_path_or_url)
                image = np.asarray(bytearray(response.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(image_path_or_url)

            return image
        
        img1 = load_image(img_location_list[0])
        img2 = load_image(img_location_list[1])

        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both of the images.")
        
        num_frames = settings['interpolation_steps']  # Number of frames in the video
        video_frames = []

        for alpha in np.linspace(0, 1, num_frames):
            morphed_image = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
            video_frames.append(morphed_image)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_bytes = []
        for frame in video_frames:
            ret, frame_bytes = cv2.imencode('.mp4', frame, fourcc)
            if not ret:
                raise ValueError("Failed to encode video frame")
            video_bytes.append(frame_bytes.tobytes())

        video_data = b''.join(video_bytes)
        return video_data
        
