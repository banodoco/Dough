import os
import cv2
import streamlit as st
import requests as r
import numpy as np
from shared.constants import AnimationStyleType, AnimationToolType
from ui_components.constants import DefaultTimingStyleParams
from ui_components.methods.file_methods import generate_temp_file
from ui_components.models import InferenceLogObject

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
    def create_interpolated_clip(img_location_list, animation_style, settings, variant_count=1, queue_inference=False):
        if not animation_style:
            animation_style = DefaultTimingStyleParams.animation_style

        if animation_style == AnimationStyleType.INTERPOLATION.value:
            return VideoInterpolator.video_through_frame_interpolation(
                img_location_list,
                settings,
                variant_count,
                queue_inference
            )

        elif animation_style == AnimationStyleType.DIRECT_MORPHING.value:
            return VideoInterpolator.video_through_direct_morphing(
                img_location_list,
                settings
                )
        

    # returns a video bytes generated through interpolating frames between the given list of frames
    @staticmethod
    def video_through_frame_interpolation(img_location_list, settings, variant_count, queue_inference=False):
        # TODO: extend this for more than two images
        img1 = img_location_list[0]
        img2 = img_location_list[1]
        img3 = img_location_list[2]

        if not img1.startswith("http"):
            img1 = open(img1, "rb")

        if not img2.startswith("http"):
            img2 = open(img2, "rb")

        if not img3.startswith("http"):
            img3 = open(img3, "rb")

        ml_client = get_ml_client()
        animation_tool = settings['animation_tool'] if 'animation_tool' in settings else AnimationToolType.G_FILM.value

        final_res = []
        for _ in range(variant_count):
            if animation_tool == AnimationToolType.G_FILM.value:
                res = ml_client.predict_model_output(
                                    REPLICATE_MODEL.google_frame_interpolation, 
                                    frame1=img1, 
                                    frame2=img2,
                                    times_to_interpolate=settings['interpolation_steps'], 
                                    queue_inference=queue_inference
                                )
            
            # since workflows can have multiple input params it's not standardized yet
            elif animation_tool == AnimationToolType.ANIMATEDIFF.value:
                # data = {
                #     "positive_prompt": settings['positive_prompt'],
                #     "negative_prompt": settings['negative_prompt'],
                #     "image_dimension": settings['image_dimension'],
                #     "starting_image_path": img1,
                #     "ending_image_path": img2,
                #     "sampling_steps": settings['sampling_steps'],
                #     "motion_module": settings['motion_module'],
                #     "model": settings['model'],
                #     "queue_inference": queue_inference
                # }

                data = {
                    "prompt_travel" : "0_:16_:24_",     # default value.. format {idx_prompt}:...
                    "negative_prompt" : settings['negative_prompt'],
                    "img_1" : img1,
                    "img_2" : img2,
                    "img_3" : img3,
                    "motion_module" : settings['motion_module'],
                    "model" : settings['model'],
                    "img_1_latent_cn_weights" : "0=1.00,1=0.82,2=0.74,3=0.56,4=0.47,5=0.41,6=0.38,7=0.33,8=0.30,9=0.28,10=0.25,11=0.24,12=0.20,13=0.17,14=0.15,15=0.13,16=0.13,17=0.11,18=0.11,19=0.11,20=0.11,21=0.11,22=0.10,23=0.09,24=0.06,25=0.04,26=0.03,27=0.01,28=0.00,29=0.00,30=0.00,31=0.00,32=0.00,33=0.00,34=0.00,35=0.00,36=0.00,37=0.00,38=0.00,39=0.00,40=0.00,41=0.00,42=0.00,43=0.00,44=0.00,45=0.00,46=0.00,47=0.00",
                    "img_2_latent_cn_weights" : "0=0.09,1=0.10,2=0.11,3=0.11,4=0.11,5=0.11,6=0.11,7=0.13,8=0.13,9=0.15,10=0.17,11=0.20,12=0.24,13=0.25,14=0.28,15=0.30,16=0.33,17=0.38,18=0.41,19=0.47,20=0.56,21=0.74,22=0.82,23=1.00,24=1.00,25=0.82,26=0.74,27=0.56,28=0.47,29=0.41,30=0.38,31=0.33,32=0.30,33=0.28,34=0.25,35=0.24,36=0.20,37=0.17,38=0.15,39=0.13,40=0.13,41=0.11,42=0.11,43=0.11,44=0.11,45=0.11,46=0.10,47=0.09\n\n\n\n",
                    "img_3_latent_cn_weights" : "0=0.00,1=0.00,2=0.00,3=0.00,4=0.00,5=0.00,6=0.00,7=0.00,8=0.00,9=0.00,10=0.00,11=0.00,12=0.00,13=0.00,14=0.00,15=0.00,16=0.00,17=0.00,18=0.00,19=0.00,20=0.01,21=0.03,22=0.04,23=0.06,24=0.09,25=0.10,26=0.11,27=0.11,28=0.11,29=0.11,30=0.11,31=0.13,32=0.13,33=0.15,34=0.17,35=0.20,36=0.24,37=0.25,38=0.28,39=0.30,40=0.33,41=0.38,42=0.41,43=0.47,44=0.56,45=0.74,46=0.82,47=1.00",
                    "ip_adapter_weight" : 0.4,
                    "ip_adapter_noise" : 0.5,
                    "output_format" : "video/h264-mp4",     # can also be "image/gif"
                    "queue_inference" : queue_inference,
                    "image_dimension": settings['image_dimension']
                }

                res = ml_client.predict_model_output(REPLICATE_MODEL.ad_interpolation, **data)

            final_res.append(res)
        
        # final_res = []
        # for (output, log) in res:
        #     temp_output_file = generate_temp_file(output, '.mp4')
        #     video_bytes = None
        #     with open(temp_output_file.name, 'rb') as f:
        #         video_bytes = f.read()

        #     os.remove(temp_output_file.name)
        #     final_res.append((video_bytes, log))

        return final_res
    

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
        return [(video_data, InferenceLogObject({}))]    # returning None for inference log