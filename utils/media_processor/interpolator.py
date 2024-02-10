import os
import time
import cv2
import streamlit as st
import requests as r
import numpy as np
from shared.constants import QUEUE_INFERENCE_QUERIES, AnimationStyleType, AnimationToolType
from ui_components.constants import DefaultTimingStyleParams
from ui_components.methods.file_methods import generate_temp_file, zip_images
from ui_components.models import InferenceLogObject
from utils.constants import MLQueryObject

from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL


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

        if animation_style == AnimationStyleType.CREATIVE_INTERPOLATION.value:
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
        ml_client = get_ml_client()
        # zip_filename = zip_images(img_location_list)
        # zip_url = ml_client.upload_training_data(zip_filename, delete_after_upload=True)
        # print("zipped file url: ", zip_url)
        # animation_tool = settings['animation_tool'] if 'animation_tool' in settings else AnimationToolType.G_FILM.value

        final_res = []
        for _ in range(variant_count):
            # if animation_tool == AnimationToolType.G_FILM.value:
            #     res = ml_client.predict_model_output(
            #                         ML_MODEL.google_frame_interpolation, 
            #                         frame1=img1, 
            #                         frame2=img2,
            #                         times_to_interpolate=settings['interpolation_steps'], 
            #                         queue_inference=queue_inference
            #                     )
            
            # since workflows can have multiple input params it's not standardized yet
            # elif animation_tool == AnimationToolType.ANIMATEDIFF.value:

            # defaulting to animatediff interpolation
            if True:
                # NOTE: @Peter these are all the settings you passed in from the UI
                sm_data = {
                    "ckpt": settings['ckpt'],
                    "buffer": settings['buffer'],
                    "motion_scale": settings['motion_scale'],
                    "output_format": settings['output_format'],
                    "image_dimension": settings["image_dimension"],
                    "negative_prompt": settings["negative_prompt"],
                    "image_prompt_list": settings["image_prompt_list"],
                    "interpolation_type": settings["interpolation_type"],
                    "stmfnet_multiplier": settings["stmfnet_multiplier"],                    
                    "relative_ipadapter_strength": settings["relative_ipadapter_strength"],
                    "relative_ipadapter_influence": settings["relative_ipadapter_influence"],
                    "type_of_cn_strength_distribution": settings["type_of_cn_strength_distribution"],
                    "linear_cn_strength_value": settings["linear_cn_strength_value"],
                    "dynamic_cn_strength_values": settings["dynamic_strength_values"],
                    "linear_frame_distribution_value": settings["linear_frame_distribution_value"],
                    "type_of_frame_distribution": settings["type_of_frame_distribution"],
                    "dynamic_frame_distribution_values": settings["dynamic_frame_distribution_values"],
                    "type_of_key_frame_influence": settings["type_of_key_frame_influence"],
                    "linear_key_frame_influence_value": settings["linear_key_frame_influence_value"],
                    "dynamic_key_frame_influence_values": settings["dynamic_key_frame_influence_values"],
                    "relative_ipadapter_strength": settings["relative_ipadapter_strength"],
                    "relative_ipadapter_influence": settings["relative_ipadapter_influence"],
                    "ipadapter_noise": 0.25,
                    "queue_inference": True
                }

                # adding the input images
                for idx, img_uuid in enumerate(settings['file_uuid_list']):
                    sm_data["file_image_" + str(idx) + "_uuid"] = img_uuid

                # NOTE: @Peter all the above settings are put in the 'data' parameter below
                ml_query_object = MLQueryObject(
                    prompt="SM",  # hackish fix
                    timing_uuid=None,
                    model_uuid=None,
                    guidance_scale=None,
                    seed=None,
                    num_inference_steps=None,            
                    strength=None,
                    adapter_type=None,
                    negative_prompt="",
                    height=512,
                    width=512,
                    low_threshold=100,
                    high_threshold=200,
                    image_uuid=None,
                    mask_uuid=None,
                    data=sm_data
                )
                res = ml_client.predict_model_output_standardized(ML_MODEL.ad_interpolation, ml_query_object, QUEUE_INFERENCE_QUERIES)

            final_res.append(res)

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