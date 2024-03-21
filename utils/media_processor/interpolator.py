from shared.constants import QUEUE_INFERENCE_QUERIES, AnimationStyleType, AnimationToolType
from ui_components.constants import DefaultTimingStyleParams
from utils.common_utils import padded_integer
from utils.constants import MLQueryObject
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
    def create_interpolated_clip(animation_style, settings, variant_count=1, queue_inference=False, backlog=False):
        if not animation_style:
            animation_style = DefaultTimingStyleParams.animation_style

        if animation_style == AnimationStyleType.CREATIVE_INTERPOLATION.value:
            return VideoInterpolator.video_through_frame_interpolation(
                settings,
                variant_count,
                queue_inference,
                backlog
            )

        elif animation_style == AnimationStyleType.DIRECT_MORPHING.value:
            return VideoInterpolator.video_through_direct_morphing(
                settings,
                variant_count,
                queue_inference,                
                backlog
                )
        

    # returns a video bytes generated through interpolating frames between the given list of frames
    @staticmethod
    def video_through_frame_interpolation(settings, variant_count, queue_inference=False, backlog=False):
        ml_client = get_ml_client()

        final_res = []
        for _ in range(variant_count):
            sm_data = {
                "ckpt": settings['ckpt'],
                "width": settings['width'],  # "width": "512",
                "height": settings['height'],  # "height": "512",
                "buffer": settings['buffer'],
                "motion_scale": settings['motion_scale'],  # "motion_scale": "1.0",
                "motion_scales": settings['motion_scales'],
                "image_dimension": settings["image_dimension"],
                "output_format": settings['output_format'],                    
                "prompt": settings["prompt"],
                "negative_prompt": settings["negative_prompt"],
                # "image_prompt_list": settings["image_prompt_list"],
                "interpolation_type": settings["interpolation_type"],
                "stmfnet_multiplier": settings["stmfnet_multiplier"],                    
                "relative_ipadapter_strength": settings["relative_ipadapter_strength"],
                "relative_cn_strength": settings["relative_cn_strength"],
                "type_of_strength_distribution": settings["type_of_strength_distribution"],
                "linear_strength_value": settings["linear_strength_value"],
                "dynamic_strength_values": settings["dynamic_strength_values"],
                "linear_frame_distribution_value": settings["linear_frame_distribution_value"],
                "dynamic_frame_distribution_values": settings["dynamic_frame_distribution_values"],
                "type_of_frame_distribution": settings["type_of_frame_distribution"],                    
                "type_of_key_frame_influence": settings["type_of_key_frame_influence"],
                "linear_key_frame_influence_value": settings["linear_key_frame_influence_value"],
                "dynamic_key_frame_influence_values": settings["dynamic_key_frame_influence_values"],                    
                "normalise_speed": settings["normalise_speed"],
                "ipadapter_noise": settings["ipadapter_noise"],
                "queue_inference": True,
                "context_length": settings["context_length"],
                "context_stride": settings["context_stride"],
                "context_overlap": settings["context_overlap"],
                "multipled_base_end_percent": settings["multipled_base_end_percent"],
                "multipled_base_adapter_strength": settings["multipled_base_adapter_strength"],
                "individual_prompts": settings["individual_prompts"],
                "individual_negative_prompts": settings["individual_negative_prompts"],
                "max_frames": settings["max_frames"],
                "lora_data": settings["lora_data"],
                "shot_data": settings["shot_data"],
                "strength_of_structure_control_image": settings["strength_of_structure_control_image"]
            }

            # adding the input images
            for idx, img_uuid in enumerate(settings['file_uuid_list']):
                sm_data[f"file_image_{padded_integer(idx+1)}" + "_uuid"] = img_uuid

            # adding structure control img
            if "structure_control_image_uuid" in settings and settings["structure_control_image_uuid"] is not None:
                sm_data[f"file_structure_control_img_uuid"] = settings["structure_control_image_uuid"]

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
                mask_uuid=None,
                data=sm_data
            )
            res = ml_client.predict_model_output_standardized(ML_MODEL.ad_interpolation, ml_query_object, QUEUE_INFERENCE_QUERIES, backlog)
            final_res.append(res)

        return final_res
    

    @staticmethod
    def video_through_direct_morphing(settings, variant_count, queue_inference=False, backlog=False):
        ml_client = get_ml_client()

        final_res = []
        for _ in range(variant_count):
            sm_data = {
                "width": settings['width'],
                "height": settings['height'],
                "prompt": settings["prompt"]
            }

            for idx, img_uuid in enumerate(settings['file_uuid_list']):
                sm_data[f"file_image_{padded_integer(idx+1)}" + "_uuid"] = img_uuid

            ml_query_object = MLQueryObject(
                prompt="Dynamicrafter",  # hackish fix
                timing_uuid=None,
                model_uuid=None,
                guidance_scale=None,
                seed=None,
                num_inference_steps=None,            
                strength=None,
                adapter_type=None,
                negative_prompt="",
                height=settings['height'],
                width=settings['width'],                       
                image_uuid=None,
                mask_uuid=None,
                data=sm_data
            )
            res = ml_client.predict_model_output_standardized(ML_MODEL.dynamicrafter, ml_query_object, QUEUE_INFERENCE_QUERIES, backlog)
            final_res.append(res)

        return final_res
    


