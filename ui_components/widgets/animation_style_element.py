import time
import streamlit as st
from typing import List
from shared.constants import AnimationStyleType, AnimationToolType
from ui_components.constants import DefaultProjectSettingParams
from ui_components.methods.video_methods import create_single_interpolated_clip
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.motion_module import AnimateDiffCheckpoint
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from utils import st_memory

def animation_style_element(shot_uuid):
    motion_modules = AnimateDiffCheckpoint.get_name_list()
    variant_count = 1
    current_animation_style = AnimationStyleType.CREATIVE_INTERPOLATION.value    # setting a default value
    data_repo = DataRepo()

    # animation_type = st.radio("Animation Interpolation:", \
      #                        options=[AnimationStyleType.CREATIVE_INTERPOLATION.value, AnimationStyleType.IMAGE_TO_VIDEO.value], \
       #                         key="animation_tool", horizontal=True, disabled=True)

    
    st.markdown("***")
    
    shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
    st.session_state['project_uuid'] = str(shot.project.uuid)
    timing_list: List[InternalFrameTimingObject] = shot.timing_list
    st.markdown("#### Keyframe Settings")
    if timing_list and len(timing_list):
        columns = st.columns(len(timing_list)) 
        disable_generate = False
        help = ""
        for idx, timing in enumerate(timing_list):
            if timing.primary_image and timing.primary_image.location:
                columns[idx].image(timing.primary_image.location, use_column_width=True)
                b = timing.primary_image.inference_params
                prompt = columns[idx].text_area(f"Prompt {idx+1}", value=(b['prompt'] if b else ""), key=f"prompt_{idx+1}")
                # base_style_on_image = columns[idx].checkbox(f"Use base style image for prompt {idx+1}", key=f"base_style_image_{idx+1}",value=True)
            else:
                columns[idx].warning("No primary image present")
                disable_generate = True
                help = "You can't generate a video because one of your keyframes is missing an image."
    else:
        st.warning("No keyframes present")

    st.markdown("***")
    video_resolution = None

    settings = {
        'animation_tool': AnimationToolType.ANIMATEDIFF.value,
    }


    st.markdown("#### Keyframe Influence Settings")
    d1, d2 = st.columns([1, 4])

    with d1:
        setting_a_1, setting_a_2 = st.columns([1, 1])
        with setting_a_1:
            type_of_frame_distribution = st_memory.radio("Type of Frame Distribution", options=["Linear", "Dynamic"], key="type_of_frame_distribution").lower()                        
        if type_of_frame_distribution == "linear":
            with setting_a_2:
                linear_frame_distribution_value = st_memory.number_input("Frames per Keyframe", min_value=8, max_value=36, value=16, step=1, key="frames_per_keyframe")
                dynamic_frame_distribution_values = ""
        setting_b_1, setting_b_2 = st.columns([1, 1])
        with setting_b_1:
            type_of_key_frame_influence = st_memory.radio("Type of Keyframe Influence", options=["Linear", "Dynamic"], key="type_of_key_frame_influence").lower()
        if type_of_key_frame_influence == "linear":
            with setting_b_2:
                linear_key_frame_influence_value = st_memory.number_input("Length of Keyframe Influence", min_value=0.0, max_value=2.0, value=1.1, step=0.1, key="length_of_key_frame_influence")
                dynamic_key_frame_influence_values = ""
        setting_c_1, setting_c_2 = st.columns([1, 1])
        with setting_c_1:
            type_of_cn_strength_distribution = st_memory.radio("Type of CN Strength Distribution", options=["Linear", "Dynamic"], key="type_of_cn_strength_distribution").lower()
        if type_of_cn_strength_distribution == "linear":
            with setting_c_2:
                linear_cn_strength_value = st_memory.slider("CN Strength", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="linear_cn_strength_value")
                dynamic_cn_strength_values = ""

        
        # length_of_key_frame_influence = st_memory.slider("Length of Keyframe Influence", min_value=0.0, max_value=2.0, value=1.1, step=0.1, key="length_of_key_frame_influence")
        interpolation_style = st_memory.selectbox("Interpolation Style", options=["ease-in-out", "ease-in", "ease-out", "linear"], key="interpolation_style")                                    
        motion_scale = st_memory.slider("Motion Scale", min_value=0.0, max_value=2.0, value=1.1, step=0.1, key="motion_scale")

    with d2:
        import numpy as np
        import matplotlib.pyplot as plt
        columns = st.columns(len(timing_list)) 
        disable_generate = False
        help = ""            
        dynamic_frame_distribution_values = []
        dynamic_key_frame_influence_values = []
        dynamic_cn_strength_values = [] 

        if type_of_frame_distribution == "dynamic" or type_of_key_frame_influence == "dynamic" or type_of_cn_strength_distribution == "dynamic":
            for idx, timing in enumerate(timing_list):
                if timing.primary_image and timing.primary_image.location:
                    columns[idx].info(f"Frame {idx+1}")
                    columns[idx].image(timing.primary_image.location, use_column_width=True)
                    b = timing.primary_image.inference_params
                if type_of_frame_distribution == "dynamic":
                    linear_frame_distribution_value = ""                        
                    if f"frame_{idx+1}" not in st.session_state:
                        st.session_state[f"frame_{idx+1}"] = idx * 16  # Default values in increments of 16
                    if idx == 0:  # For the first frame, position is locked to 0
                        frame_position = columns[idx].number_input(f"Frame Position {idx+1}", min_value=0, max_value=0, value=0, step=1, key=f"dynamic_frame_distribution_values_{idx+1}", disabled=True)
                    else:                        
                        min_value = st.session_state[f"frame_{idx}"] + 1
                        frame_position = columns[idx].number_input(f"Frame Position {idx+1}", min_value=min_value, value=st.session_state[f"frame_{idx+1}"], step=1, key=f"dynamic_frame_distribution_values_{idx+1}")
                    st.session_state[f"frame_{idx+1}"] = frame_position
                    dynamic_frame_distribution_values.append(frame_position)
                if type_of_key_frame_influence == "dynamic":
                    linear_key_frame_influence_value = ""
                    dynamic_key_frame_influence_individual_value = columns[idx].number_input(f"Length of Keyframe Influence {idx+1}", min_value=0.0, max_value=5.0, value=(b['dynamic_key_frame_influence_values'] if 'dynamic_key_frame_influence_values' in b else 1.1), step=0.1, key=f"dynamic_key_frame_influence_values_{idx+1}")
                    dynamic_key_frame_influence_values.append(str(dynamic_key_frame_influence_individual_value))
                if type_of_cn_strength_distribution == "dynamic":
                    linear_cn_strength_value = ""
                    dynamic_cn_strength_individual_value = columns[idx].slider(f"CN Strength {idx+1}", min_value=0.0, max_value=1.0, value=(b['dynamic_cn_strength_values'] if 'dynamic_cn_strength_values' in b else 0.5), step=0.1, key=f"dynamic_cn_strength_values_{idx+1}")
                    dynamic_cn_strength_values.append(str(dynamic_cn_strength_individual_value))

        # Convert lists to strings
        dynamic_frame_distribution_values = ",".join(map(str, dynamic_frame_distribution_values))  # Convert integers to strings before joining
        dynamic_key_frame_influence_values = ",".join(dynamic_key_frame_influence_values)
        dynamic_cn_strength_values = ",".join(dynamic_cn_strength_values)
                    
                            
        def calculate_dynamic_influence_ranges(keyframe_positions, key_frame_influence_values):
            if len(keyframe_positions) < 2 or len(keyframe_positions) != len(key_frame_influence_values):
                return []

            influence_ranges = []
            for i, position in enumerate(keyframe_positions):
                influence_factor = key_frame_influence_values[i]

                # Calculate the base range size
                range_size = influence_factor * (keyframe_positions[-1] - keyframe_positions[0]) / (len(keyframe_positions) - 1) / 2

                # Calculate symmetric start and end influence
                start_influence = position - range_size
                end_influence = position + range_size

                # Adjust start and end influence to not exceed previous and next keyframes
                start_influence = max(start_influence, keyframe_positions[i - 1] if i > 0 else 0)
                end_influence = min(end_influence, keyframe_positions[i + 1] if i < len(keyframe_positions) - 1 else keyframe_positions[-1])

                influence_ranges.append((round(start_influence), round(end_influence)))

            return influence_ranges
        
        def get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value):
            if type_of_frame_distribution == "dynamic":
                # Sort the keyframe positions in numerical order
                return sorted([int(kf.strip()) for kf in dynamic_frame_distribution_values.split(',')])
            else:
                # Calculate the number of keyframes based on the total duration and linear_frames_per_keyframe
                return [i * linear_frame_distribution_value for i in range(len(images))]
        
        def extract_keyframe_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
            if type_of_key_frame_influence == "dynamic":
                # Parse the dynamic key frame influence values without sorting
                return [float(influence.strip()) for influence in dynamic_key_frame_influence_values.split(',')]
            else:
                # Create a list with the linear_key_frame_influence_value for each keyframe
                return [linear_key_frame_influence_value for _ in keyframe_positions]
        
        def calculate_weights_and_plot(influence_ranges, interpolation, strengths):
            plt.figure(figsize=(12, 6))

            # Automatically generate frame names
            frame_names = [f'Frame {i+1}' for i in range(len(influence_ranges))]
            for i, (range_start, range_end) in enumerate(influence_ranges):
                strength = float(strengths[i])  # Get the corresponding strength value
                if i == 0:
                    strength_from = 1.0
                    strength_to = 0.0
                    revert_direction_at_midpoint = False
                elif i == len(influence_ranges) - 1:
                    strength_from = 0.0
                    strength_to = 1.0
                    revert_direction_at_midpoint = False
                else:
                    strength_from = 0.0
                    strength_to = 1.0
                    revert_direction_at_midpoint = True

                steps = range_end - range_start
                diff = strength_to - strength_from

                if revert_direction_at_midpoint:
                    index = np.linspace(0, 1, steps // 2 + 1)
                else:
                    index = np.linspace(0, 1, steps)

                # Applying different interpolation styles
                if interpolation == "linear":
                    weights = np.linspace(strength_from, strength_to, len(index))
                elif interpolation == "ease-in":
                    weights = diff * np.power(index, 2) + strength_from
                elif interpolation == "ease-out":
                    weights = diff * (1 - np.power(1 - index, 2)) + strength_from
                elif interpolation == "ease-in-out":
                    weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from
                
                weights = weights.astype(float) * strength

                if revert_direction_at_midpoint:
                    if steps % 2 == 0:
                        weights = np.concatenate([weights, weights[-1::-1]])
                    else:
                        weights = np.concatenate([weights, weights[-2::-1]])

                frame_numbers = np.arange(range_start, range_start + len(weights))
                plt.plot(frame_numbers, weights, label=f'{frame_names[i]}')

            plt.xlabel('Frame Number')
            plt.ylabel('Weight')
            plt.title('Key Framing Influence Over Frames')
            plt.legend()
            plt.ylim(0, 1.0)
            plt.show()
        
        
        keyframe_positions = get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, timing_list, linear_frame_distribution_value)

        cn_strength_values = extract_keyframe_values(type_of_cn_strength_distribution, dynamic_cn_strength_values, keyframe_positions, linear_cn_strength_value)
        
        key_frame_influence_values = extract_keyframe_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value)    
        
        # st.markdown("Key Frame Positions")
        # st.markdown(keyframe_positions)
        # st.markdown("Key Frame Influence Values")
        # st.markdown(key_frame_influence_values)
        # st.markdown("CN Strength Values")
        # st.markdown(cn_strength_values)
                
        influence_ranges = calculate_dynamic_influence_ranges(keyframe_positions,key_frame_influence_values)
                                                                        
        calculate_weights_and_plot(influence_ranges, interpolation_style, cn_strength_values)
                    
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.pyplot()
    
    st.markdown("***")
    e1, e2 = st.columns([1, 1])
    
    with e1:
        st.markdown("#### Styling Settings")
        sd_model_list = [
            "Realistic_Vision_V5.0.safetensors",
            "Counterfeit-V3.0_fp32.safetensors",
            "epic_realism.safetensors",
            "dreamshaper_v8.safetensors",
            "deliberate_v3.safetensors"
        ]

        # remove .safe tensors from the end of each model name
        sd_model = st_memory.selectbox("Which model would you like to use?", options=sd_model_list, key="sd_model")
        negative_prompt = st_memory.text_area("What would you like to avoid in the videos?", value="bad image, worst quality", key="negative_prompt")
        ip_adapter_weight = st_memory.slider("How tightly would you like the style to adhere to the input images?", min_value=0.0, max_value=1.0, value=0.66, step=0.1, key="ip_adapter_weight")
        soft_scaled_cn_weights_multipler = st_memory.slider("How much would you like to scale the CN weights?", min_value=0.0, max_value=10.0, value=0.85, step=0.1, key="soft_scaled_cn_weights_multipler")
            
    st.markdown("***")
    st.markdown("#### Generation Settings")
    animate_col_1, _, _ = st.columns([1, 1, 2])

    with animate_col_1:
        # img_dimension_list = ["512x512", "512x768", "768x512"]
        # img_dimension = st.selectbox("Image Dimension:", options=img_dimension_list, key="img_dimension")  
        project_settings = data_repo.get_project_setting(shot.project.uuid)
        width = project_settings.width
        height = project_settings.height
        img_dimension = f"{width}x{height}"
        variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")
    
    normalise_speed = True
    context_length = 16
    context_stride = 2
    context_overlap = 2
    buffer = 4
    # normalise_speed = st.checkbox("Normalise Speed", value=True, key="normalise_speed")
    
    st.write(f"type_of_frame_distribution: {type_of_frame_distribution}")
    st.write(f"dynamic_frame_distribution_values: {dynamic_frame_distribution_values}")
    st.write(f"linear_frame_distribution_value: {linear_frame_distribution_value}")
    st.write(f"type_of_key_frame_influence: {type_of_key_frame_influence}")
    st.write(f"linear_key_frame_influence_value: {linear_key_frame_influence_value}")
    st.write(f"dynamic_key_frame_influence_values: {dynamic_key_frame_influence_values}")
    st.write(f"type_of_cn_strength_distribution: {type_of_cn_strength_distribution}")
    st.write(f"dynamic_cn_strength_values: {dynamic_cn_strength_values}")
    st.write(f"linear_cn_strength_value: {linear_cn_strength_value}")
    st.write(f"buffer: {buffer}")
    st.write(f"context_length: {context_length}")
    st.write(f"context_stride: {context_stride}")
    st.write(f"context_overlap: {context_overlap}")
    if type_of_frame_distribution == "linear":
        batch_size = ((len(timing_list)-1) * linear_frame_distribution_value) + buffer
        
    elif type_of_frame_distribution == "dynamic":
        batch_size = int(dynamic_frame_distribution_values.split(',')[-1]) + buffer
        
    st.write(f"batch_size: {batch_size}")


    settings.update(
        negative_prompt=negative_prompt,
        image_dimension=img_dimension,
        ip_adapter_weight=ip_adapter_weight,
        soft_scaled_cn_weights_multipler=soft_scaled_cn_weights_multipler,
        sampling_steps=30,
        motion_module="",
        model=sd_model,
        normalise_speed=normalise_speed,
        motion_scale=motion_scale,
        cn_strength=cn_strength,
        interpolation_style=interpolation_style,
        frames_per_keyframe=frames_per_keyframe,
        length_of_key_frame_influence=length_of_key_frame_influence
    )




    
    if st.button("Generate Animation Clip", key="generate_animation_clip", disabled=disable_generate, help=help):
        vid_quality = "full" if video_resolution == "Full Resolution" else "preview"
        st.write("Generating animation clip...")
        settings.update(animation_style=current_animation_style)
        
        if animation_type == AnimationStyleType.CREATIVE_INTERPOLATION.value:
            positive_prompt = ""
            for idx, timing in enumerate(timing_list):
                if timing.primary_image and timing.primary_image.location:
                    b = timing.primary_image.inference_params
                    prompt = b['prompt'] if b else ""
                    frame_prompt = f"{idx * frames_per_keyframe}_" + prompt
                    positive_prompt +=  ":" + frame_prompt if positive_prompt else frame_prompt
                else:
                    st.error("Please generate primary images")
                    time.sleep(0.5)
                    st.rerun()

            settings.update(positive_prompt=positive_prompt)

        create_single_interpolated_clip(
            shot_uuid,
            vid_quality,
            settings,
            variant_count
        )
        st.rerun()
    