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

    animation_type = st.radio("Animation Interpolation:", \
                              options=[AnimationStyleType.CREATIVE_INTERPOLATION.value, AnimationStyleType.IMAGE_TO_VIDEO.value], \
                                key="animation_tool", horizontal=True, disabled=True)

    if animation_type == AnimationStyleType.CREATIVE_INTERPOLATION.value:
        st.markdown("***")
        
        shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
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
        d1, d2 = st.columns([1, 1])

        with d1:            
            
            frames_per_keyframe = st_memory.number_input("Frames per Keyframe", min_value=8, max_value=36, value=16, step=1, key="frames_per_keyframe")
            cn_strength = st_memory.slider("CN Strength", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="cn_strength")
            length_of_key_frame_influence = st_memory.slider("Length of Keyframe Influence", min_value=0.0, max_value=2.0, value=1.1, step=0.1, key="length_of_key_frame_influence")
            interpolation_style = st_memory.selectbox("Interpolation Style", options=["ease-in-out", "ease-in", "ease-out", "linear"], key="interpolation_style")                                    
            motion_scale = st_memory.slider("Motion Scale", min_value=0.0, max_value=2.0, value=1.1, step=0.1, key="motion_scale")


        with d2:
            import numpy as np
            import matplotlib.pyplot as plt

            def calculate_keyframe_peaks_and_influence(frames_per_keyframe, number_of_keyframes, length_of_influence):
                number_of_frames = frames_per_keyframe * number_of_keyframes
                interval = (number_of_frames - 1) // (number_of_keyframes - 1)
                adjustment = (number_of_frames - 1) % (number_of_keyframes - 1)

                peaks = [0]
                for i in range(1, number_of_keyframes - 1):
                    peak = peaks[-1] + interval
                    if i <= adjustment:
                        peak += 1
                    peaks.append(peak)
                peaks.append(number_of_frames - 1)

                full_interval = (number_of_frames - 1) / (number_of_keyframes - 1)
                scaled_interval = full_interval * length_of_influence

                influence_ranges = []
                for peak in peaks:
                    start_influence = max(0, int(peak - scaled_interval / 2.0))
                    end_influence = min(number_of_frames - 1, int(peak + scaled_interval / 2.0))
                    influence_ranges.append((start_influence, end_influence))

                return influence_ranges

            def calculate_weights_and_plot(influence_ranges, interpolation, strength=1.0):
                plt.figure(figsize=(12, 6))

                # Automatically generate frame names
                frame_names = [f'Frame {i+1}' for i in range(len(influence_ranges))]
                strength = float(strength)
                for i, (range_start, range_end) in enumerate(influence_ranges):
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
                    plt.plot(frame_numbers, weights, label=f'{frame_names[i]}: Range {range_start}-{range_end}')

                plt.xlabel('Frame Number')
                plt.ylabel('Weight')
                plt.title('Key Framing Influence Over Frames')
                plt.legend()
                plt.ylim(0, 1.0)
                plt.show()
            
            influence_ranges = calculate_keyframe_peaks_and_influence(frames_per_keyframe, len(timing_list), length_of_key_frame_influence)
            
            calculate_weights_and_plot(influence_ranges, interpolation_style, cn_strength)
            
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

            sd_model_list = [model_name.replace(".safetensors", "") for model_name in sd_model_list]

            
            sd_model = st_memory.selectbox("Which model would you like to use?", options=sd_model_list, key="sd_model")
            negative_prompt = st_memory.text_area("What would you like to avoid in the videos?", value="bad image, worst quality", key="negative_prompt")
            ip_adapter_strength = st_memory.slider("How tightly would you like the style to adhere to the input images?", min_value=0.0, max_value=1.0, value=0.66, step=0.1, key="ip_adapter_strength")
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
        # normalise_speed = st.checkbox("Normalise Speed", value=True, key="normalise_speed")

        settings.update(
            # positive_prompt=positive_prompt,
            negative_prompt="bad image, worst quality",     # default value, change this to something else
            image_dimension=img_dimension,
            sampling_steps=30,
            motion_module="",
            model=sd_model,
            normalise_speed=normalise_speed
        )
    
    elif animation_type == AnimationStyleType.IMAGE_TO_VIDEO.value:
        st.info("For image to video, you can select one or more prompts, and how many frames you want to generate for each prompt - it'll attempt to travel from one prompt to the next.")
        which_motion_module = st.selectbox("Which motion module would you like to use?", options=motion_modules, key="which_motion_module")

        # Initialize the list of dictionaries if it doesn't exist
        if 'travel_list' not in st.session_state:
            st.session_state['travel_list'] = []

        st.markdown("### Add to Prompt Travel List")
        prompt = st.text_area("Prompt")
        frame_count = st.number_input("How many frames would you like?", min_value=1, value=1, step=1)
        
        if st.button("Add to travel"):
            st.session_state['travel_list'].append({'prompt': prompt, 'frame_count': frame_count})

        st.markdown("***")
        st.markdown("### Travel List")

        # Display each item in the list
        if not st.session_state['travel_list']:
            st.error("The travel list is empty.")
        else:
            for i, item in enumerate(st.session_state['travel_list']):
                new_prompt = st.text_area(f"Prompt {i+1}", value=item['prompt'])
                bottom1, bottom2,bottom3 = st.columns([1, 2,1])
                with bottom1:
                    new_frame_count = st.number_input(f"Frame Count {i+1}", min_value=1, value=item['frame_count'], step=1)
                with bottom3:
                    if st.button(f"Delete Prompt {i+1}"):
                        del st.session_state['travel_list'][i]
                        st.rerun()
                # Update the item if it has been edited
                if new_prompt != item['prompt'] or new_frame_count != item['frame_count']:
                    st.session_state['travel_list'][i] = {'prompt': new_prompt, 'frame_count': new_frame_count}
                st.markdown("***")
        
        st.markdown("***")

        animate_col_1, animate_col_2 = st.columns([1, 3])

        with animate_col_1:
            variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")
    
    if st.button("Generate Animation Clip", key="generate_animation_clip", disabled=disable_generate, help=help):
        vid_quality = "full" if video_resolution == "Full Resolution" else "preview"
        st.write("Generating animation clip...")
        settings.update(animation_style=current_animation_style)
        create_single_interpolated_clip(
            shot_uuid,
            vid_quality,
            settings,
            variant_count
        )
        st.rerun()
    