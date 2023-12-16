import json
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
import numpy as np
import matplotlib.pyplot as plt

def animation_style_element(shot_uuid):

    motion_modules = AnimateDiffCheckpoint.get_name_list()
    variant_count = 1
    current_animation_style = AnimationStyleType.CREATIVE_INTERPOLATION.value    # setting a default value
    data_repo = DataRepo()
    
    shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
    st.session_state['project_uuid'] = str(shot.project.uuid)
    timing_list: List[InternalFrameTimingObject] = shot.timing_list


    video_resolution = None

    settings = {
        'animation_tool': AnimationToolType.ANIMATEDIFF.value,
    }


    st.markdown("#### Key Frame Settings")
    d1, d2 = st.columns([1, 4])
    st.session_state['frame_position'] = 0
    with d1:
        setting_a_1, setting_a_2, = st.columns([1, 1])
        with setting_a_1:
            type_of_frame_distribution = st_memory.radio("Type of key frame distribution:", options=["Linear", "Dynamic"], key="type_of_frame_distribution").lower()                                    
        if type_of_frame_distribution == "linear":
            with setting_a_2:
                linear_frame_distribution_value = st_memory.number_input("Frames per key frame:", min_value=8, max_value=36, value=16, step=1, key="frames_per_keyframe")
                dynamic_frame_distribution_values = []
        st.markdown("***")
        setting_b_1, setting_b_2 = st.columns([1, 1])
        with setting_b_1:
            type_of_key_frame_influence = st_memory.radio("Type of key frame length influence:", options=["Linear", "Dynamic"], key="type_of_key_frame_influence").lower()
        if type_of_key_frame_influence == "linear":
            with setting_b_2:
                linear_key_frame_influence_value = st_memory.slider("Length of key frame influence:", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="length_of_key_frame_influence")
                dynamic_key_frame_influence_values = []
        st.markdown("***")

        setting_d_1, setting_d_2 = st.columns([1, 1])

        with setting_d_1:
            type_of_cn_strength_distribution = st_memory.radio("Type of key frame strength control:", options=["Linear", "Dynamic"], key="type_of_cn_strength_distribution").lower()
        if type_of_cn_strength_distribution == "linear":
            with setting_d_2:
                linear_cn_strength_value = st_memory.slider("Range of strength:", min_value=0.0, max_value=1.0, value=(0.0,0.7), step=0.1, key="linear_cn_strength_value")                
                dynamic_cn_strength_values = []
        
        st.markdown("***")
        footer1, _ = st.columns([2, 1])
        with footer1:
            interpolation_style = 'ease-in-out'
            motion_scale = st_memory.slider("Motion scale:", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="motion_scale")
        
        st.markdown("***")
        if st.button("Reset to default settings", key="reset_animation_style"):
            update_interpolation_settings(timing_list=timing_list)
            st.rerun()

    with d2:
        columns = st.columns(max(7, len(timing_list))) 
        disable_generate = False
        help = ""            
        dynamic_frame_distribution_values = []
        dynamic_key_frame_influence_values = []
        dynamic_cn_strength_values = []         


        for idx, timing in enumerate(timing_list):
            # Use modulus to cycle through colors
            # color = color_names[idx % len(color_names)]
            # Only create markdown text for the current index
            markdown_text = f'##### **Frame {idx + 1}** ___'

            with columns[idx]:
                st.markdown(markdown_text)

            if timing.primary_image and timing.primary_image.location:                
                columns[idx].image(timing.primary_image.location, use_column_width=True)
                b = timing.primary_image.inference_params
            if type_of_frame_distribution == "dynamic":
                linear_frame_distribution_value = 16
                if f"frame_{idx+1}" not in st.session_state:
                    st.session_state[f"frame_{idx+1}"] = idx * 16  # Default values in increments of 16
                if idx == 0:  # For the first frame, position is locked to 0
                    with columns[idx]:
                        frame_position = st_memory.number_input(f"{idx+1} frame Position", min_value=0, max_value=0, value=0, step=1, key=f"dynamic_frame_distribution_values_{idx+1}", disabled=True)
                else:                        
                    min_value = st.session_state[f"frame_{idx}"] + 1
                    with columns[idx]:
                        frame_position = st_memory.number_input(f"#{idx+1} position:", min_value=min_value, value=st.session_state[f"frame_{idx+1}"], step=1, key=f"dynamic_frame_distribution_values_{idx+1}")
                # st.session_state[f"frame_{idx+1}"] = frame_position
                dynamic_frame_distribution_values.append(frame_position)

            if type_of_key_frame_influence == "dynamic":
                linear_key_frame_influence_value = 1.1
                with columns[idx]:                    
                    dynamic_key_frame_influence_individual_value = st_memory.slider(f"#{idx+1} length of influence:", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key=f"dynamic_key_frame_influence_values_{idx}")
                dynamic_key_frame_influence_values.append(str(dynamic_key_frame_influence_individual_value))

            if type_of_cn_strength_distribution == "dynamic":
                linear_cn_strength_value = (0.0,1.0)
                with columns[idx]:
                    help_texts = ["For the first frame, it'll start at the endpoint and decline to the starting point",
                                  "For the final frame, it'll start at the starting point and end at the endpoint",
                                  "For intermediate frames, it'll start at the starting point, peak in the middle at the endpoint, and decline to the starting point"]
                    label_texts = [f"#{idx+1} end -> start:", f"#{idx+1} start -> end:", f"#{idx+1} start -> peak:"]
                    help_text = help_texts[0] if idx == 0 else help_texts[1] if idx == len(timing_list) - 1 else help_texts[2]
                    label_text = label_texts[0] if idx == 0 else label_texts[1] if idx == len(timing_list) - 1 else label_texts[2]
                    dynamic_cn_strength_individual_value = st_memory.slider(label_text, min_value=0.0, max_value=1.0, value=(0.0,0.7), step=0.1, key=f"dynamic_cn_strength_values_{idx}",help=help_text)
                dynamic_cn_strength_values.append(str(dynamic_cn_strength_individual_value))

        # Convert lists to strings
        dynamic_frame_distribution_values = ",".join(map(str, dynamic_frame_distribution_values))  # Convert integers to strings before joining
        dynamic_key_frame_influence_values = ",".join(dynamic_key_frame_influence_values)
        dynamic_cn_strength_values = ",".join(dynamic_cn_strength_values)
        # dynamic_start_and_endpoint_values = ",".join(dynamic_start_and_endpoint_values)
        # st.write(dynamic_start_and_endpoint_values)
                    
                            
        def calculate_dynamic_influence_ranges(keyframe_positions, key_frame_influence_values, allow_extension=True):
            if len(keyframe_positions) < 2 or len(keyframe_positions) != len(key_frame_influence_values):
                return []

            influence_ranges = []
            for i, position in enumerate(keyframe_positions):
                influence_factor = key_frame_influence_values[i]
                range_size = influence_factor * (keyframe_positions[-1] - keyframe_positions[0]) / (len(keyframe_positions) - 1) / 2

                start_influence = position - range_size
                end_influence = position + range_size

                # If extension beyond the adjacent keyframe is allowed, do not constrain the start and end influence.
                if not allow_extension:
                    start_influence = max(start_influence, keyframe_positions[i - 1] if i > 0 else 0)
                    end_influence = min(end_influence, keyframe_positions[i + 1] if i < len(keyframe_positions) - 1 else keyframe_positions[-1])

                influence_ranges.append((round(start_influence), round(end_influence)))

            return influence_ranges
        
        
        def get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value):
            if type_of_frame_distribution == "dynamic":
                return sorted([int(kf.strip()) for kf in dynamic_frame_distribution_values.split(',')])
            else:
                return [i * linear_frame_distribution_value for i in range(len(images))]
        
        def extract_keyframe_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
            if type_of_key_frame_influence == "dynamic":
                return [float(influence.strip()) for influence in dynamic_key_frame_influence_values.split(',')]
            else:
                return [linear_key_frame_influence_value for _ in keyframe_positions]
        
        def extract_start_and_endpoint_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
            if type_of_key_frame_influence == "dynamic":
                # If dynamic_key_frame_influence_values is a list of characters representing tuples, process it
                if isinstance(dynamic_key_frame_influence_values[0], str) and dynamic_key_frame_influence_values[0] == "(":
                    # Join the characters to form a single string and evaluate to convert into a list of tuples
                    string_representation = ''.join(dynamic_key_frame_influence_values)
                    dynamic_values = eval(f'[{string_representation}]')
                else:
                    # If it's already a list of tuples or a single tuple, use it directly
                    dynamic_values = dynamic_key_frame_influence_values if isinstance(dynamic_key_frame_influence_values, list) else [dynamic_key_frame_influence_values]
                return dynamic_values
            else:
                # Return a list of tuples with the linear_key_frame_influence_value as a tuple repeated for each position
                return [linear_key_frame_influence_value for _ in keyframe_positions]

        def calculate_weights(influence_ranges, interpolation, start_and_endpoint_strength, last_key_frame_position):
            weights_list = []
            frame_numbers_list = []

            for i, (range_start, range_end) in enumerate(influence_ranges):
                # Initialize variables
                if i == 0:
                    strength_to, strength_from = start_and_endpoint_strength[i] if i < len(start_and_endpoint_strength) else (0.0, 1.0)
                else:
                    strength_from, strength_to = start_and_endpoint_strength[i] if i < len(start_and_endpoint_strength) else (1.0, 0.0)
                revert_direction_at_midpoint = (i != 0) and (i != len(influence_ranges) - 1)

                # if it's the first value, set influence range from 1.0 to 0.0
                if i == 0:
                    range_start = 0

                # if it's the last value, set influence range to end at last_key_frame_position
                if i == len(influence_ranges) - 1:
                    range_end = last_key_frame_position

                steps = range_end - range_start
                diff = strength_to - strength_from

                # Calculate index for interpolation
                index = np.linspace(0, 1, steps // 2 + 1) if revert_direction_at_midpoint else np.linspace(0, 1, steps)

                # Calculate weights based on interpolation type
                if interpolation == "linear":
                    weights = np.linspace(strength_from, strength_to, len(index))
                elif interpolation == "ease-in":
                    weights = diff * np.power(index, 2) + strength_from
                elif interpolation == "ease-out":
                    weights = diff * (1 - np.power(1 - index, 2)) + strength_from
                elif interpolation == "ease-in-out":
                    weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from

                # If it's a middle keyframe, mirror the weights
                if revert_direction_at_midpoint:
                    weights = np.concatenate([weights, weights[::-1]])

                # Generate frame numbers
                frame_numbers = np.arange(range_start, range_start + len(weights))

                # "Dropper" component: For keyframes with negative start, drop the weights
                if range_start < 0 and i > 0:
                    drop_count = abs(range_start)
                    weights = weights[drop_count:]
                    frame_numbers = frame_numbers[drop_count:]

                # Dropper component: for keyframes a range_End is greater than last_key_frame_position, drop the weights
                if range_end > last_key_frame_position and i < len(influence_ranges) - 1:
                    drop_count = range_end - last_key_frame_position
                    weights = weights[:-drop_count]
                    frame_numbers = frame_numbers[:-drop_count]

                weights_list.append(weights)
                frame_numbers_list.append(frame_numbers)

            return weights_list, frame_numbers_list


        def plot_weights(weights_list, frame_numbers_list, frame_names):
            plt.figure(figsize=(12, 6))

            for i, weights in enumerate(weights_list):
                frame_numbers = frame_numbers_list[i]
                plt.plot(frame_numbers, weights, label=f'{frame_names[i]}')

            # Plot settings
            plt.xlabel('Frame Number')
            plt.ylabel('Weight')
            plt.legend()
            plt.ylim(0, 1.0)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
                                    
        keyframe_positions = get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, timing_list, linear_frame_distribution_value)
        last_key_frame_position = keyframe_positions[-1]
        cn_strength_values = extract_start_and_endpoint_values(type_of_cn_strength_distribution, dynamic_cn_strength_values, keyframe_positions, linear_cn_strength_value)
        key_frame_influence_values = extract_keyframe_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value)    
        # start_and_endpoint_values = extract_start_and_endpoint_values(type_of_start_and_endpoint, dynamic_start_and_endpoint_values, keyframe_positions, linear_start_and_endpoint_value)
        influence_ranges = calculate_dynamic_influence_ranges(keyframe_positions, key_frame_influence_values)        
        weights_list, frame_numbers_list = calculate_weights(influence_ranges, interpolation_style, cn_strength_values, last_key_frame_position)
        frame_names = [f'Frame {i+1}' for i in range(len(influence_ranges))]
        plot_weights(weights_list, frame_numbers_list, frame_names)

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
        sd_model = st_memory.selectbox("Which model would you like to use?", options=sd_model_list, key="sd_model_video")
        negative_prompt = st_memory.text_area("What would you like to avoid in the videos?", value="bad image, worst quality", key="negative_prompt_video")
        relative_ipadapter_strength = st_memory.slider("How much would you like to influence the style?", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="ip_adapter_strength")
        relative_ipadapter_influence = st_memory.slider("For how long would you like to influence the style?", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="ip_adapter_influence")
        soft_scaled_cn_weights_multipler = st_memory.slider("How much would you like to scale the CN weights?", min_value=0.0, max_value=10.0, value=0.85, step=0.1, key="soft_scaled_cn_weights_multiple_video")
            
    normalise_speed = True
    
    project_settings = data_repo.get_project_setting(shot.project.uuid)
    width = project_settings.width
    height = project_settings.height
    img_dimension = f"{width}x{height}"

    settings.update(
        ckpt=sd_model,
        buffer=4,
        motion_scale=motion_scale,
        image_dimension=img_dimension,
        output_format="video/h264-mp4",
        negative_prompt=negative_prompt,
        interpolation_type=interpolation_style,
        stmfnet_multiplier=2,
        relative_ipadapter_strength=relative_ipadapter_strength,
        relative_ipadapter_influence=relative_ipadapter_influence,
        soft_scaled_cn_weights_multiplier=soft_scaled_cn_weights_multipler,
        type_of_cn_strength_distribution=type_of_cn_strength_distribution,
        linear_cn_strength_value=str(linear_cn_strength_value),
        dynamic_cn_strength_values=str(dynamic_cn_strength_values),
        type_of_frame_distribution=type_of_frame_distribution,
        linear_frame_distribution_value=linear_frame_distribution_value,
        dynamic_frame_distribution_values=dynamic_frame_distribution_values,
        type_of_key_frame_influence=type_of_key_frame_influence,
        linear_key_frame_influence_value=float(linear_key_frame_influence_value),
        dynamic_key_frame_influence_values=dynamic_key_frame_influence_values,
        normalise_speed=normalise_speed,
        animation_style=AnimationStyleType.CREATIVE_INTERPOLATION.value
    )
    
    st.markdown("***")
    st.markdown("#### Generation Settings")
    where_to_generate = st_memory.radio("Where would you like to generate the video?", options=["Cloud", "Local"], key="where_to_generate", horizontal=True)
    if where_to_generate == "Cloud":
        animate_col_1, animate_col_2, _ = st.columns([1, 1, 2])
        with animate_col_1:
            variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")
            
            if st.button("Generate Animation Clip", key="generate_animation_clip", disabled=disable_generate, help=help):
                vid_quality = "full" if video_resolution == "Full Resolution" else "preview"
                st.success("Generating clip - see status in the Generation Log in the sidebar. Press 'Refresh log' to update.")

                positive_prompt = ""
                for idx, timing in enumerate(timing_list):
                    if timing.primary_image and timing.primary_image.location:
                        b = timing.primary_image.inference_params
                        prompt = b['prompt'] if b else ""
                        frame_prompt = f"{idx * linear_frame_distribution_value}_" + prompt
                        positive_prompt +=  ":" + frame_prompt if positive_prompt else frame_prompt
                    else:
                        st.error("Please generate primary images")
                        time.sleep(0.7)
                        st.rerun()

                settings.update(
                    image_prompt_list=positive_prompt,
                    animation_stype=current_animation_style,
                )

                create_single_interpolated_clip(
                    shot_uuid,
                    vid_quality,
                    settings,
                    variant_count
                )
                st.rerun()

        with animate_col_2:
            number_of_frames = len(timing_list)
            
            if height==width:
                cost_per_key_frame = 0.035
            else:
                cost_per_key_frame = 0.045

            cost_per_generation = cost_per_key_frame * number_of_frames * variant_count
            st.info(f"Generating a video with {number_of_frames} frames in the cloud will cost c. ${cost_per_generation:.2f} USD.")
    
    elif where_to_generate == "Local":
        h1,h2 = st.columns([1,1])
        with h1:
            st.info("You can run this locally in ComfyUI but you'll need at least 16GB VRAM. To get started, you can follow the instructions [here](https://github.com/peteromallet/steerable-motion) and download the workflow and images below.")

            btn1, btn2 = st.columns([1,1])
            with btn1:
                st.download_button(
                    label="Download workflow JSON",
                    data=json.dumps(prepare_workflow_json(shot_uuid, settings)),
                    file_name='workflow.json'
                )
            with btn2:
                st.download_button(
                    label="Download images",
                    data=prepare_workflow_images(shot_uuid),
                    file_name='data.zip'
                )

def prepare_workflow_json(shot_uuid, settings):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    positive_prompt = ""
    for idx, timing in enumerate(shot.timing_list):
        b = None
        if timing.primary_image and timing.primary_image.location:
            b = timing.primary_image.inference_params
        prompt = b['prompt'] if b else ""
        frame_prompt = f'"{idx * settings["linear_frame_distribution_value"]}":"{prompt}"' + ("," if idx != len(shot.timing_list) - 1 else "")
        positive_prompt +=  frame_prompt

    settings['image_prompt_list'] = positive_prompt
    workflow_data = create_workflow_json(shot.timing_list, settings)

    return workflow_data

def prepare_workflow_images(shot_uuid):
    import requests
    import io
    import zipfile

    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    image_locations = [t.primary_image.location if t.primary_image else None for t in shot.timing_list]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for idx, image_location in enumerate(image_locations):
            if not image_location:
                continue

            image_name = f"{idx}.png"
            if image_location.startswith('http'):
                response = requests.get(image_location)
                image_data = response.content
                zip_file.writestr(image_name, image_data)
            else:
                zip_file.write(image_location, image_name)

    buffer.seek(0)
    return buffer.getvalue()


def create_workflow_json(image_locations, settings):
    import os

    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    print(current_working_directory)

    with open('./sample_assets/interpolation_workflow.json', 'r') as json_file:
        json_data = json.load(json_file)

    image_prompt_list = settings['image_prompt_list']
    negative_prompt = settings['negative_prompt']
    type_of_frame_distribution = settings['type_of_frame_distribution']
    linear_frame_distribution_value = settings['linear_frame_distribution_value']
    dynamic_frame_distribution_values = settings['dynamic_frame_distribution_values']
    type_of_key_frame_influence = settings['type_of_key_frame_influence']
    linear_key_frame_influence_value = settings['linear_key_frame_influence_value']
    dynamic_key_frame_influence_values = settings['dynamic_key_frame_influence_values']
    type_of_cn_strength_distribution=settings['type_of_cn_strength_distribution']
    linear_cn_strength_value=settings['linear_cn_strength_value']
    buffer = settings['buffer']
    dynamic_cn_strength_values = settings['dynamic_cn_strength_values']
    interpolation_type = settings['interpolation_type']
    ckpt = settings['ckpt']
    motion_scale = settings['motion_scale']    
    relative_ipadapter_strength = settings['relative_ipadapter_strength']
    relative_ipadapter_influence = settings['relative_ipadapter_influence']
    image_dimension = settings['image_dimension']
    output_format = settings['output_format']
    soft_scaled_cn_weights_multiplier = settings['soft_scaled_cn_weights_multiplier']
    stmfnet_multiplier = settings['stmfnet_multiplier']

    if settings['type_of_frame_distribution'] == 'linear':
        batch_size = (len(image_locations) - 1) * settings['linear_frame_distribution_value'] + int(buffer)
    else:
        batch_size = int(settings['dynamic_frame_distribution_values'].split(',')[-1]) + int(buffer)

    img_width, img_height = image_dimension.split("x")
        
    
        
    for node in json_data['nodes']:
        if node['id'] == 189:
            node['widgets_values'][-3] = int(img_width)
            node['widgets_values'][-2] = int(img_height)
            node['widgets_values'][0] = ckpt
            node['widgets_values'][-1] = batch_size

        elif node['id'] == 187:
            node['widgets_values'][-2] = motion_scale

        elif node['id'] == 347:
            node['widgets_values'][0] = image_prompt_list

        elif node['id'] == 352:
            node['widgets_values'] = [negative_prompt]

        elif node['id'] == 365:
            node['widgets_values'][1] = type_of_frame_distribution
            node['widgets_values'][2] = linear_frame_distribution_value
            node['widgets_values'][3] = dynamic_frame_distribution_values
            node['widgets_values'][4] = type_of_key_frame_influence
            node['widgets_values'][5] = linear_key_frame_influence_value
            node['widgets_values'][6] = dynamic_key_frame_influence_values
            node['widgets_values'][7] = type_of_cn_strength_distribution
            node['widgets_values'][8] = linear_cn_strength_value
            node['widgets_values'][9] = dynamic_cn_strength_values
            node['widgets_values'][-1] = buffer
            node['widgets_values'][-2] = interpolation_type
            node['widgets_values'][-3] = soft_scaled_cn_weights_multiplier

        elif node['id'] == 292:
            node['widgets_values'][-2] = stmfnet_multiplier

        elif node['id'] == 301:
            node['widgets_values'] = [ip_adapter_model_weight]

        elif node['id'] == 281:
            # Special case: 'widgets_values' is a dictionary for this node
            node['widgets_values']['output_format'] = output_format
        
        # st.write(json_data)
    
    return json_data
            

def update_interpolation_settings(values=None, timing_list=None):
    default_values = {
        'type_of_frame_distribution': 0,
        'frames_per_keyframe': 16,
        'type_of_key_frame_influence': 0,
        'length_of_key_frame_influence': 1.0,
        'type_of_cn_strength_distribution': 0,
        'linear_cn_strength_value': (0.0,0.7),
        'interpolation_style': 0,
        'motion_scale': 1.0,            
        'negative_prompt_video': 'bad image, worst quality',        
        'ip_adapter_strength': 1.0,
        'ip_adapter_influence': 1.0,
        'soft_scaled_cn_weights_multiple_video': 0.85
    }

    for idx in range(0, len(timing_list)):
        default_values[f'dynamic_frame_distribution_values_{idx}'] = (idx - 1) * 16
        default_values[f'dynamic_key_frame_influence_values_{idx}'] = 1.0
        default_values[f'dynamic_cn_strength_values_{idx}'] = (0.0,0.7)

    for key, default_value in default_values.items():
        st.session_state[key] = values.get(key, default_value) if values and values.get(key) is not None else default_value