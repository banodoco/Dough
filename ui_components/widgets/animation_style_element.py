import json
import time
import streamlit as st
from typing import List
from shared.constants import AnimationStyleType, AnimationToolType
from ui_components.constants import DEFAULT_SHOT_MOTION_VALUES, DefaultProjectSettingParams, ShotMetaData
from ui_components.methods.video_methods import create_single_interpolated_clip
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.motion_module import AnimateDiffCheckpoint
from ui_components.models import InternalFrameTimingObject, InternalShotObject
from utils import st_memory
import numpy as np
import matplotlib.pyplot as plt
import os
import requests


def animation_style_element(shot_uuid):
    disable_generate = False
    help = ""
    motion_modules = AnimateDiffCheckpoint.get_name_list()
    variant_count = 1
    current_animation_style = AnimationStyleType.CREATIVE_INTERPOLATION.value    # setting a default value
    data_repo = DataRepo()
    
    shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
    st.session_state['project_uuid'] = str(shot.project.uuid)
    timing_list: List[InternalFrameTimingObject] = shot.timing_list
    buffer = 4
    

    settings = {
        'animation_tool': AnimationToolType.ANIMATEDIFF.value,
    }

    interpolation_style = 'ease-in-out'
    
    advanced1, advanced2, advanced3 = st.columns([1.0,1.5, 1.0])
    with advanced1:
        st.markdown("#### Individual frame settings")
   
    items_per_row = 3
    strength_of_frames = []
    distances_to_next_frames = []
    speeds_of_transitions = []
    freedoms_between_frames = []
    individual_prompts = []
    individual_negative_prompts = []
    motions_during_frames = []

    if len(timing_list) <= 1:
        st.warning("You need at least two frames to generate a video.")
        st.stop()
    
    for i in range(0, len(timing_list) , items_per_row):
        with st.container():
            grid = st.columns([2 if j%2==0 else 1 for j in range(2*items_per_row)])  # Adjust the column widths
            for j in range(items_per_row):                    
                idx = i + j
                if idx < len(timing_list):                        
                    with grid[2*j]:  # Adjust the index for image column
                        timing = timing_list[idx]
                        if timing.primary_image and timing.primary_image.location:
                            st.info(f"**Frame {idx + 1}**")
                            st.image(timing.primary_image.location, use_column_width=True)
                            
                            motion_data = DEFAULT_SHOT_MOTION_VALUES
                            # setting default parameters (fetching data from the shot if it's present)
                            if f'strength_of_frame_{shot.uuid}_{idx}' not in st.session_state:
                                shot_meta_data = shot.meta_data_dict.get(ShotMetaData.MOTION_DATA.value, json.dumps({}))
                                timing_data = json.loads(shot_meta_data).get("timing_data", [])
                                if timing_data and len(timing_data) >= idx + 1:
                                    motion_data = timing_data[idx]

                                for k, v in motion_data.items():
                                    st.session_state[f'{k}_{shot.uuid}_{idx}'] = v
                                                                                                                        
                            # settings control
                            with st.expander("Advanced settings:"):
                                strength_of_frame = st.slider("Strength of current frame:", min_value=0.25, max_value=1.0, step=0.01, key=f"strength_of_frame_widget_{shot.uuid}_{idx}", value=st.session_state[f'strength_of_frame_{shot.uuid}_{idx}'])
                                strength_of_frames.append(strength_of_frame)                                    
                                individual_prompt = st.text_input("What to include:", key=f"individual_prompt_widget_{idx}_{timing.uuid}", value=st.session_state[f'individual_prompt_{shot.uuid}_{idx}'], help="Use this sparingly, as it can have a large impact on the video and cause weird distortions.")
                                individual_prompts.append(individual_prompt)
                                individual_negative_prompt = st.text_input("What to avoid:", key=f"negative_prompt_widget_{idx}_{timing.uuid}", value=st.session_state[f'individual_negative_prompt_{shot.uuid}_{idx}'],help="Use this sparingly, as it can have a large impact on the video and cause weird distortions.")
                                individual_negative_prompts.append(individual_negative_prompt)
                                # motion_during_frame = st.slider("Motion during frame:", min_value=0.5, max_value=1.5, step=0.01, key=f"motion_during_frame_widget_{idx}_{timing.uuid}", value=st.session_state[f'motion_during_frame_{shot.uuid}_{idx}'])
                                motion_during_frame = 1.3
                                motions_during_frames.append(motion_during_frame)
                        else:                        
                            st.warning("No primary image present.")    

                    # distance, speed and freedom settings (also aggregates them into arrays)
                    with grid[2*j+1]:  # Add the new column after the image column
                        if idx < len(timing_list) - 1:                                                                       
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")                   
                            # if st.session_state[f'distance_to_next_frame_{shot.uuid}_{idx}'] is a int, make it a float
                            if isinstance(st.session_state[f'distance_to_next_frame_{shot.uuid}_{idx}'], int):
                                st.session_state[f'distance_to_next_frame_{shot.uuid}_{idx}'] = float(st.session_state[f'distance_to_next_frame_{shot.uuid}_{idx}'])
                            distance_to_next_frame = st.slider("Seconds to next frame:", min_value=0.25, max_value=6.00, step=0.25, key=f"distance_to_next_frame_widget_{idx}_{timing.uuid}", value=st.session_state[f'distance_to_next_frame_{shot.uuid}_{idx}'])                                
                            distances_to_next_frames.append(distance_to_next_frame)                                    
                            speed_of_transition = st.slider("Speed of transition:", min_value=0.45, max_value=0.7, step=0.01, key=f"speed_of_transition_widget_{idx}_{timing.uuid}", value=st.session_state[f'speed_of_transition_{shot.uuid}_{idx}'])
                            speeds_of_transitions.append(speed_of_transition)                                      
                            freedom_between_frames = st.slider("Freedom between frames:", min_value=0.2, max_value=0.95, step=0.01, key=f"freedom_between_frames_widget_{idx}_{timing.uuid}", value=st.session_state[f'freedom_between_frames_{shot.uuid}_{idx}'])
                            freedoms_between_frames.append(freedom_between_frames)
                                            
            if (i < len(timing_list) - 1)  or (len(timing_list) % items_per_row != 0):
                st.markdown("***")
    
    

    
    dynamic_strength_values, dynamic_key_frame_influence_values, dynamic_frame_distribution_values = transform_data(strength_of_frames, freedoms_between_frames, speeds_of_transitions, distances_to_next_frames)

    type_of_frame_distribution = "dynamic"
    type_of_key_frame_influence = "dynamic"
    type_of_strength_distribution = "dynamic"
    linear_frame_distribution_value = 16
    linear_key_frame_influence_value = 1.0
    linear_cn_strength_value = 1.0

    with st.sidebar:
        with st.expander("📈 Visualise motion data", expanded=True): 
            if st_memory.toggle("Open", key="open_motion_data"):
                                

                
                st.markdown("### Keyframe positions")
                keyframe_positions = get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, timing_list, linear_frame_distribution_value)                    
                keyframe_positions = [int(kf * 16) for kf in keyframe_positions]                                        
                last_key_frame_position = (keyframe_positions[-1])
                strength_values = extract_strength_values(type_of_strength_distribution, dynamic_strength_values, keyframe_positions, linear_cn_strength_value)
                key_frame_influence_values = extract_influence_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value)                                                                                                            
                weights_list, frame_numbers_list = calculate_weights(keyframe_positions, strength_values, 4, key_frame_influence_values,last_key_frame_position)                                                    
                plot_weights(weights_list, frame_numbers_list)
            
                st.markdown("***")

                bulk1, bulk2 = st.columns([1, 1])
                with bulk1:
                    st.markdown("### Bulk edit")
                with bulk2:
                    if st.button("Reset to Default", use_container_width=True, key="reset_to_default"):
                        for idx, timing in enumerate(timing_list):
                            
                            for k, v in DEFAULT_SHOT_MOTION_VALUES.items():
                                st.session_state[f'{k}_{shot.uuid}_{idx}'] = v

                        st.success("All frames have been reset to default values.")
                        st.rerun()
                                        
                what_would_you_like_to_edit = st.selectbox("What would you like to edit?", options=["Seconds to next frames", "Speed of transitions", "Freedom between frames","Strength of frames"], key="what_would_you_like_to_edit")
                if what_would_you_like_to_edit == "Seconds to next frames":
                    what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.25, max_value=6.00, step=0.25, value=1.0, key="what_to_change_it_to")
                if what_would_you_like_to_edit == "Strength of frames":
                    what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.25, max_value=1.0, step=0.01, value=0.5, key="what_to_change_it_to")
                elif what_would_you_like_to_edit == "Speed of transitions":
                    what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.45, max_value=0.7, step=0.01, value=0.6, key="what_to_change_it_to")
                elif what_would_you_like_to_edit == "Freedom between frames":
                    what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.2, max_value=0.95, step=0.01, value=0.5, key="what_to_change_it_to")
                elif what_would_you_like_to_edit == "Motion during frames":
                    what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.5, max_value=1.5, step=0.01, value=1.3, key="what_to_change_it_to")
                
                bulk1, bulk2 = st.columns([1, 1])
                with bulk1:
                    if st.button("Bulk edit", key="bulk_edit", use_container_width=True):
                        if what_would_you_like_to_edit == "Strength of frames":
                            for idx, timing in enumerate(timing_list):
                                st.session_state[f'strength_of_frame_{shot.uuid}_{idx}'] = what_to_change_it_to
                        elif what_would_you_like_to_edit == "Seconds to next frames":
                            for idx, timing in enumerate(timing_list):
                                st.session_state[f'distance_to_next_frame_{shot.uuid}_{idx}'] = what_to_change_it_to
                        elif what_would_you_like_to_edit == "Speed of transitions":
                            for idx, timing in enumerate(timing_list):
                                st.session_state[f'speed_of_transition_{shot.uuid}_{idx}'] = what_to_change_it_to
                        elif what_would_you_like_to_edit == "Freedom between frames":
                            for idx, timing in enumerate(timing_list):
                                st.session_state[f'freedom_between_frames_{shot.uuid}_{idx}'] = what_to_change_it_to
                        # elif what_would_you_like_to_edit == "Motion during frames":
                          #  for idx, timing in enumerate(timing_list):
                           #     st.session_state[f'motion_during_frame_{shot.uuid}_{idx}'] = what_to_change_it_to
                        st.rerun()
                
                st.markdown("***")
                st.markdown("### Save current settings")
                if st.button("Save current settings", key="save_current_settings",use_container_width=True,help="Settings will also be saved when you generate the animation."):
                    update_session_state_with_animation_details(shot.uuid, timing_list, strength_of_frames, distances_to_next_frames, speeds_of_transitions, freedoms_between_frames, motions_during_frames, individual_prompts, individual_negative_prompts)
                    st.success("Settings saved successfully.")
                    time.sleep(0.7)
                    st.rerun()

    # if it's in local DEVELOPMENT ENVIRONMENT
    st.markdown("***")
    st.markdown("#### Motion guidance")

    tab1, tab2, tab3  = st.tabs(["Apply LoRAs","Explore LoRAs","Train LoRAs"])
    with tab1:
        if "current_loras" not in st.session_state:
            st.session_state["current_loras"] = []

        # Initialize a single list to hold dictionaries for LoRA data
        lora_data = []

        # Check if the directory exists and list files, or use a default list
        if os.path.exists("ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora"):
            files = os.listdir("ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora")
            # remove files that start with a dot
            files = [file for file in files if not file.startswith(".")]
        else:
            files = ['zooming_in_temporal_unet.safetensors', 'cat_walking_temporal_unet.safetensors', 'playing_banjo_temporal_unet.safetensors']

        # Iterate through each current LoRA in session state
        for idx, lora in enumerate(st.session_state["current_loras"]):
            if st.session_state["current_loras"][idx] == "":
                h1, h2, h3, h4 = st.columns([1, 1, 1, 0.5])

                with h1:
                    if len(files) == 0:
                        st.error("No LoRAs found in the directory - go to Explore to download some, or drop them into ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora")
                        st.stop()
                    else:
                        # User selects the LoRA they want to use
                        which_lora = st.selectbox("Which LoRA would you like to use?", options=files, key=f"which_lora_{idx}")
                        
                with h2:
                    # User selects the strength for the LoRA
                    strength_of_lora = st.slider("How strong would you like the LoRA to be?", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"strength_of_lora_{idx}")
                    
                    # Append the selected LoRA name and strength as a dictionary to lora_data
                    lora_data.append({"lora_name": which_lora, "lora_strength": strength_of_lora})
                    # st.write(lora_data)
                with h3:
                    when_to_apply_lora = st.slider("When to apply the LoRA?", min_value=0, max_value=100, value=(0,100), step=1, key=f"when_to_apply_lora_{idx}",disabled=True,help="This feature is not yet available.")
                with h4:
                    # remove button
                    st.write("")
                    if st.button("Remove", key=f"remove_lora_{idx}"):
                        # pop the current lora from the list
                        st.session_state["current_loras"].pop(idx)
                        st.rerun()
            
        text = "Add a LoRA" if not st.session_state["current_loras"] else "Add another LoRA"
        if st.button(text, key="add_motion_guidance"):
            st.session_state["current_loras"].append("")
            st.rerun()

    with tab2:
        file_links = [
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/250_tony_stark_r64_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/250_train_r128_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/300_car_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_car_desert_48_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_car_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_jeep_driving_r32_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_man_running_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_rotation_temporal_unet.safetensors",
            "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/750_jeep_driving_r32_temporal_unet.safetensors",
            "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/300_zooming_in_temporal_unet.safetensors",
            "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_cat_walking_temporal_unet.safetensors",
            "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_playing_banjo_temporal_unet.safetensors",
            "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_woman_dancing_temporal_unet.safetensors",
            "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_zooming_out_temporal_unet.safetensors"
        ]
                
        motion_lora_url = st.selectbox("Which LoRA would you like to download?", options=file_links, key="motion_lora_url")
        if st.button("Download LoRA", key="download_lora"):
            with st.spinner("Downloading LoRA..."):
                save_directory = "ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora"
                os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
                
                # Extract the filename from the URL
                filename = motion_lora_url.split("/")[-1]
                save_path = os.path.join(save_directory, filename)
                
                # Download the file
                response = requests.get(motion_lora_url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    st.success(f"Downloaded LoRA to {save_path}")
                else:
                    st.error("Failed to download LoRA")

    with tab3:
        b1, b2 = st.columns([1, 1])
        with b1:
            st.error("This feature is not yet available.")
            # name_this_lora = st.text_input("Name this LoRA", key="name_this_lora")
            # describe_the_motion = st.text_area("Describe the motion", key="describe_the_motion")
            # training_video = st.file_uploader("Upload a video to train a new LoRA", type=["mp4"])

            # if st.button("Train LoRA", key="train_lora", use_container_width=True):
            #     st.write("Training LoRA")
                                
    st.markdown("***")
    st.markdown("#### Overall style settings")

    sd_model_list = [
        "Realistic_Vision_V5.1.safetensors",
        "anything-v3-fp16-pruned.safetensors",
        "counterfeitV30_25.safetensors",
        "Deliberate_v2.safetensors",
        "dreamshaper_8.safetensors",
        "epicrealism_pureEvolutionV5.safetensors",
        "majicmixRealistic_v6.safetensors",
        "perfectWorld_v6Baked.safetensors",            
        "wd-illusion-fp16.safetensors",
        "aniverse_v13.safetensors",
        "juggernaut_v21.safetensor"
    ]
    # remove .safe tensors from the end of each model name
    # motion_scale = st_memory.slider("Motion scale:", min_value=0.0, max_value=2.0, value=1.3, step=0.01, key="motion_scale")        
    z1,z2 = st.columns([1, 1])
    with z1:
        sd_model = st_memory.selectbox("Which model would you like to use?", options=sd_model_list, key="sd_model_video")

    e1, e2, e3 = st.columns([1, 1,1])

    with e1:        
        strength_of_adherence = st_memory.slider("How much would you like to force adherence to the input images?", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="stregnth_of_adherence")
    with e2:
        st.info("Higher values may cause flickering and sudden changes in the video. Lower values may cause the video to be less influenced by the input images but can also fix colouring issues.")

    f1, f2, f3 = st.columns([1, 1, 1])
    
    with f1:
        overall_positive_prompt = st_memory.text_area("What would you like to see in the videos?", value="", key="positive_prompt_video")
    with f2:
        overall_negative_prompt = st_memory.text_area("What would you like to avoid in the videos?", value="", key="negative_prompt_video")
    
    with f3:
        st.write("")
        st.write("")
        st.info("Use these sparingly, as they can have a large impact on the video. You can also edit them for individual frames in the advanced settings above.")
        soft_scaled_cn_weights_multiplier = ""

    st.markdown("***")
    st.markdown("#### Overall motion settings")
    h1, h2, h3 = st.columns([0.5, 1.5, 1])
    with h1:
        
        type_of_motion_context = st.radio("Type of motion context:", options=["Low", "Standard", "High"], key="type_of_motion_context", horizontal=False, index=1)
        
    with h2: 
        st.info("This is how much the motion will be informed by the previous and next frames. 'High' can make it smoother but increase artifacts - while 'Low' make the motion less smooth but removes artifacts. Naturally, we recommend Standard.")
    
    i1, i2, i3 = st.columns([1, 1, 1])
    with i1:
        motion_scale = st.slider("Motion scale:", min_value=0.0, max_value=2.0, value=1.3, step=0.01, key="motion_scale")

    with i2:
        st.info("This is how much the video moves. Above 1.4 gets jittery, below 0.8 makes it too fluid.")
    context_length = 16
    context_stride = 2
    context_overlap = 4

    if type_of_motion_context == "Low":
        context_length = 16
        context_stride = 1
        context_overlap = 2

    elif type_of_motion_context == "Standard":
        context_length = 16
        context_stride = 2
        context_overlap = 4
    
    elif type_of_motion_context == "High":
        context_length = 16
        context_stride = 4
        context_overlap = 4
    

    relative_ipadapter_strength = 1.0
    relative_cn_strength = 0.0
    project_settings = data_repo.get_project_setting(shot.project.uuid)
    width = project_settings.width
    height = project_settings.height
    img_dimension = f"{width}x{height}"

    # st.write(dynamic_frame_distribution_values)
    dynamic_frame_distribution_values = [float(value) * 16 for value in dynamic_frame_distribution_values]
    
    individual_prompts = format_frame_prompts_with_buffer(dynamic_frame_distribution_values, individual_prompts, buffer)    
    individual_negative_prompts = format_frame_prompts_with_buffer(dynamic_frame_distribution_values, individual_negative_prompts, buffer)
                
    multipled_base_end_percent = 0.05 * (strength_of_adherence * 10)
    multipled_base_adapter_strength = 0.05 * (strength_of_adherence * 20)
    
    motion_scales = format_motion_strengths_with_buffer(dynamic_frame_distribution_values, motions_during_frames, buffer)
        
    settings.update(
        ckpt=sd_model,
        width=width,
        height=height,
        buffer=4,
        motion_scale=motion_scale,
        motion_scales=motion_scales,
        image_dimension=img_dimension,
        output_format="video/h264-mp4",
        prompt=overall_positive_prompt,
        negative_prompt=overall_negative_prompt,
        interpolation_type=interpolation_style,
        stmfnet_multiplier=2,
        relative_ipadapter_strength=relative_ipadapter_strength,
        relative_cn_strength=relative_cn_strength,      
        type_of_strength_distribution=type_of_strength_distribution,
        linear_strength_value=str(linear_cn_strength_value),
        dynamic_strength_values=str(dynamic_strength_values),
        linear_frame_distribution_value=linear_frame_distribution_value,
        dynamic_frame_distribution_values=dynamic_frame_distribution_values,
        type_of_frame_distribution=type_of_frame_distribution,                
        type_of_key_frame_influence=type_of_key_frame_influence,
        linear_key_frame_influence_value=float(linear_key_frame_influence_value),
        dynamic_key_frame_influence_values=dynamic_key_frame_influence_values,
        normalise_speed=True,
        ipadapter_noise=0.3,
        animation_style=AnimationStyleType.CREATIVE_INTERPOLATION.value,
        context_length=context_length,
        context_stride=context_stride,
        context_overlap=context_overlap,
        multipled_base_end_percent=multipled_base_end_percent,
        multipled_base_adapter_strength=multipled_base_adapter_strength,
        individual_prompts=individual_prompts,
        individual_negative_prompts=individual_negative_prompts,
        animation_stype=AnimationStyleType.CREATIVE_INTERPOLATION.value,
        # make max_frame the final value in the dynamic_frame_distribution_values
        max_frames=str(dynamic_frame_distribution_values[-1]),
        lora_data=lora_data
    )
    
    st.markdown("***")
    st.markdown("#### Generation Settings")

    position = "generate_vid"
    animate_col_1, animate_col_2, _ = st.columns([1, 1, 2])
    with animate_col_1:
        variant_count = st.number_input("How many variants?", min_value=1, max_value=5, value=1, step=1, key="variant_count")
        
        if "generate_vid_generate_inference" in st.session_state and st.session_state["generate_vid_generate_inference"]:
            # last keyframe position * 16
            duration = float(dynamic_frame_distribution_values[-1] / 16)
            data_repo.update_shot(uuid=shot.uuid, duration=duration)
            update_session_state_with_animation_details(shot.uuid, timing_list, strength_of_frames, distances_to_next_frames, speeds_of_transitions, freedoms_between_frames, motions_during_frames, individual_prompts, individual_negative_prompts)
            vid_quality = "full"    # TODO: add this if video_resolution == "Full Resolution" else "preview"
            st.success("Generating clip - see status in the Generation Log in the sidebar. Press 'Refresh log' to update.")

            positive_prompt = ""
            append_to_prompt = ""       # TODO: add this
            for idx, timing in enumerate(timing_list):
                if timing.primary_image and timing.primary_image.location:
                    b = timing.primary_image.inference_params
                    prompt = b.get("prompt", "") if b else ""
                    prompt += append_to_prompt
                    frame_prompt = f"{idx * linear_frame_distribution_value}_" + prompt
                    positive_prompt += ":" + frame_prompt if positive_prompt else frame_prompt
                else:
                    st.error("Please generate primary images")
                    time.sleep(0.7)
                    st.rerun()

            create_single_interpolated_clip(
                shot_uuid,
                vid_quality,
                settings,
                variant_count
            )

            toggle_generate_inference(position)
            st.rerun()

        st.button("Generate Animation Clip", key="generate_animation_clip", disabled=disable_generate, help=help, on_click=lambda: toggle_generate_inference(position))

    with animate_col_2:
            number_of_frames = len(timing_list)
            if height==width:
                cost_per_key_frame = 0.035
            else:
                cost_per_key_frame = 0.045

            cost_per_generation = cost_per_key_frame * number_of_frames * variant_count
            # st.info(f"Generating a video with {number_of_frames} frames in the cloud will cost c. ${cost_per_generation:.2f} USD.")

def toggle_generate_inference(position):
    if position + '_generate_inference' not in st.session_state:
        st.session_state[position + '_generate_inference'] = True
    else:
        st.session_state[position + '_generate_inference'] = not st.session_state[position + '_generate_inference']

def update_session_state_with_animation_details(shot_uuid, timing_list, strength_of_frames, distances_to_next_frames, speeds_of_transitions, freedoms_between_frames, motions_during_frames, individual_prompts, individual_negative_prompts):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    meta_data = shot.meta_data_dict
    timing_data = []
    for idx, timing in enumerate(timing_list):
        if idx < len(timing_list):
            st.session_state[f'strength_of_frame_{shot_uuid}_{idx}'] = strength_of_frames[idx]
            st.session_state[f'individual_prompt_{shot_uuid}_{idx}'] = individual_prompts[idx]
            st.session_state[f'individual_negative_prompt_{shot_uuid}_{idx}'] = individual_negative_prompts[idx]
            st.session_state[f'motion_during_frame_{shot_uuid}_{idx}'] = motions_during_frames[idx]
            if idx < len(timing_list) - 1:                             
                st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'] = distances_to_next_frames[idx]
                st.session_state[f'speed_of_transition_{shot_uuid}_{idx}'] = speeds_of_transitions[idx]
                st.session_state[f'freedom_between_frames_{shot_uuid}_{idx}'] = freedoms_between_frames[idx]

        # adding into the meta-data
        state_data = {
            "strength_of_frame" : strength_of_frames[idx],
            "individual_prompt" : individual_prompts[idx],
            "individual_negative_prompt" : individual_negative_prompts[idx],
            "motion_during_frame" : motions_during_frames[idx],
            "distance_to_next_frame" : distances_to_next_frames[idx] if idx < len(timing_list) - 1 else DEFAULT_SHOT_MOTION_VALUES["distance_to_next_frame"],
            "speed_of_transition" : speeds_of_transitions[idx] if idx < len(timing_list) - 1 else DEFAULT_SHOT_MOTION_VALUES["speed_of_transition"],
            "freedom_between_frames" : freedoms_between_frames[idx] if idx < len(timing_list) - 1 else DEFAULT_SHOT_MOTION_VALUES["freedom_between_frames"],
        }

        timing_data.append(state_data)

    meta_data.update({ShotMetaData.MOTION_DATA.value : json.dumps({"timing_data": timing_data})})
    data_repo.update_shot(**{"uuid": shot_uuid, "meta_data": json.dumps(meta_data)})


def format_frame_prompts_with_buffer(frame_numbers, individual_prompts, buffer):
    adjusted_frame_numbers = [frame + buffer for frame in frame_numbers]
    
    # Preprocess prompts to remove any '/' or '"' from the values
    processed_prompts = [prompt.replace("/", "").replace('"', '') for prompt in individual_prompts]
    
    # Format the adjusted frame numbers and processed prompts
    formatted = ', '.join(f'"{int(frame)}": "{prompt}"' for frame, prompt in zip(adjusted_frame_numbers, processed_prompts))
    return formatted

'''
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

'''
def extract_strength_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):

    if type_of_key_frame_influence == "dynamic":
        # Process the dynamic_key_frame_influence_values depending on its format
        if isinstance(dynamic_key_frame_influence_values, str):
            dynamic_values = eval(dynamic_key_frame_influence_values)
        else:
            dynamic_values = dynamic_key_frame_influence_values

        # Iterate through the dynamic values and convert tuples with two values to three values
        dynamic_values_corrected = []
        for value in dynamic_values:
            if len(value) == 2:
                value = (value[0], value[1], value[0])
            dynamic_values_corrected.append(value)

        return dynamic_values_corrected
    else:
        # Process for linear or other types
        if len(linear_key_frame_influence_value) == 2:
            linear_key_frame_influence_value = (linear_key_frame_influence_value[0], linear_key_frame_influence_value[1], linear_key_frame_influence_value[0])
        return [linear_key_frame_influence_value for _ in range(len(keyframe_positions) - 1)]

'''
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
    dynamic_strength_values = settings['dynamic_strength_values']
    interpolation_type = settings['interpolation_type']
    ckpt = settings['ckpt']
    motion_scale = settings['motion_scale']    
    relative_ipadapter_strength = settings['relative_ipadapter_strength']
    relative_ipadapter_influence = settings['relative_ipadapter_influence']
    image_dimension = settings['image_dimension']
    output_format = settings['output_format']
    # soft_scaled_cn_weights_multiplier = settings['soft_scaled_cn_weights_multiplier']
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
            node['widgets_values'][9] = dynamic_strength_values
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
            
'''
def update_interpolation_settings(values=None, timing_list=None):
    default_values = {
        'type_of_frame_distribution': 0,
        'frames_per_keyframe': 16,
        'type_of_key_frame_influence': 0,
        'length_of_key_frame_influence': 1.0,
        'type_of_cn_strength_distribution': 0,
        'linear_cn_strength_value': (0.0,0.7),
        'linear_frame_distribution_value': 16,
        'linear_key_frame_influence_value': 1.0,
        'interpolation_style': 0,
        'motion_scale': 1.0,            
        'negative_prompt_video': 'bad image, worst quality',        
        'ip_adapter_strength': 1.0,
        'ip_adapter_influence': 1.0,
        'soft_scaled_cn_weights_multiple_video': 0.85
    }

    for idx in range(0, len(timing_list)):
        default_values[f'dynamic_frame_distribution_values_{idx}'] = (idx) * 16
        default_values[f'dynamic_key_frame_influence_values_{idx}'] = 1.0
        default_values[f'dynamic_strength_values_{idx}'] = (0.0,0.7)

    for key, default_value in default_values.items():
        st.session_state[key] = values.get(key, default_value) if values and values.get(key) is not None else default_value
        # print(f"{key}: {st.session_state[key]}")


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
        
def extract_influence_values(type_of_key_frame_influence, dynamic_key_frame_influence_values, keyframe_positions, linear_key_frame_influence_value):
    # Check and convert linear_key_frame_influence_value if it's a float or string float        
    # if it's a string that starts with a parenthesis, convert it to a tuple
    if isinstance(linear_key_frame_influence_value, str) and linear_key_frame_influence_value[0] == "(":
        linear_key_frame_influence_value = eval(linear_key_frame_influence_value)


    if not isinstance(linear_key_frame_influence_value, tuple):
        if isinstance(linear_key_frame_influence_value, (float, str)):
            try:
                value = float(linear_key_frame_influence_value)
                linear_key_frame_influence_value = (value, value)
            except ValueError:
                raise ValueError("linear_key_frame_influence_value must be a float or a string representing a float")

    number_of_outputs = len(keyframe_positions)

    if type_of_key_frame_influence == "dynamic":
        # Convert list of individual float values into tuples
        if all(isinstance(x, float) for x in dynamic_key_frame_influence_values):
            dynamic_values = [(value, value) for value in dynamic_key_frame_influence_values]
        elif isinstance(dynamic_key_frame_influence_values[0], str) and dynamic_key_frame_influence_values[0] == "(":
            string_representation = ''.join(dynamic_key_frame_influence_values)
            dynamic_values = eval(f'[{string_representation}]')
        else:
            dynamic_values = dynamic_key_frame_influence_values if isinstance(dynamic_key_frame_influence_values, list) else [dynamic_key_frame_influence_values]
        return dynamic_values[:number_of_outputs]
    else:
        return [linear_key_frame_influence_value for _ in range(number_of_outputs)]


def get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value):
    if type_of_frame_distribution == "dynamic":
        # Check if the input is a string or a list
        if isinstance(dynamic_frame_distribution_values, str):
            # Sort the keyframe positions in numerical order
            return sorted([int(kf.strip()) for kf in dynamic_frame_distribution_values.split(',')])
        elif isinstance(dynamic_frame_distribution_values, list):
            return sorted(dynamic_frame_distribution_values)
    else:
        # Calculate the number of keyframes based on the total duration and linear_frames_per_keyframe
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

def calculate_weights(keyframe_positions, strength_values, buffer, key_frame_influence_values,last_key_frame_position):

    def calculate_influence_frame_number(key_frame_position, next_key_frame_position, distance):
        # Calculate the absolute distance between key frames
        key_frame_distance = abs(next_key_frame_position - key_frame_position)
        
        # Apply the distance multiplier
        extended_distance = key_frame_distance * distance

        # Determine the direction of influence based on the positions of the key frames
        if key_frame_position < next_key_frame_position:
            # Normal case: influence extends forward
            influence_frame_number = key_frame_position + extended_distance
        else:
            # Reverse case: influence extends backward
            influence_frame_number = key_frame_position - extended_distance
        
        # Return the result rounded to the nearest integer
        return round(influence_frame_number)

    def find_curve(batch_index_from, batch_index_to, strength_from, strength_to, interpolation,revert_direction_at_midpoint, last_key_frame_position,i, number_of_items,buffer):

        # Initialize variables based on the position of the keyframe
        range_start = batch_index_from
        range_end = batch_index_to
        # if it's the first value, set influence range from 1.0 to 0.0

        
        if i == number_of_items - 1:
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
        if range_end > last_key_frame_position and i < number_of_items - 1:
            drop_count = range_end - last_key_frame_position
            weights = weights[:-drop_count]
            frame_numbers = frame_numbers[:-drop_count]

        return weights, frame_numbers 
    
    weights_list = []
    frame_numbers_list = []
    

    for i in range(len(keyframe_positions)):

        keyframe_position = keyframe_positions[i]                                    
        interpolation = "ease-in-out"
        # strength_from = strength_to = 1.0

        if i == 0: # first image 

            # GET IMAGE AND KEYFRAME INFLUENCE VALUES        
                       
            key_frame_influence_from, key_frame_influence_to = key_frame_influence_values[i]      
                  
            start_strength, mid_strength, end_strength = strength_values[i]
                            
            keyframe_position = keyframe_positions[i]
            next_key_frame_position = keyframe_positions[i+1]
            
            batch_index_from = keyframe_position     
            
            batch_index_to_excl = calculate_influence_frame_number(keyframe_position, next_key_frame_position, key_frame_influence_to)
            
            
            weights, frame_numbers = find_curve(batch_index_from, batch_index_to_excl, mid_strength, end_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
            # interpolation = "ease-in"                                
        
        elif i == len(keyframe_positions) - 1:  # last image

            
            # GET IMAGE AND KEYFRAME INFLUENCE VALUES                           


            key_frame_influence_from,key_frame_influence_to = key_frame_influence_values[i]       
            start_strength, mid_strength, end_strength = strength_values[i]
            # strength_from, strength_to = cn_strength_values[i-1]

            keyframe_position = keyframe_positions[i]
            previous_key_frame_position = keyframe_positions[i-1]


            batch_index_from = calculate_influence_frame_number(keyframe_position, previous_key_frame_position, key_frame_influence_from)

            batch_index_to_excl = keyframe_position
            weights, frame_numbers = find_curve(batch_index_from, batch_index_to_excl, start_strength, mid_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                                    
            # interpolation =  "ease-out"                                
        
        else:  # middle images
            

            # GET IMAGE AND KEYFRAME INFLUENCE VALUES              
            key_frame_influence_from,key_frame_influence_to = key_frame_influence_values[i]                              
            start_strength, mid_strength, end_strength = strength_values[i]
            keyframe_position = keyframe_positions[i]
                        
            # CALCULATE WEIGHTS FOR FIRST HALF
            previous_key_frame_position = keyframe_positions[i-1]   
            batch_index_from = calculate_influence_frame_number(keyframe_position, previous_key_frame_position, key_frame_influence_from)                
            batch_index_to_excl = keyframe_position                
            first_half_weights, first_half_frame_numbers = find_curve(batch_index_from, batch_index_to_excl, start_strength, mid_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)                
            
            # CALCULATE WEIGHTS FOR SECOND HALF                
            next_key_frame_position = keyframe_positions[i+1]
            batch_index_from = keyframe_position
            batch_index_to_excl = calculate_influence_frame_number(keyframe_position, next_key_frame_position, key_frame_influence_to)                                
            second_half_weights, second_half_frame_numbers = find_curve(batch_index_from, batch_index_to_excl, mid_strength, end_strength, interpolation, False, last_key_frame_position, i, len(keyframe_positions), buffer)
            
            # COMBINE FIRST AND SECOND HALF
            weights = np.concatenate([first_half_weights, second_half_weights])                
            frame_numbers = np.concatenate([first_half_frame_numbers, second_half_frame_numbers])
        
        weights_list.append(weights)
        frame_numbers_list.append(frame_numbers)

    return weights_list, frame_numbers_list

def plot_weights(weights_list, frame_numbers_list):
    plt.figure(figsize=(12, 6))


    for i, weights in enumerate(weights_list):
        frame_numbers = frame_numbers_list[i]
        plt.plot(frame_numbers, weights, label=f'Frame {i + 1}')

    # Plot settings
    plt.xlabel('Frame Number')
    plt.ylabel('Weight')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()



def transform_data(strength_of_frames, movements_between_frames, speeds_of_transitions, distances_to_next_frames):
    def adjust_and_invert_relative_value(middle_value, relative_value):
        if relative_value is not None:
            adjusted_value = middle_value * relative_value
            return round(middle_value - adjusted_value, 2)
        return None

    def invert_value(value):
        return round(1.0 - value, 2) if value is not None else None

    # Creating output_strength with relative and inverted start and end values
    output_strength = []
    for i, strength in enumerate(strength_of_frames):
        start_value = None if i == 0 else movements_between_frames[i - 1]
        end_value = None if i == len(strength_of_frames) - 1 else movements_between_frames[i]

        # Adjusting and inverting start and end values relative to the middle value
        adjusted_start = adjust_and_invert_relative_value(strength, start_value)
        adjusted_end = adjust_and_invert_relative_value(strength, end_value)

        output_strength.append((adjusted_start, strength, adjusted_end))

    # Creating output_speeds with inverted values
    output_speeds = [(None, None) for _ in range(len(speeds_of_transitions) + 1)]
    for i in range(len(speeds_of_transitions)):
        current_tuple = list(output_speeds[i])
        next_tuple = list(output_speeds[i + 1])

        inverted_speed = invert_value(speeds_of_transitions[i])
        current_tuple[1] = inverted_speed * 2
        next_tuple[0] = inverted_speed * 2

        output_speeds[i] = tuple(current_tuple)
        output_speeds[i + 1] = tuple(next_tuple)

    # Creating cumulative_distances
    cumulative_distances = [0]
    for distance in distances_to_next_frames:
        cumulative_distances.append(cumulative_distances[-1] + distance)
                                        
    return output_strength, output_speeds, cumulative_distances



def format_motion_strengths_with_buffer(frame_numbers, motion_strengths, buffer):
    # Adjust the first frame number to 0 and shift the others by the buffer
    adjusted_frame_numbers = [0] + [frame + buffer for frame in frame_numbers[1:]]
    
    # Format the adjusted frame numbers and strengths
    formatted = ', '.join(f'{frame}:({strength})' for frame, strength in zip(adjusted_frame_numbers, motion_strengths))
    return formatted