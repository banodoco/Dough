import json
import tarfile
import time
import zipfile
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
# import re
import re


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
        
    settings = {
        'animation_tool': AnimationToolType.ANIMATEDIFF.value,
    }
    
    st.markdown("### 🎥 Generate animations  _________")  
    st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")

    with st.container():
        advanced1, advanced2, advanced3 = st.columns([1.0,1.5, 1.0])

        with advanced1:
            st.markdown("##### Individual frame settings")
                    
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

        open_advanced_settings = st_memory.toggle("Open all advanced settings", key="advanced_settings", value=False)

        # SET DEFAULT VALUES FOR LORA, MODEL, ADHERENCE, PROMPTS, MOTION CONTEXT

        if f'lora_data_{shot.uuid}' not in st.session_state:
            st.session_state[f'lora_data_{shot.uuid}'] = []

        if f'strength_of_adherence_value_{shot.uuid}' not in st.session_state:
            st.session_state[f'strength_of_adherence_value_{shot.uuid}'] = 0.3

        if f'type_of_motion_context_index_{shot.uuid}' not in st.session_state:
            st.session_state[f'type_of_motion_context_index_{shot.uuid}'] = 0

        if f'positive_prompt_video_{shot.uuid}' not in st.session_state:
            st.session_state[f"positive_prompt_video_{shot.uuid}"] = ""

        if f'negative_prompt_video_{shot.uuid}' not in st.session_state:
            st.session_state[f"negative_prompt_video_{shot.uuid}"] = ""

        if f'ckpt_{shot.uuid}' not in st.session_state:
            st.session_state[f'ckpt_{shot.uuid}'] = ""
                
        
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
                                with st.expander("Advanced settings:", expanded=open_advanced_settings):
                                
                                    individual_prompt = st.text_input("What to include:", key=f"individual_prompt_widget_{idx}_{timing.uuid}", value=st.session_state[f'individual_prompt_{shot.uuid}_{idx}'], help="Use this sparingly, as it can have a large impact on the video and cause weird distortions.")
                                    individual_prompts.append(individual_prompt)
                                    individual_negative_prompt = st.text_input("What to avoid:", key=f"negative_prompt_widget_{idx}_{timing.uuid}", value=st.session_state[f'individual_negative_prompt_{shot.uuid}_{idx}'],help="Use this sparingly, as it can have a large impact on the video and cause weird distortions.")
                                    individual_negative_prompts.append(individual_negative_prompt)
                                    strength1, strength2 = st.columns([1, 1])
                                    with strength1:
                                        strength_of_frame = st.slider("Strength of current frame:", min_value=0.25, max_value=1.0, step=0.01, key=f"strength_of_frame_widget_{shot.uuid}_{idx}", value=st.session_state[f'strength_of_frame_{shot.uuid}_{idx}'])
                                        strength_of_frames.append(strength_of_frame)         
                                    with strength2:
                                        motion_during_frame = st.slider("Motion during frame:", min_value=0.5, max_value=1.5, step=0.01, key=f"motion_during_frame_widget_{idx}_{timing.uuid}", value=st.session_state[f'motion_during_frame_{shot.uuid}_{idx}'])                            
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
                                freedom_between_frames = st.slider("Freedom between frames:", min_value=0.05, max_value=0.95, step=0.01, key=f"freedom_between_frames_widget_{idx}_{timing.uuid}", value=st.session_state[f'freedom_between_frames_{shot.uuid}_{idx}'])
                                freedoms_between_frames.append(freedom_between_frames)
                                                
                if (i < len(timing_list) - 1)  or (len(timing_list) % items_per_row != 0):
                    st.markdown("***")
        
            
        st.markdown("##### Style model")
        tab1, tab2   = st.tabs(["Choose Model","Download Models"])
        
        checkpoints_dir = "ComfyUI/models/checkpoints"

        # List all files in the directory
        all_files = os.listdir(checkpoints_dir)


        if len(all_files) == 0:
            model_files = ['Realistic_Vision_V5.1.safetensors']

        else:
            # Filter files to only include those with .safetensors and .ckpt extensions
            
            model_files = [file for file in all_files if file.endswith('.safetensors') or file.endswith('.ckpt')]
            # drop all files that contain xl
            model_files = [file for file in model_files if "xl" not in file]


        current_model_index = model_files.index(st.session_state[f'ckpt_{shot.uuid}']) if st.session_state[f'ckpt_{shot.uuid}'] in model_files else 0

        with tab1:

            model1, model2 = st.columns([1, 1])
            with model1:
                if model_files and len(model_files):
                    sd_model = st_memory.selectbox("Which model would you like to use?", options=model_files, key="sd_model_video",index=current_model_index)
                else:
                    sd_model = ""
            with model2:
                if len(all_files) == 0:
                    st.write("")
                    st.info("This is the default model - to download more, go to the Download Models tab.")
                else:
                    st.write("")
                    st.info("To download more models, go to the Download Models tab.")

            # if it's in sd_model-list, just pass the name. If not, stick checkpoints_dir in front of it        
            sd_model = checkpoints_dir + "/" + sd_model
        
        with tab2:
            # Mapping of model names to their download URLs
            sd_model_dict = {
                "Anything V3 FP16 Pruned": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/anything-v3-fp16-pruned.safetensors.tar",
                "Deliberate V2": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/Deliberate_v2.safetensors.tar",
                "Dreamshaper 8": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/dreamshaper_8.safetensors.tar",
                "epicrealism_pureEvolutionV5": "https://civitai.com/api/download/models/134065",
                "majicmixRealistic_v6": "https://civitai.com/api/download/models/94640",            
            }

            where_to_get_model = st.radio("Where would you like to get the model from?", options=["Our list", "Upload a model", "From a URL"], key="where_to_get_model")

            if where_to_get_model == "Our list":
                # Use the keys (model names) for the selection box
                model_name_selected = st.selectbox("Which model would you like to download?", options=list(sd_model_dict.keys()), key="model_to_download")
                
                if st.button("Download Model", key="download_model"):
                    with st.spinner("Downloading model..."):
                        download_bar = st.progress(0, text="")
                        save_directory = "ComfyUI/models/checkpoints"
                        os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
                        
                        # Retrieve the URL using the selected model name
                        model_url = sd_model_dict[model_name_selected]
                        
                        # Download the model and save it to the directory
                        response = requests.get(model_url, stream=True)
                        zip_filename = model_url.split("/")[-1]
                        filepath = os.path.join(save_directory, zip_filename)
                        print("filepath: ", filepath)
                        if response.status_code == 200:
                            total_size = int(response.headers.get('content-length', 0))
                            
                            with open(filepath, 'wb') as f:
                                received_bytes = 0

                                for data in response.iter_content(chunk_size=8192):
                                    f.write(data)
                                    received_bytes += len(data)
                                    progress = received_bytes / total_size
                                    download_bar.progress(progress)

                            st.success(f"Downloaded {model_name_selected} to {save_directory}")
                            download_bar.empty()
                            
                        if model_url.endswith(".zip") or model_url.endswith(".tar"):
                            st.success("Extracting the zip file. Please wait...")
                            new_filepath = filepath.replace(zip_filename, "")
                            if model_url.endswith(".zip"):
                                with zipfile.ZipFile(f"{filepath}", "r") as zip_ref:
                                    zip_ref.extractall(new_filepath)
                            else:
                                with tarfile.open(f"{filepath}", "r") as tar_ref:
                                    tar_ref.extractall(new_filepath)
                            
                            # os.remove(filepath)
                            st.rerun()
                        else:
                            st.error("Failed to download model")

            elif where_to_get_model == "Upload a model":
                st.info("It's simpler to just drop this into the ComfyUI/models/checkpoints directory.")
            
            elif where_to_get_model == "From a URL":
                text1, text2 = st.columns([1, 1])
                with text1:

                    text_input = st.text_input("Enter the URL of the model", key="text_input")
                with text2:
                    st.info("Make sure to get the download url of the model. \n\n For example, from Civit, this should look like this: https://civitai.com/api/download/models/179446. \n\n While from Hugging Face, it should look like this: https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors")
                if st.button("Download Model", key="download_model"):
                    with st.spinner("Downloading model..."):
                        save_directory = "ComfyUI/models/checkpoints"
                        os.makedirs(save_directory, exist_ok=True)
                        response = requests.get(text_input)
                        if response.status_code == 200:
                            with open(os.path.join(save_directory, text_input.split("/")[-1]), 'wb') as f:
                                f.write(response.content)
                            st.success(f"Downloaded model to {save_directory}")
                        else:
                            st.error("Failed to download model")
                
        # if it's in local DEVELOPMENT ENVIRONMENT
        st.markdown("***")        
        st.markdown("##### Motion guidance")
        tab1, tab2, tab3  = st.tabs(["Apply LoRAs","Download LoRAs","Train LoRAs"])

        lora_data = []
        lora_file_dest = "ComfyUI/models/animatediff_motion_lora"
        with tab1:
           
            # Initialize a single list to hold dictionaries for LoRA data
            #lora_data = []
            # Check if the directory exists and list files, or use a default list
            if os.path.exists(lora_file_dest):
                files = os.listdir(lora_file_dest)
                # remove files that start with a dot
                files = [file for file in files if not file.startswith(".")]
            else:
                files = []

            # Iterate through each current LoRA in session state
            if len(files) == 0:
                st.error("No LoRAs found in the directory - go to Explore to download some, or drop them into ComfyUI/models/animatediff_motion_lora")                    
                if st.button("Check again", key="check_again"):
                    st.rerun()
            else:
                for idx, lora in enumerate(st.session_state[f"lora_data_{shot.uuid}"]):
                    h1, h2, h3, h4 = st.columns([1, 1, 1, 0.5])
                    with h1:
                        which_lora = st.selectbox("Which LoRA would you like to use?", options=files, key=f"which_lora_{idx}")                                                    
                    with h2:
                        # User selects the strength for the LoRA
                        strength_of_lora = st.slider("How strong would you like the LoRA to be?", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"strength_of_lora_{idx}")
                        
                        # Append the selected LoRA name and strength as a dictionary to lora_data
                        lora_data.append({"filename": which_lora, "lora_strength": strength_of_lora, "filepath": lora_file_dest + "/" + which_lora})
                        # st.write(lora_data)
                    with h3:
                        when_to_apply_lora = st.slider("When to apply the LoRA?", min_value=0, max_value=100, value=(0,100), step=1, key=f"when_to_apply_lora_{idx}",disabled=True,help="This feature is not yet available.")
                    with h4:
                        # remove button
                        st.write("")
                        if st.button("Remove", key=f"remove_lora_{idx}"):
                            # pop the current lora from the list
                            st.session_state[f"lora_data_{shot.uuid}"].pop(idx)
                            st.rerun()
                # st.write(lora_data)
                if len(st.session_state[f"lora_data_{shot.uuid}"]) == 0:
                    text = "Add a LoRA"
                else:
                    text = "Add another LoRA"
                if st.button(text, key="add_motion_guidance"):
                    st.session_state[f"lora_data_{shot.uuid}"].append("")
                    st.rerun()
        with tab2:
            text1, text2 = st.columns([1, 1])
            with text1:
                where_to_download_from = st.radio("Where would you like to get the LoRA from?", options=["Our list", "From a URL","Upload a LoRA"], key="where_to_download_from", horizontal=True)

            if where_to_download_from == "Our list":
                with text1:
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
                            
                    which_would_you_like_to_download = st.selectbox("Which LoRA would you like to download?", options=file_links, key="which_would_you_like_to_download")
                    if st.button("Download LoRA", key="download_lora"):
                        with st.spinner("Downloading LoRA..."):
                            save_directory = "ComfyUI/models/animatediff_motion_lora"
                            os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
                            
                            # Extract the filename from the URL
                            filename = which_would_you_like_to_download.split("/")[-1]
                            save_path = os.path.join(save_directory, filename)
                            
                            # Download the file
                            download_lora_bar = st.progress(0, text="")
                            response = requests.get(which_would_you_like_to_download, stream=True)
                            if response.status_code == 200:
                                total_size = int(response.headers.get('content-length', 0))
                                with open(save_path, 'wb') as f:
                                    received_bytes = 0
                                    
                                    for data in response.iter_content(chunk_size=8192):
                                        f.write(data)
                                        received_bytes += len(data)
                                        progress = received_bytes / total_size
                                        download_lora_bar.progress(progress)
                                        
                                st.success(f"Downloaded LoRA to {save_path}")
                                download_lora_bar.empty()
                                st.rerun()
                            else:
                                st.error("Failed to download LoRA")
            
            elif where_to_download_from == "From a URL":
                
                with text1:
                    text_input = st.text_input("Enter the URL of the LoRA", key="text_input_lora")
                with text2:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.info("Make sure to get the download url of the LoRA. \n\n For example, from Hugging Face, it should look like this: https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors")
                with text1:
                    if st.button("Download LoRA", key="download_lora"):
                        with st.spinner("Downloading LoRA..."):
                            save_directory = "ComfyUI/models/animatediff_motion_lora"
                            os.makedirs(save_directory, exist_ok=True)
                            response = requests.get(text_input)
                            if response.status_code == 200:
                                with open(os.path.join(save_directory, text_input.split("/")[-1]), 'wb') as f:
                                    f.write(response.content)
                                st.success(f"Downloaded LoRA to {save_directory}")
                            else:
                                st.error("Failed to download LoRA")
            elif where_to_download_from == "Upload a LoRA":
                st.info("It's simpler to just drop this into the ComfyUI/models/animatediff_motion_lora directory.")
        with tab3:
            b1, b2 = st.columns([1, 1])
            with b1:
                st.error("This feature is not yet available.")
                name_this_lora = st.text_input("Name this LoRA", key="name_this_lora")
                describe_the_motion = st.text_area("Describe the motion", key="describe_the_motion")
                training_video = st.file_uploader("Upload a video to train a new LoRA", type=["mp4"])

                if st.button("Train LoRA", key="train_lora", use_container_width=True):
                    st.write("Training LoRA")
                                                    

        st.markdown("***")
        st.markdown("##### Overall style settings")

        e1, e2, e3 = st.columns([1, 1,1])
        with e1:        
            strength_of_adherence = st.slider("How much would you like to force adherence to the input images?", min_value=0.0, max_value=1.0, step=0.01, key="strength_of_adherence", value=st.session_state[f"strength_of_adherence_value_{shot.uuid}"])
        with e2:
            st.info("Higher values may cause flickering and sudden changes in the video. Lower values may cause the video to be less influenced by the input images but can lead to smoother motion and better colours.")

        f1, f2, f3 = st.columns([1, 1, 1])
        with f1:
            overall_positive_prompt = st.text_area("What would you like to see in the videos?", value=st.session_state[f"positive_prompt_video_{shot.uuid}"])
        with f2:
            overall_negative_prompt = st.text_area("What would you like to avoid in the videos?", value=st.session_state[f"negative_prompt_video_{shot.uuid}"])
        
        with f3:
            st.write("")
            st.write("")
            st.info("Use these sparingly, as they can have a large impact on the video. You can also edit them for individual frames in the advanced settings above.")
            soft_scaled_cn_weights_multiplier = ""

        st.markdown("***")
        st.markdown("##### Overall motion settings")
        h1, h2, h3 = st.columns([0.5, 1.5, 1])
        with h1:
            type_of_motion_context = st.radio("Type of motion context:", options=["Low", "Standard", "High"], key="type_of_motion_context", horizontal=False, index=st.session_state[f"type_of_motion_context_index_{shot.uuid}"])

        with h2: 
            st.info("This is how much the motion will be informed by the previous and next frames. 'High' can make it smoother but increase artifacts - while 'Low' make the motion less smooth but removes artifacts. Naturally, we recommend Standard.")
        st.write("")
        i1, i3,_ = st.columns([1,2,1])
        with i1:
            amount_of_motion = st.slider("Amount of motion:", min_value=0.5, max_value=1.5, step=0.01, value=1.3, key="amount_of_motion")        
            st.write("")
            if st.button("Update amount of motion", key="update_motion"):
                for idx, timing in enumerate(timing_list):
                    st.session_state[f'motion_during_frame_{shot.uuid}_{idx}'] = amount_of_motion                
                st.success("Updated amount of motion")
                time.sleep(0.3)
                st.rerun()
        with i3:
            st.write("")
            st.write("")
            st.info("This actually updates the motion during frames in the advanced settings above - but we put it here because it has a big impact on the video. You can scroll up to see the changes and tweak for individual frames.")

        type_of_frame_distribution = "dynamic"
        type_of_key_frame_influence = "dynamic"
        type_of_strength_distribution = "dynamic"
        linear_frame_distribution_value = 16
        linear_key_frame_influence_value = 1.0
        linear_cn_strength_value = 1.0            
        relative_ipadapter_strength = 1.0
        relative_cn_strength = 0.0
        project_settings = data_repo.get_project_setting(shot.project.uuid)
        width = project_settings.width
        height = project_settings.height
        img_dimension = f"{width}x{height}"
        motion_scale = 1.3
        interpolation_style = 'ease-in-out'
        buffer = 4


        (dynamic_strength_values, dynamic_key_frame_influence_values, dynamic_frame_distribution_values, 
        context_length, context_stride, context_overlap, multipled_base_end_percent, multipled_base_adapter_strength, 
        prompt_travel,  negative_prompt_travel, motion_scales) = transform_data(strength_of_frames, 
            freedoms_between_frames, speeds_of_transitions, distances_to_next_frames, type_of_motion_context, 
            strength_of_adherence,individual_prompts, individual_negative_prompts, buffer, motions_during_frames)


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
            individual_prompts=prompt_travel,
            individual_negative_prompts=negative_prompt_travel,
            animation_stype=AnimationStyleType.CREATIVE_INTERPOLATION.value,            
            max_frames=str(dynamic_frame_distribution_values[-1]),
            lora_data=lora_data
        )
        
        position = "generate_vid"
        st.markdown("***")
        st.markdown("##### Generation Settings")

        animate_col_1, animate_col_2, _ = st.columns([1, 1, 1])
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
                
            st.button("Generate Animation Clip", key="generate_animation_clip", disabled=disable_generate, help=help, on_click=lambda: toggle_generate_inference(position),type="primary")


        with st.sidebar:
                with st.expander("⚙️ Animation settings", expanded=True): 
                    if st_memory.toggle("Open", key="open_motion_data"):
                                                            
                        st.markdown("### Visualisation of current motion")
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
                            st.markdown("### Bulk edit frame settings")
                        with bulk2:
                            if st.button("Reset to Default", use_container_width=True, key="reset_to_default"):
                                for idx, timing in enumerate(timing_list):                                    
                                    for k, v in DEFAULT_SHOT_MOTION_VALUES.items():                                        
                                        st.session_state[f'{k}_{shot.uuid}_{idx}'] = v
                                                                        
                                st.success("All frames have been reset to default values.")
                                st.rerun()
                                                
                        what_would_you_like_to_edit = st.selectbox("What would you like to edit?", options=["Seconds to next frames", "Speed of transitions", "Freedom between frames","Strength of frames","Motion during frames"], key="what_would_you_like_to_edit")
                        if what_would_you_like_to_edit == "Seconds to next frames":
                            what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.25, max_value=6.00, step=0.25, value=1.0, key="what_to_change_it_to")
                        if what_would_you_like_to_edit == "Strength of frames":
                            what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.25, max_value=1.0, step=0.01, value=0.5, key="what_to_change_it_to")
                        elif what_would_you_like_to_edit == "Speed of transitions":
                            what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.45, max_value=0.7, step=0.01, value=0.6, key="what_to_change_it_to")
                        elif what_would_you_like_to_edit == "Freedom between frames":
                            what_to_change_it_to = st.slider("What would you like to change it to?", min_value=0.05, max_value=0.95, step=0.01, value=0.5, key="what_to_change_it_to")
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
                                elif what_would_you_like_to_edit == "Motion during frames":
                                    for idx, timing in enumerate(timing_list):
                                        st.session_state[f'motion_during_frame_{shot.uuid}_{idx}'] = what_to_change_it_to
                                st.rerun()
                        
                        st.markdown("***")
                        st.markdown("### Save current settings")
                        if st.button("Save current settings", key="save_current_settings",use_container_width=True,help="Settings will also be saved when you generate the animation."):
                            update_session_state_with_animation_details(shot.uuid, timing_list, strength_of_frames, distances_to_next_frames, speeds_of_transitions, freedoms_between_frames, motions_during_frames, individual_prompts, individual_negative_prompts)
                            st.success("Settings saved successfully.")
                            time.sleep(0.7)
                            st.rerun()

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



def transform_data(strength_of_frames, movements_between_frames, speeds_of_transitions, distances_to_next_frames, type_of_motion_context, strength_of_adherence, individual_prompts, individual_negative_prompts, buffer, motions_during_frames):

    # FRAME SETTINGS

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

    cumulative_distances = [int(float(value) * 16) for value in cumulative_distances]

    # MOTION CONTEXT SETTINGS

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

    # SPARSE CTRL SETTINGS

    multipled_base_end_percent = 0.05 * (strength_of_adherence * 10)
    multipled_base_adapter_strength = 0.05 * (strength_of_adherence * 20)

    # FRAME PROMPTS FORMATTING

    def format_frame_prompts_with_buffer(frame_numbers, individual_prompts, buffer):
        adjusted_frame_numbers = [frame + buffer for frame in frame_numbers]
        
        # Preprocess prompts to remove any '/' or '"' from the values
        processed_prompts = [prompt.replace("/", "").replace('"', '') for prompt in individual_prompts]
        
        # Format the adjusted frame numbers and processed prompts
        formatted = ', '.join(f'"{int(frame)}": "{prompt}"' for frame, prompt in zip(adjusted_frame_numbers, processed_prompts))
        return formatted

    # Applying format_frame_prompts_with_buffer
    formatted_individual_prompts = format_frame_prompts_with_buffer(cumulative_distances, individual_prompts, buffer)
    formatted_individual_negative_prompts = format_frame_prompts_with_buffer(cumulative_distances, individual_negative_prompts, buffer)

    # MOTION STRENGTHS FORMATTING

    adjusted_frame_numbers = [0] + [frame + buffer for frame in cumulative_distances[1:]]
    
    # Format the adjusted frame numbers and strengths
    motions_during_frames = ', '.join(f'{int(frame)}:({strength})' for frame, strength in zip(adjusted_frame_numbers, motions_during_frames))    
            
    return output_strength, output_speeds, cumulative_distances, context_length, context_stride, context_overlap, multipled_base_end_percent, multipled_base_adapter_strength, formatted_individual_prompts, formatted_individual_negative_prompts,motions_during_frames

