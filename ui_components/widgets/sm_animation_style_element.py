import time
import uuid
import os
import zipfile
import requests
import random
import string
import tarfile
from PIL import Image
import streamlit as st
from shared.constants import InternalFileType
from ui_components.methods.common_methods import save_new_image
from utils import st_memory
from ui_components.constants import DEFAULT_SHOT_MOTION_VALUES
from ui_components.methods.animation_style_methods import calculate_weights, extract_influence_values, \
    extract_strength_values, get_keyframe_positions, load_shot_settings, plot_weights, update_session_state_with_animation_details
from ui_components.methods.file_methods import get_files_in_a_directory, get_media_dimensions, save_or_host_file
from ui_components.widgets.display_element import display_motion_lora
from ui_components.methods.ml_methods import train_motion_lora
from utils.data_repo.data_repo import DataRepo
     
def animation_sidebar(shot_uuid, img_list, type_of_frame_distribution, dynamic_frame_distribution_values, linear_frame_distribution_value,\
    type_of_strength_distribution, dynamic_strength_values, linear_cn_strength_value, type_of_key_frame_influence, dynamic_key_frame_influence_values, \
       linear_key_frame_influence_value, strength_of_frames, distances_to_next_frames, speeds_of_transitions, freedoms_between_frames, \
        motions_during_frames, individual_prompts, individual_negative_prompts, default_model):
    with st.sidebar:
        with st.expander("⚙️ Animation settings", expanded=True): 
            if st_memory.toggle("Open", key="open_motion_data"):
                                                    
                st.markdown("### Visualisation of current motion")
                keyframe_positions = get_keyframe_positions(type_of_frame_distribution, dynamic_frame_distribution_values, img_list, linear_frame_distribution_value)                    
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
                        for idx, _ in enumerate(img_list):                                    
                            for k, v in DEFAULT_SHOT_MOTION_VALUES.items():                                        
                                st.session_state[f'{k}_{shot_uuid}_{idx}'] = v
                                                                
                        st.success("All frames have been reset to default values.")
                        st.rerun()
                                        
                # New feature: Selecting a range to edit
                range_to_edit = st.slider("Select the range of frames you would like to edit:",
                                        min_value=1, max_value=len(img_list),
                                        value=(1, len(img_list)), step=1, key="range_to_edit")
                edit1, edit2 = st.columns([1, 1])
                with edit1:
                    editable_entity = st.selectbox("What would you like to edit?", options=["Seconds to next frames", "Speed of transitions", "Freedom between frames","Strength of frames","Motion during frames"], key="editable_entity")
                with edit2:
                    if editable_entity == "Seconds to next frames":
                        entity_new_val = st.slider("What would you like to change it to?", min_value=0.25, max_value=6.00, step=0.25, value=1.0, key="entity_new_val_seconds")
                    elif editable_entity == "Strength of frames":
                        entity_new_val = st.slider("What would you like to change it to?", min_value=0.25, max_value=1.0, step=0.01, value=0.5, key="entity_new_val_strength")
                    elif editable_entity == "Speed of transitions":
                        entity_new_val = st.slider("What would you like to change it to?", min_value=0.45, max_value=0.7, step=0.01, value=0.6, key="entity_new_val_speed")
                    elif editable_entity == "Freedom between frames":
                        entity_new_val = st.slider("What would you like to change it to?", min_value=0.15, max_value=0.85, step=0.01, value=0.5, key="entity_new_val_freedom")
                    elif editable_entity == "Motion during frames":
                        entity_new_val = st.slider("What would you like to change it to?", min_value=0.5, max_value=1.5, step=0.01, value=1.3, key="entity_new_val_motion")
                
                if st.button("Bulk edit", key="bulk_edit", use_container_width=True):
                    start_idx, end_idx = range_to_edit
                    for idx in range(start_idx - 1, end_idx): # Adjusting index to be 0-based
                        if editable_entity == "Strength of frames":
                            st.session_state[f'strength_of_frame_{shot_uuid}_{idx}'] = entity_new_val
                        elif editable_entity == "Seconds to next frames":
                            st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'] = entity_new_val
                        elif editable_entity == "Speed of transitions":
                            st.session_state[f'speed_of_transition_{shot_uuid}_{idx}'] = entity_new_val
                        elif editable_entity == "Freedom between frames":
                            st.session_state[f'freedom_between_frames_{shot_uuid}_{idx}'] = entity_new_val
                        elif editable_entity == "Motion during frames":
                            st.session_state[f'motion_during_frame_{shot_uuid}_{idx}'] = entity_new_val
                    st.rerun()
                
                st.markdown("***")
                st.markdown("### Save current settings")
                if st.button("Save current settings", key="save_current_settings",use_container_width=True,help="Settings will also be saved when you generate the animation."):
                    update_session_state_with_animation_details(
                        shot_uuid, 
                        img_list, 
                        strength_of_frames, 
                        distances_to_next_frames, 
                        speeds_of_transitions, 
                        freedoms_between_frames, 
                        motions_during_frames, 
                        individual_prompts, 
                        individual_negative_prompts,
                        [],
                        default_model
                    )
                    st.success("Settings saved successfully.")
                    time.sleep(0.7)
                    st.rerun()

                      
def video_motion_settings(shot_uuid, img_list):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    st.markdown("***")
    st.markdown("##### Overall style settings")

    e1, _, _ = st.columns([1, 1,1])
    with e1:        
        strength_of_adherence = st.slider("How much would you like to force adherence to the input images?", min_value=0.0, max_value=1.0, step=0.01, key="strength_of_adherence", value=st.session_state[f"strength_of_adherence_value_{shot_uuid}"])

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        overall_positive_prompt = ""
        def update_prompt():
            global overall_positive_prompt
            overall_positive_prompt = st.session_state[f"positive_prompt_video_{shot_uuid}"]

        overall_positive_prompt = st.text_area(
            "What would you like to see in the videos?", 
            key="overall_positive_prompt", 
            value=st.session_state[f"positive_prompt_video_{shot_uuid}"],
            on_change=update_prompt
        )
    with f2:
        overall_negative_prompt = st.text_area(
            "What would you like to avoid in the videos?", 
            key="overall_negative_prompt", 
            value=st.session_state[f"negative_prompt_video_{shot_uuid}"]
        )
    
    with f3:
        st.write("")
        st.write("")
        st.info("Use these sparingly, as they can have a large impact on the video. You can also edit them for individual frames in the advanced settings above.")

    st.markdown("***")
    st.markdown("##### Overall motion settings")
    h1, h2, h3 = st.columns([1, 0.5, 1.0])
    with h1:
        # will fix this later
        def update_motion_for_all_frames(shot_uuid, timing_list):
            amount_of_motion = st.session_state.get("amount_of_motion_overall", 1.0)  # Default to 1.0 if not set
            for idx, _ in enumerate(timing_list):
                st.session_state[f'motion_during_frame_{shot_uuid}_{idx}'] = amount_of_motion

        if f"type_of_motion_context_index_{shot_uuid}" in st.session_state and isinstance(st.session_state[f"type_of_motion_context_index_{shot_uuid}"], str):
            st.session_state[f"type_of_motion_context_index_{shot_uuid}"] = ["Low", "Standard", "High"].index(st.session_state[f"type_of_motion_context_index_{shot_uuid}"])
        type_of_motion_context = st.radio("Type of motion context:", options=["Low", "Standard", "High"], key="type_of_motion_context", horizontal=True, index=st.session_state[f"type_of_motion_context_index_{shot.uuid}"], help="This is how much the motion will be informed by the previous and next frames. 'High' can make it smoother but increase artifacts - while 'Low' make the motion less smooth but removes artifacts. Naturally, we recommend Standard.")
        st.session_state[f"amount_of_motion_{shot_uuid}"] = st.slider("Amount of motion:", min_value=0.5, max_value=1.5, step=0.01,value=1.3, key="amount_of_motion_overall", on_change=lambda: update_motion_for_all_frames(shot.uuid, img_list), help="You can also tweak this on an individual frame level in the advanced settings above.")
                 
    i1, i2, i3 = st.columns([1, 0.5, 1.5])
    with i1:
        if f'structure_control_image_{shot_uuid}' not in st.session_state:
            st.session_state[f"structure_control_image_{shot_uuid}"] = None

        if f"strength_of_structure_control_image_{shot_uuid}" not in st.session_state:
            st.session_state[f"strength_of_structure_control_image_{shot_uuid}"] = None
        control_motion_with_image = st_memory.toggle("Control motion with an image", help="This will allow you to upload images to control the motion of the video.",key=f"control_motion_with_image_{shot_uuid}")

        if control_motion_with_image:
            uploaded_image = st.file_uploader("Upload images to control motion", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
            if st.button("Add image", key="add_images"):
                if uploaded_image:
                    project_settings = data_repo.get_project_setting(shot.project.uuid)
                    width, height = project_settings.width, project_settings.height
                    # Convert the uploaded image file to PIL Image
                    uploaded_image_pil = Image.open(uploaded_image)
                    uploaded_image_pil = uploaded_image_pil.resize((width, height))
                    st.session_state[f"structure_control_image_{shot.uuid}"] = uploaded_image_pil
                    st.rerun()
                else:
                    st.warning("No images uploaded")
        else:
            st.session_state[f"structure_control_image_{shot_uuid}"] = None
    
    with i2:
        if f"structure_control_image_{shot_uuid}" in st.session_state and st.session_state[f"structure_control_image_{shot_uuid}"]:
            st.info("Control image:")                    
            st.image(st.session_state[f"structure_control_image_{shot_uuid}"])        
            st.session_state[f"strength_of_structure_control_image_{shot_uuid}"] = st.slider("Strength of control image:", min_value=0.0, max_value=1.0, step=0.01, key="strength_of_structure_control_image", value=0.5, help="This is how much the control image will influence the motion of the video.")
            if st.button("Remove image", key="remove_images"):
                st.session_state[f"structure_control_image_{shot_uuid}"] = None
                st.success("Image removed")
                st.rerun()
                
    return strength_of_adherence, overall_positive_prompt, overall_negative_prompt, type_of_motion_context

def select_motion_lora_element(shot_uuid, model_files):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    # if it's in local DEVELOPMENT ENVIRONMENT
    st.markdown("***")        
    st.markdown("##### Motion guidance")
    tab1, tab2, tab3  = st.tabs(["Apply LoRAs","Download LoRAs","Train LoRAs"])

    lora_data = []
    lora_file_dest = "ComfyUI/models/animatediff_motion_lora"
    lora_file_links = {
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/250_tony_stark_r64_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/250_train_r128_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/300_car_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_car_desert_48_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_car_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_jeep_driving_r32_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_man_running_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_rotation_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/750_jeep_driving_r32_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/300_zooming_in_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_cat_walking_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_playing_banjo_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_woman_dancing_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_zooming_out_temporal_unet.safetensors" :"https://cdn.pixabay.com/animation/2023/06/17/16/02/16-02-33-34_512.gif"
    }
    
    # ---------------- ADD LORA -----------------
    with tab1:
        files = get_files_in_a_directory(lora_file_dest, ['safetensors', 'ckpt'])

        # Iterate through each current LoRA in session state
        if len(files) == 0:
            st.error("No LoRAs found in the directory - go to Explore to download some, or drop them into ComfyUI/models/animatediff_motion_lora")                    
            if st.button("Check again", key="check_again"):
                st.rerun()
        else:
            # cleaning empty lora vals
            for idx, lora in enumerate(st.session_state[f"lora_data_{shot_uuid}"]):
                if not lora:
                    st.session_state[f"lora_data_{shot_uuid}"].pop(idx)
            
            for idx, lora in enumerate(st.session_state[f"lora_data_{shot_uuid}"]):
                if not lora:
                    continue
                h1, h2, h3, h4 = st.columns([1, 1, 1, 0.5])
                with h1:
                    file_idx = files.index(lora["filename"])
                    motion_lora = st.selectbox("Which LoRA would you like to use?", options=files, key=f"motion_lora_{idx}", index=file_idx)                                                    
                
                with h2:
                    strength_of_lora = st.slider("How strong would you like the LoRA to be?", min_value=0.0, max_value=1.0, value=lora["lora_strength"], step=0.01, key=f"strength_of_lora_{idx}")
                    lora_data.append({"filename": motion_lora, "lora_strength": strength_of_lora, "filepath": lora_file_dest + "/" + motion_lora})
                
                with h3:
                    when_to_apply_lora = st.slider("When to apply the LoRA?", min_value=0, max_value=100, value=(0,100), step=1, key=f"when_to_apply_lora_{idx}",disabled=True,help="This feature is not yet available.")
                
                with h4:
                    st.write("")
                    if st.button("Remove", key=f"remove_lora_{idx}"):
                        st.session_state[f"lora_data_{shot_uuid}"].pop(idx)
                        st.rerun()
                
                # displaying preview
                display_motion_lora(motion_lora, lora_file_links)
            
            if len(st.session_state[f"lora_data_{shot_uuid}"]) == 0:
                text = "Add a LoRA"
            else:
                text = "Add another LoRA"
            if st.button(text, key="add_motion_guidance"):
                if files and len(files):
                    st.session_state[f"lora_data_{shot_uuid}"].append({
                        "filename": files[0],
                        "lora_strength": 0.5,
                        "filepath": lora_file_dest + "/" + files[0]
                    })
                    st.rerun()
    
    # ---------------- DOWNLOAD LORA ---------------
    with tab2:
        text1, text2 = st.columns([1, 1])
        with text1:
            where_to_download_from = st.radio("Where would you like to get the LoRA from?", options=["Our list", "From a URL","Upload a LoRA"], key="where_to_download_from", horizontal=True)

        if where_to_download_from == "Our list":
            with text1: 
                selected_lora_optn = st.selectbox("Which LoRA would you like to download?", options=[a.split("/")[-1] for a in lora_file_links], key="selected_lora")
                # Display selected Lora
                display_motion_lora(selected_lora_optn, lora_file_links)
                
                if st.button("Download LoRA", key="download_lora"):
                    with st.spinner("Downloading LoRA..."):
                        save_directory = "ComfyUI/models/animatediff_motion_lora"
                        os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
                        
                        # Extract the filename from the URL
                        selected_lora, lora_idx = next(((ele, idx) for idx, ele in enumerate(lora_file_links.keys()) if selected_lora_optn in ele), None)
                        filename = selected_lora.split("/")[-1]
                        save_path = os.path.join(save_directory, filename)
                        
                        # Download the file
                        download_lora_bar = st.progress(0, text="")
                        response = requests.get(selected_lora, stream=True)
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
    
    # ---------------- TRAIN LORA --------------
    with tab3:
        b1, b2 = st.columns([1, 1])
        with b1:
            lora_name = st.text_input("Name this LoRA", key="lora_name")
            if model_files and len(model_files):
                base_sd_model = st.selectbox(
                        label="Select base sd model for training", 
                        options=model_files, 
                        key="base_sd_model_video", 
                        index=0
                    )
            else:
                base_sd_model = ""
                st.info("Default model Deliberate V2 would be selected")

            lora_prompt = st.text_area("Describe the motion", key="lora_prompt")
            training_video = st.file_uploader("Upload a video to train a new LoRA", type=["mp4"])

            if st.button("Train LoRA", key="train_lora", use_container_width=True):
                filename = str(uuid.uuid4()) + ".mp4"
                hosted_url = save_or_host_file(training_video, "videos/temp/" + filename, "video/mp4")

                file_data = {
                    "name": filename,
                    "type": InternalFileType.VIDEO.value,
                    "project_id": shot.project.uuid,
                }
                
                if hosted_url:
                    file_data.update({'hosted_url': hosted_url})
                else:
                    file_data.update({'local_path': "videos/temp/" + filename})
                
                video_file = data_repo.create_file(**file_data)
                video_width, video_height = get_media_dimensions(video_file.location)
                unique_file_tag = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                train_motion_lora(
                    video_file,
                    lora_prompt,
                    lora_name + "_" + unique_file_tag,
                    video_width,
                    video_height,
                    base_sd_model
                )

    return lora_data


def select_sd_model_element(shot_uuid, default_model):
    st.markdown("##### Style model")
    tab1, tab2 = st.tabs(["Choose Model","Download Models"])
    
    checkpoints_dir = "ComfyUI/models/checkpoints"
    all_files = os.listdir(checkpoints_dir)
    if len(all_files) == 0:
        model_files = [default_model]

    else:
        model_files = [file for file in all_files if file.endswith('.safetensors') or file.endswith('.ckpt')]
        model_files = [file for file in model_files if "xl" not in file]

    sd_model_dict = {
        "Anything V3 FP16 Pruned": {
            "url": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/anything-v3-fp16-pruned.safetensors.tar",
            "filename": "anything-v3-fp16-pruned.safetensors.tar"
        },
        "Deliberate V2": {
            "url": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/Deliberate_v2.safetensors.tar",
            "filename": "Deliberate_v2.safetensors.tar"
        },
        "Dreamshaper 8": {
            "url": "https://weights.replicate.delivery/default/comfy-ui/checkpoints/dreamshaper_8.safetensors.tar",
            "filename": "dreamshaper_8.safetensors.tar"
        },
        "epicrealism_pureEvolutionV5": {
            "url": "https://civitai.com/api/download/models/134065", 
            "filename": "epicrealism_pureEvolutionv5.safetensors"
        },
        "majicmixRealistic_v6": {
            "url": "https://civitai.com/api/download/models/94640", 
            "filename": "majicmixRealistic_v6.safetensors"
        },
    }

    cur_model = st.session_state[f'ckpt_{shot_uuid}']
    current_model_index = model_files.index(cur_model) if (cur_model and cur_model in model_files) else 0

    # ---------------- SELECT CKPT --------------
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            sd_model = ""
            def update_model():
                global sd_model
                sd_model = checkpoints_dir + "/" + st.session_state['sd_model_video']
                
            if model_files and len(model_files):
                sd_model = st.selectbox(
                    label="Which model would you like to use?", 
                    options=model_files, 
                    key="sd_model_video", 
                    index=current_model_index,
                    on_change=update_model
                )
            else:
                st.write("")
                st.info("Default model Deliberate V2 would be selected")

        with col2:
            if len(all_files) == 0:
                st.write("")
                st.info("This is the default model - to download more, go to the Download Models tab.")
            else:
                st.write("")
                st.info("To download more models, go to the Download Models tab.")

    
    # ---------------- ADD CKPT ---------------
    with tab2:
        where_to_get_model = st.radio("Where would you like to get the model from?", options=["Our list", "Upload a model", "From a URL"], key="where_to_get_model")

        if where_to_get_model == "Our list":
            model_name_selected = st.selectbox("Which model would you like to download?", options=list(sd_model_dict.keys()), key="model_to_download")
            
            if st.button("Download Model", key="download_model"):
                with st.spinner("Downloading model..."):
                    download_bar = st.progress(0, text="")
                    save_directory = "ComfyUI/models/checkpoints"
                    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
                    
                    # Retrieve the URL using the selected model name
                    model_url = sd_model_dict[model_name_selected]["url"]
                    
                    # Download the model and save it to the directory
                    response = requests.get(model_url, stream=True)
                    zip_filename = sd_model_dict[model_name_selected]["filename"]
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
                        
                        os.remove(filepath)
                    st.rerun()

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
                        
    return sd_model, model_files, 
          

def individual_frame_settings_element(shot_uuid, img_list, display_indent):
    with display_indent:
        st.markdown("##### Individual frame settings")
                    
    items_per_row = 3
    strength_of_frames = []
    distances_to_next_frames = []
    speeds_of_transitions = []
    freedoms_between_frames = []
    individual_prompts = []
    individual_negative_prompts = []
    motions_during_frames = []

    if len(img_list) <= 1:
        st.warning("You need at least two frames to generate a video.")
        st.stop()

    open_advanced_settings = st_memory.toggle("Open all advanced settings", key="advanced_settings", value=False)

    # setting default values to main shot settings
    if f'lora_data_{shot_uuid}' not in st.session_state:
        st.session_state[f'lora_data_{shot_uuid}'] = []

    if f'strength_of_adherence_value_{shot_uuid}' not in st.session_state:
        st.session_state[f'strength_of_adherence_value_{shot_uuid}'] = 0.10

    if f'type_of_motion_context_index_{shot_uuid}' not in st.session_state:
        st.session_state[f'type_of_motion_context_index_{shot_uuid}'] = 1

    if f'positive_prompt_video_{shot_uuid}' not in st.session_state:
        st.session_state[f"positive_prompt_video_{shot_uuid}"] = ""

    if f'negative_prompt_video_{shot_uuid}' not in st.session_state:
        st.session_state[f"negative_prompt_video_{shot_uuid}"] = ""

    if f'ckpt_{shot_uuid}' not in st.session_state:
        st.session_state[f'ckpt_{shot_uuid}'] = ""
        
    if f"amount_of_motion_{shot_uuid}" not in st.session_state:
        st.session_state[f"amount_of_motion_{shot_uuid}"] = 1.3
        
    # loading settings of the last shot (if this shot is being loaded for the first time)
    if f'strength_of_frame_{shot_uuid}_0' not in st.session_state:
        load_shot_settings(shot_uuid)
            
    # ------------- Timing Frame and their settings -------------------
    for i in range(0, len(img_list) , items_per_row):
        with st.container():
            grid = st.columns([2 if j%2==0 else 1 for j in range(2*items_per_row)])  # Adjust the column widths
            for j in range(items_per_row):
                idx = i + j
                if idx < len(img_list):                        
                    with grid[2*j]:  # Adjust the index for image column
                        img = img_list[idx]
                        if img.location:
                            st.info(f"**Frame {idx + 1}**")
                            st.image(img.location, use_column_width=True)
                                                                                                                        
                            # settings control
                            with st.expander("Advanced settings:", expanded=open_advanced_settings):
                                # checking for newly added frames
                                if f'individual_prompt_{shot_uuid}_{idx}' not in st.session_state:
                                    for k, v in DEFAULT_SHOT_MOTION_VALUES.items():
                                        st.session_state[f"{k}_{shot_uuid}_{idx}"] = v
                                
                                individual_prompt = st.text_input("What to include:", key=f"individual_prompt_widget_{idx}_{img.uuid}", value=st.session_state[f'individual_prompt_{shot_uuid}_{idx}'], help="Use this sparingly, as it can have a large impact on the video and cause weird distortions.")
                                individual_prompts.append(individual_prompt)
                                individual_negative_prompt = st.text_input("What to avoid:", key=f"negative_prompt_widget_{idx}_{img.uuid}", value=st.session_state[f'individual_negative_prompt_{shot_uuid}_{idx}'],help="Use this sparingly, as it can have a large impact on the video and cause weird distortions.")
                                individual_negative_prompts.append(individual_negative_prompt)
                                strength1, strength2 = st.columns([1, 1])
                                with strength1:
                                    strength_of_frame = st.slider("Strength of current frame:", min_value=0.25, max_value=1.0, step=0.01, key=f"strength_of_frame_widget_{shot_uuid}_{idx}", value=st.session_state[f'strength_of_frame_{shot_uuid}_{idx}'])
                                    strength_of_frames.append(strength_of_frame)         
                                with strength2:
                                    motion_during_frame = st.slider("Motion during frame:", min_value=0.5, max_value=1.5, step=0.01, key=f"motion_during_frame_widget_{idx}_{img.uuid}", value=st.session_state[f'motion_during_frame_{shot_uuid}_{idx}'])                            
                                    motions_during_frames.append(motion_during_frame)
                        else:                        
                            st.warning("No primary image present.")    

                    # distance, speed and freedom settings (also aggregates them into arrays)
                    with grid[2*j+1]:  # Add the new column after the image column
                        if idx < len(img_list) - 1:                                                                       
            
                            # if st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'] is a int, make it a float
                            if isinstance(st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'], int):
                                st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'] = float(st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'])                                    
                            distance_to_next_frame = st.slider("Seconds to next frame:", min_value=0.25, max_value=6.00, step=0.25, key=f"distance_to_next_frame_widget_{idx}_{img.uuid}", value=st.session_state[f'distance_to_next_frame_{shot_uuid}_{idx}'])
                            distances_to_next_frames.append(distance_to_next_frame/2)                                    
                            speed_of_transition = st.slider("Speed of transition:", min_value=0.45, max_value=0.7, step=0.01, key=f"speed_of_transition_widget_{idx}_{img.uuid}", value=st.session_state[f'speed_of_transition_{shot_uuid}_{idx}'])
                            speeds_of_transitions.append(speed_of_transition)                                      
                            freedom_between_frames = st.slider("Freedom between frames:", min_value=0.15, max_value=0.85, step=0.01, key=f"freedom_between_frames_widget_{idx}_{img.uuid}", value=st.session_state[f'freedom_between_frames_{shot_uuid}_{idx}'])
                            freedoms_between_frames.append(freedom_between_frames)
                                            
            if (i < len(img_list) - 1)  or (len(img_list) % items_per_row != 0):
                st.markdown("***")
                
    return strength_of_frames, distances_to_next_frames, speeds_of_transitions, freedoms_between_frames, individual_prompts, individual_negative_prompts, motions_during_frames
    