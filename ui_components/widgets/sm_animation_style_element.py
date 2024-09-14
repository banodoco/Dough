import time
import uuid
import os
import requests
import random
import string
from ui_components.widgets.model_selector_element import list_dir_files
import streamlit as st
from shared.constants import COMFY_BASE_PATH, InternalFileTag, InternalFileType
from ui_components.widgets.download_file_progress_bar import download_file_widget
from utils import st_memory
from ui_components.constants import DEFAULT_SHOT_MOTION_VALUES
from ui_components.methods.animation_style_methods import (
    calculate_weights,
    extract_influence_values,
    extract_strength_values,
    get_keyframe_positions,
    load_shot_settings,
    plot_weights,
)
from ui_components.methods.file_methods import (
    get_files_in_a_directory,
    get_media_dimensions,
    save_or_host_file,
)
from ui_components.widgets.display_element import display_motion_lora
from ui_components.methods.ml_methods import train_motion_lora
from utils.common_decorators import update_refresh_lock
from utils.constants import StabliseMotionOption
from utils.data_repo.data_repo import DataRepo
from utils.state_refresh import refresh_app
from streamlit.elements.utils import _shown_default_value_warning


_shown_default_value_warning = True

checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
SD_MODEL_DICT = {
    "realisticVisionV60B1_v51VAE.safetensors": {
        "url": "https://civitai.com/api/download/models/130072",
        "filename": "realisticVisionV60B1_v51VAE.safetensors",
        "dest": checkpoints_dir,
    },
    "anything_v50.safetensors": {
        "url": "https://civitai.com/api/download/models/30163",
        "filename": "anything_v50.safetensors",
        "dest": checkpoints_dir,
    },
    "dreamshaper_8.safetensors": {
        "url": "https://civitai.com/api/download/models/128713",
        "filename": "dreamshaper_8.safetensors",
        "dest": checkpoints_dir,
    },
    "epicrealism_pureEvolutionV5.safetensors": {
        "url": "https://civitai.com/api/download/models/134065",
        "filename": "epicrealism_pureEvolutionV5.safetensors",
        "dest": checkpoints_dir,
    },
    "majicmixRealistic_v6.safetensors": {
        "url": "https://civitai.com/api/download/models/94640",
        "filename": "majicmixRealistic_v6.safetensors",
        "dest": checkpoints_dir,
    },
}


def animation_sidebar(
    shot_uuid,
    img_list,
    type_of_frame_distribution,
    dynamic_frame_distribution_values,
    linear_frame_distribution_value,
    type_of_strength_distribution,
    dynamic_strength_values,
    linear_cn_strength_value,
    type_of_key_frame_influence,
    dynamic_key_frame_influence_values,
    linear_key_frame_influence_value,
    strength_of_frames,
    distances_to_next_frames,
    speeds_of_transitions,
    freedoms_between_frames,
    motions_during_frames,
    individual_prompts,
    individual_negative_prompts,
    default_model,
):
    with st.sidebar:
        with st.expander("⚙️ Visualisation of motion settings", expanded=True):
            if st_memory.toggle(
                "Open", key="open_motion_data", help="Closing this will speed up the interface.", value=True
            ):

                keyframe_positions = get_keyframe_positions(
                    type_of_frame_distribution,
                    dynamic_frame_distribution_values,
                    img_list,
                    linear_frame_distribution_value,
                )
                keyframe_positions = [int(kf * 16) for kf in keyframe_positions]
                last_key_frame_position = keyframe_positions[-1]
                strength_values = extract_strength_values(
                    type_of_strength_distribution,
                    dynamic_strength_values,
                    keyframe_positions,
                    linear_cn_strength_value,
                )
                key_frame_influence_values = extract_influence_values(
                    type_of_key_frame_influence,
                    dynamic_key_frame_influence_values,
                    keyframe_positions,
                    linear_key_frame_influence_value,
                )
                weights_list, frame_numbers_list = calculate_weights(
                    keyframe_positions,
                    strength_values,
                    4,
                    key_frame_influence_values,
                    last_key_frame_position,
                )
                plot_weights(weights_list, frame_numbers_list)


def video_shortlist_btn(video_uuid, type="add_to_shortlist"):
    data_repo = DataRepo()
    # add to shortlist
    if type == "add_to_shortlist":
        if st.button("Add to shortlist", key=f"{video_uuid}_shortlist_btn", use_container_width=True):
            data_repo.update_file(
                video_uuid,
                tag=InternalFileTag.SHORTLISTED_VIDEO.value,
            )
            refresh_app()
    # remove from shortlist btn
    else:
        if st.button(
            "Remove from shortlist",
            key=f"{video_uuid}_remove_shortlist_btn",
            use_container_width=True,
        ):
            data_repo.update_file(
                video_uuid,
                tag="",
            )
            refresh_app()


def video_motion_settings(shot_uuid, img_list):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    st.markdown("***")
    st.markdown("##### Overall style settings")

    """
    high_detail_mode_val = st.session_state.get(f"high_detail_mode_val_{shot_uuid}", True)
    high_detail_mode = st.checkbox(
        "Enable high detail mode",
        help="This improves the detail of the video, but doubles the amount of VRAM used per input frame.",
        key=f"high_detail_mode_{shot_uuid}",
        value=high_detail_mode_val,
    )
    """
    high_detail_mode = True
    e1, _, _ = st.columns([2, 1, 1])
    with e1:
        strength_of_adherence = st.slider(
            "Adherence to input frames:",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="strength_of_adherence_value",
            value=st.session_state[f"strength_of_adherence_value_{shot_uuid}"],
        )

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        overall_positive_prompt = ""

        def update_prompt():
            global overall_positive_prompt
            overall_positive_prompt = st.session_state[f"positive_prompt_video_{shot_uuid}"]

        overall_positive_prompt = st.text_area(
            "Overall prompt:",
            key="overall_positive_prompt",
            value=st.session_state[f"positive_prompt_video_{shot_uuid}"],
            on_change=update_prompt,
        )
    with f2:
        overall_negative_prompt = st.text_area(
            "Overall negative prompt:",
            key="overall_negative_prompt",
            value=st.session_state[f"negative_prompt_video_{shot_uuid}"],
        )

    styling_lora_data = select_styling_lora_element(shot_uuid)

    st.markdown("***")
    st.markdown("##### Overall motion settings")
    h1, h2, h3 = st.columns([1, 0.5, 1.0])
    with h1:
        # will fix this later

        if f"type_of_motion_context_index_{shot_uuid}" in st.session_state and isinstance(
            st.session_state[f"type_of_motion_context_index_{shot_uuid}"], str
        ):
            st.session_state[f"type_of_motion_context_index_{shot_uuid}"] = ["Low", "Standard", "High"].index(
                st.session_state[f"type_of_motion_context_index_{shot_uuid}"]
            )
        type_of_motion_context = st.radio(
            "Type of motion context:",
            options=["Low", "Standard", "High"],
            key="type_of_motion_context",
            horizontal=True,
            index=st.session_state[f"type_of_motion_context_index_{shot.uuid}"],
            help="This is how much the motion will be informed by the previous and next frames. 'High' can make it smoother but increase artifacts - while 'Low' make the motion less smooth but removes artifacts. Naturally, we recommend Standard.",
        )

        stabilise_motion_options = StabliseMotionOption.value_list()
        stabilise_index = 2
        if f"stabilise_motion_{shot_uuid}" in st.session_state and isinstance(
            st.session_state[f"stabilise_motion_{shot_uuid}"], str
        ):
            stabilise_index = stabilise_motion_options.index(
                st.session_state[f"stabilise_motion_{shot_uuid}"]
            )

        loop1, loop2 = st.columns([1, 1])
        with loop1:
            allow_for_looping = st_memory.checkbox("Allow for looping", key="allow_for_looping", value=False)
        with loop2:
            if allow_for_looping:
                st.info("To get a perfect loop, you should add the first image as the last.")

        stabilise_motion = st.radio(
            label="Amount to constrain motion:",
            options=stabilise_motion_options,
            index=stabilise_index,
            horizontal=True,
            label_visibility="visible",
            key="stabilise_motion",
        )

    if f"structure_control_image_{shot_uuid}" not in st.session_state:
        st.session_state[f"structure_control_image_{shot_uuid}"] = None

    if f"strength_of_structure_control_image_{shot_uuid}" not in st.session_state:
        st.session_state[f"strength_of_structure_control_image_{shot_uuid}"] = None

    return (
        strength_of_adherence,
        overall_positive_prompt,
        overall_negative_prompt,
        type_of_motion_context,
        allow_for_looping,
        high_detail_mode,
        stabilise_motion,
        styling_lora_data,
    )


def select_styling_lora_element(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    st.caption("Styling LoRAs")
    tab1, tab2 = st.tabs(["Apply LoRAs", "Download LoRAs"])

    lora_data = []
    lora_file_dest = os.path.join(COMFY_BASE_PATH, "models", "loras")

    if f"normal_lora_data_{shot_uuid}" not in st.session_state:
        st.session_state[f"normal_lora_data_{shot_uuid}"] = []

    # ---------------- APPLY LORA -----------------
    with tab1:
        files = get_files_in_a_directory(lora_file_dest, ["safetensors", "ckpt"])

        if len(files) == 0:

            lora1, _ = st.columns([1.5, 1])
            with lora1:
                st.info(
                    "No LoRAs found in the directory - go to Download LoRAs to add some, or drop them into: ComfyUI/models/loras"
                )
                if st.button("Check again", key="check_again_lora"):
                    refresh_app()
        else:
            # Initialize normal_lora_data in session state if it doesn't exist

            # Display existing LoRAs
            for idx, lora in enumerate(st.session_state[f"normal_lora_data_{shot_uuid}"]):
                h1, h2, h3 = st.columns([1, 1, 0.5])
                with h1:
                    default_index = files.index(lora["filename"]) if lora["filename"] in files else 0
                    lora_file = st.selectbox(
                        "Select LoRA:",
                        options=files,
                        key=f"normal_lora_{idx}",
                        index=default_index,
                    )

                with h2:
                    strength = st.slider(
                        "Strength:",
                        min_value=0.0,
                        max_value=1.0,
                        value=lora["lora_strength"],
                        step=0.01,
                        key=f"strength_of_normal_lora_{idx}",
                    )

                with h3:
                    st.write("")
                    st.write("")

                    def remove_lora(idx):
                        st.session_state[f"normal_lora_data_{shot_uuid}"].pop(idx)

                    if st.button(
                        "Remove", key=f"remove_normal_lora_{idx}", on_click=remove_lora, args=(idx,)
                    ):
                        refresh_app()

                # Update lora data if changed
                if lora_file != lora["filename"] or strength != lora["lora_strength"]:
                    st.session_state[f"normal_lora_data_{shot_uuid}"][idx] = {
                        "filename": lora_file,
                        "lora_strength": strength,
                        "filepath": os.path.join(lora_file_dest, lora_file),
                    }
                    st.rerun()

            # Add new LoRA button
            def add_lora(files, lora_file_dest):
                if files:
                    st.session_state[f"normal_lora_data_{shot_uuid}"].append(
                        {
                            "filename": files[0],
                            "lora_strength": 0.5,
                            "filepath": os.path.join(lora_file_dest, files[0]),
                        }
                    )

            if len(st.session_state[f"normal_lora_data_{shot_uuid}"]) == 0:
                text = "Add a LoRA"
            else:
                text = "Add another LoRA"

            if st.button(text, key="add_normal_lora", on_click=add_lora, args=(files, lora_file_dest)):
                refresh_app()

    # ---------------- DOWNLOAD LORA ---------------
    with tab2:
        lora_url = st.text_input("Enter the URL of the LoRA to download:", key="normal_lora_download_url")
        if st.button("Download LoRA", key="download_normal_lora"):
            if lora_url:
                with st.spinner("Downloading LoRA..."):
                    try:
                        response = requests.get(lora_url)
                        if response.status_code == 200:
                            content_disposition = response.headers.get("Content-Disposition")
                            if content_disposition:
                                filename = content_disposition.split("filename=")[1].strip('"')
                            else:
                                filename = lora_url.split("/")[-1]

                            # Remove query parameters from filename
                            filename = filename.split("?")[0]

                            save_path = os.path.join(lora_file_dest, filename)
                            with open(save_path, "wb") as f:
                                f.write(response.content)
                            st.success(f"Downloaded LoRA to {save_path}")
                            refresh_app()
                        else:
                            st.error("Failed to download LoRA. Please check the URL and try again.")
                    except Exception as e:
                        st.error(f"An error occurred while downloading the LoRA: {str(e)}")
            else:
                st.warning("Please enter a URL to download the LoRA.")

    # Prepare lora_data to return
    for lora in st.session_state[f"normal_lora_data_{shot_uuid}"]:
        lora_data.append(
            {
                "filename": lora["filename"],
                "lora_strength": lora["lora_strength"],
                "filepath": lora["filepath"],
            }
        )

    return lora_data


def select_motion_lora_element(shot_uuid, model_files):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    # if it's in local DEVELOPMENT ENVIRONMENT
    st.markdown("***")
    st.markdown("##### Motion guidance")
    tab1, tab2, tab3 = st.tabs(["Apply LoRAs", "Download LoRAs", "Train LoRAs"])

    lora_data = []
    lora_file_dest = os.path.join(COMFY_BASE_PATH, "models", "animatediff_motion_lora")
    lora_file_links = {
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/250_tony_stark_r64_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/250_train_r128_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/300_car_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_car_desert_48_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_car_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_jeep_driving_r32_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_man_running_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/500_rotation_temporal_unet.safetensors": "",
        "https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/750_jeep_driving_r32_temporal_unet.safetensors": "",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/300_zooming_in_temporal_unet.safetensors": "",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_cat_walking_temporal_unet.safetensors": "",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_playing_banjo_temporal_unet.safetensors": "",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_woman_dancing_temporal_unet.safetensors": "",
        "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/400_zooming_out_temporal_unet.safetensors": "",
    }

    # ---------------- ADD LORA -----------------
    with tab1:
        files = get_files_in_a_directory(lora_file_dest, ["safetensors", "ckpt"])
        # add WAS26.safetensors to the start of the list

        # Iterate through each current LoRA in session state
        if len(files) == 0:
            lora1, _ = st.columns([1.5, 1])
            with lora1:
                st.info(
                    "No LoRAs found in the directory - go to Download LoRAs to download some, or drop them into: ComfyUI/models/animatediff_motion_lora"
                )
                if st.button("Check again", key="check_again"):
                    refresh_app()
        else:
            # cleaning empty lora vals
            for idx, lora in enumerate(st.session_state[f"lora_data_{shot_uuid}"]):
                if not lora:
                    st.session_state[f"lora_data_{shot_uuid}"].pop(idx)

            for idx, lora in enumerate(st.session_state[f"lora_data_{shot_uuid}"]):
                if not lora:
                    continue
                h1, h2, h3, h4, h5, h6, h7 = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 0.5])
                with h1:
                    # file_idx = files.index(lora["filename"])
                    default_index = files.index(lora["filename"]) if lora["filename"] in files else 0
                    motion_lora = st.selectbox(
                        "Which LoRA would you like to use?",
                        options=files,
                        key=f"motion_lora_{idx}",
                        index=default_index,
                    )

                with h2:
                    display_motion_lora(motion_lora, lora_file_links)

                with h3:
                    strength_of_lora = st.slider(
                        "Strength:",
                        min_value=0.0,
                        max_value=1.0,
                        value=lora["lora_strength"],
                        step=0.01,
                        key=f"strength_of_lora_{idx}",
                    )

                    lora_data.append(
                        {
                            "filename": motion_lora,
                            "lora_strength": strength_of_lora,
                            "filepath": lora_file_dest + "/" + motion_lora,
                        }
                    )

                if strength_of_lora != lora["lora_strength"] or motion_lora != lora["filename"]:
                    st.session_state[f"lora_data_{shot_uuid}"][idx] = {
                        "filename": motion_lora,
                        "lora_strength": strength_of_lora,
                        "filepath": lora_file_dest + "/" + files[0],
                    }
                    st.rerun

                with h5:
                    lora_range = st.slider(
                        "When to apply:",
                        min_value=0,
                        max_value=100,
                        value=(0, 100),
                        step=1,
                        key=f"lora_range_{idx}",
                        disabled=True,
                        help="This feature is not yet available.",
                    )

                with h7:
                    st.write("")

                    def remove_lora(idx):
                        st.session_state[f"lora_data_{shot_uuid}"].pop(idx)

                    if st.button("Remove", key=f"remove_lora_{idx}", on_click=remove_lora, args=(idx,)):
                        refresh_app()

            if len(st.session_state[f"lora_data_{shot_uuid}"]) == 0:
                text = "Add a LoRA"
            else:
                text = "Add another LoRA"

            def add_lora(files, lora_file_dest):
                if files and len(files):
                    st.session_state[f"lora_data_{shot_uuid}"].append(
                        {
                            "filename": files[0],
                            "lora_strength": 0.5,
                            "filepath": lora_file_dest + "/" + files[0],
                        }
                    )

            if st.button(text, key="add_motion_guidance", on_click=add_lora, args=(files, lora_file_dest)):
                refresh_app()

    # ---------------- DOWNLOAD LORA ---------------
    with tab2:
        text1, text2 = st.columns([1, 1])
        with text1:
            where_to_download_from = st.radio(
                "Where would you like to get the LoRA from?",
                options=["Our list", "From a URL", "Upload a LoRA"],
                key="where_to_download_from",
                horizontal=True,
            )

        if where_to_download_from == "Our list":
            with text1:
                selected_lora_optn = st.selectbox(
                    "Which LoRA would you like to download?",
                    options=[a.split("/")[-1] for a in lora_file_links],
                    key="selected_lora",
                )
                # Display selected Lora
                display_motion_lora(selected_lora_optn, lora_file_links)

                if st.button("Download LoRA", key="download_lora"):
                    save_directory = os.path.join(COMFY_BASE_PATH, "models", "animatediff_motion_lora")
                    selected_lora, lora_idx = next(
                        (
                            (ele, idx)
                            for idx, ele in enumerate(lora_file_links.keys())
                            if selected_lora_optn in ele
                        ),
                        None,
                    )
                    filename = selected_lora.split("/")[-1]
                    download_file_widget(
                        selected_lora,
                        filename,
                        save_directory,
                    )

        elif where_to_download_from == "From a URL":
            with text1:
                text_input = st.text_input("Enter the URL of the LoRA", key="text_input_lora")
            with text2:
                st.write("")
                st.write("")
                st.write("")
                st.info(
                    "Make sure to get the download url of the LoRA. \n\n For example, from Hugging Face, it should look like this: https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors"
                )
            with text1:
                if st.button("Download LoRA", key="download_lora"):
                    with st.spinner("Downloading LoRA..."):
                        save_directory = os.path.join(COMFY_BASE_PATH, "models", "animatediff_motion_lora")
                        os.makedirs(save_directory, exist_ok=True)
                        response = requests.get(text_input)
                        if response.status_code == 200:
                            with open(os.path.join(save_directory, text_input.split("/")[-1]), "wb") as f:
                                f.write(response.content)
                            st.success(f"Downloaded LoRA to {save_directory}")
                        else:
                            st.error("Failed to download LoRA")
        elif where_to_download_from == "Upload a LoRA":
            st.info(
                "It's simpler to just drop this into the ComfyUI/models/animatediff_motion_lora directory."
            )

    # ---------------- TRAIN LORA --------------
    with tab3:
        b1, b2, b3 = st.columns([1, 1, 0.5])
        with b1:
            lora_name = st.text_input("Name this LoRA", key="lora_name")
            if model_files and len(model_files):
                base_sd_model = st.selectbox(
                    label="Select base:", options=model_files, key="base_sd_model_video", index=0
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
                    file_data.update({"hosted_url": hosted_url})
                else:
                    file_data.update({"local_path": "videos/temp/" + filename})

                video_file = data_repo.create_file(**file_data)
                video_width, video_height = get_media_dimensions(video_file.location)
                unique_file_tag = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
                train_motion_lora(
                    video_file,
                    lora_prompt,
                    lora_name + "_" + unique_file_tag,
                    video_width,
                    video_height,
                    base_sd_model,
                )
    with b2:
        st.info("This takes around 30 minutes to train.")
    return lora_data


def select_sd_model_element(shot_uuid, default_model):
    st.markdown("##### Style model")
    tab1, tab2 = st.tabs(["Choose Model", "Download Models"])

    checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
    all_files = list_dir_files(checkpoints_dir, depth=1)
    if len(all_files) == 0:
        model_files = [default_model]

    else:
        model_files = [file for file in all_files if file.endswith(".safetensors") or file.endswith(".ckpt")]
        ignored_model_list = ["dynamicrafter_512_interp_v1.ckpt"]
        model_files = [file for file in model_files if file not in ignored_model_list]

        # model_files += [v["filename"] for v in SD_MODEL_DICT.values()]
        model_files = list(set(model_files))

    # setting default
    if "dreamshaper_8.safetensors" in model_files:
        model_files.remove("dreamshaper_8.safetensors")
        model_files.insert(0, "dreamshaper_8.safetensors")

    cur_model = st.session_state[f"ckpt_{shot_uuid}"]
    current_model_index = model_files.index(cur_model) if (cur_model and cur_model in model_files) else 0

    # ---------------- SELECT CKPT --------------
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            sd_model = ""

            def update_model():
                global sd_model
                sd_model = checkpoints_dir + "/" + st.session_state["sd_model_video"]

            if model_files and len(model_files):
                sd_model = st.selectbox(
                    label="Styling model:",
                    options=model_files,
                    key="sd_model_video",
                    index=current_model_index,
                    on_change=update_model,
                )
            else:
                st.write("")
                st.info("Default model Dreamshaper 8 would be selected")
                sd_model = default_model

        with col2:
            if len(all_files) == 0:
                st.write("")
                st.info("These models will be auto-downloaded during the generation")
            else:
                st.write("")
                st.info("Please only select SD1.5-based models.")

    # ---------------- ADD CKPT ---------------
    with tab2:
        where_to_get_model = st.radio(
            "Where would you like to get the model from?",
            options=["Our list", "Upload a model", "From a URL"],
            key="where_to_get_model",
        )

        if where_to_get_model == "Our list":
            model_name_selected = st.selectbox(
                "Which model would you like to download?",
                options=list(SD_MODEL_DICT.keys()),
                key="model_to_download",
            )

            # from streamlit_globalrefresh import st_globalrefresh

            if st.button("Download Model", key="download_model"):
                download_file_widget(
                    SD_MODEL_DICT[model_name_selected]["url"],
                    SD_MODEL_DICT[model_name_selected]["filename"],
                    checkpoints_dir,
                )

        elif where_to_get_model == "Upload a model":
            st.info("It's simpler to just drop this into the ComfyUI/models/checkpoints directory.")

        elif where_to_get_model == "From a URL":
            text1, text2 = st.columns([1, 1])
            with text1:
                text_input = st.text_input("Enter the URL of the model", key="text_input")
            with text2:
                st.info(
                    "Make sure to get the download url of the model. \n\n For example, from Civit, this should look like this: https://civitai.com/api/download/models/179446. \n\n While from Hugging Face, it should look like this: https://huggingface.co/Kijai/animatediff_motion_director_loras/resolve/main/1000_jeep_driving_r32_temporal_unet.safetensors"
                )

            if st.button("Download Model", key="download_model_url"):
                with st.spinner("Downloading model..."):
                    save_directory = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
                    os.makedirs(save_directory, exist_ok=True)
                    response = requests.get(text_input)
                    if response.status_code == 200:
                        with open(os.path.join(save_directory, text_input.split("/")[-1]), "wb") as f:
                            f.write(response.content)
                        st.success(f"Downloaded model to {save_directory}")
                    else:
                        st.error("Failed to download model")

    return (
        sd_model,
        model_files,
    )


def individual_frame_settings_element(shot_uuid, img_list):
    st.write("")
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

    # setting default values to main shot settings
    if f"lora_data_{shot_uuid}" not in st.session_state:
        st.session_state[f"lora_data_{shot_uuid}"] = []

    if f"strength_of_adherence_value_{shot_uuid}" not in st.session_state:
        st.session_state[f"strength_of_adherence_value_{shot_uuid}"] = 0.2

    if f"type_of_motion_context_index_{shot_uuid}" not in st.session_state:
        st.session_state[f"type_of_motion_context_index_{shot_uuid}"] = 1

    if f"positive_prompt_video_{shot_uuid}" not in st.session_state:
        st.session_state[f"positive_prompt_video_{shot_uuid}"] = ""

    if f"negative_prompt_video_{shot_uuid}" not in st.session_state:
        st.session_state[f"negative_prompt_video_{shot_uuid}"] = ""

    if f"ckpt_{shot_uuid}" not in st.session_state:
        st.session_state[f"ckpt_{shot_uuid}"] = ""

    if f"amount_of_motion_{shot_uuid}" not in st.session_state:
        st.session_state[f"amount_of_motion_{shot_uuid}"] = 1.25

    # loading settings of the last shot (if this shot is being loaded for the first time)
    if f"strength_of_frame_{shot_uuid}_{img_list[0].uuid}" not in st.session_state:
        load_shot_settings(shot_uuid)

    # ------------- Timing Frame and their settings -------------------

    def bulk_updater(label, key_suffix, img_uuid, uuid, img_list):
        key = f"{key_suffix}_{uuid}_{img_uuid}"
        if "last_frame_changed" in st.session_state and st.session_state["last_frame_changed"] == key:
            with st.expander(f"Set all '{label}' to {st.session_state['last_value_set']}", expanded=True):
                if st.session_state[f"{uuid}_preview_mode"]:
                    start_frame, end_frame = st.session_state[f"frames_to_preview_{uuid}"]
                    range_to_edit = (start_frame, end_frame)
                else:
                    range_to_edit = (1, len(img_list))

                if st.button(
                    f"Update",
                    key=f"button_{img_uuid}",
                    use_container_width=True,
                    type="primary",
                    help=f"This will set all the '{label}' within the current view.",
                ):
                    queue_updates(key_suffix, st.session_state[key], img_uuid, uuid, range_to_edit)
                    st.session_state["last_frame_changed"] = None
                    st.session_state["last_value_set"] = None
                    refresh_app()

    def apply_updates(key_suffix, value, uuid, range_to_edit):
        start_frame, end_frame = range_to_edit
        for idx, img in enumerate(img_list[start_frame - 1 : end_frame], start=start_frame):
            key = f"{key_suffix}_{uuid}_{img.uuid}"
            if key in st.session_state:
                st.session_state[key] = value

    def update_last_changed(key, value):
        st.session_state["last_frame_changed"] = key
        st.session_state["last_value_set"] = value

    def queue_updates(key_suffix, value, idx, uuid, range_to_edit):
        st.session_state["update_values"] = (key_suffix, value, uuid, range_to_edit)

    if "update_values" in st.session_state:
        key_suffix, value, uuid, range_to_edit = st.session_state["update_values"]
        apply_updates(key_suffix, value, uuid, range_to_edit)
        del st.session_state["update_values"]  # Clear the update instruction after applying

    h1, h2, h3 = st.columns([1, 2, 1])
    with h1:
        type_of_settings = st_memory.radio(
            "Type of settings:",
            options=["Simple", "Advanced"],
            key=f"{shot_uuid}_mode_type",
            horizontal=True,
        )

    with h3:

        type_of_selector = st_memory.radio(
            "Type of selector:",
            options=["Slider", "Number select"],
            key=f"{shot_uuid}_preview_mode_type",
            horizontal=True,
        )

        if type_of_selector == "Slider":
            st.session_state[f"type_of_selector"] = "slider"
        else:
            st.session_state[f"type_of_selector"] = "number_input"

    # take a range of frames from the user

    # st.markdown(
    #     '<a href="#generate-animations" target="_self">View all</a>',
    #     unsafe_allow_html=True,
    # )
    # st.markdown('<div id="generate-animation"></div>', unsafe_allow_html=True)

    if "jump_to_animation" not in st.session_state:
        st.session_state.jump_to_animation = False
    if st.session_state.jump_to_animation:
        st.markdown('<div id="generate-animation"></div>', unsafe_allow_html=True)
        # Reset the jump flag
        st.session_state.jump_to_animation = False

        # Use JavaScript to scroll to the anchor
        st.components.v1.html(
            """
            <script>
                element = window.parent.document.getElementById('generate-animation');
                element.scrollIntoView();
            </script>
            """,
            height=0,
        )

    current_preview_range = st.session_state.get(f"frames_to_preview_{shot_uuid}", (1, len(img_list)))
    # Check if the current preview range is not from the beginning to the end
    preview1, preview2, preview3 = st.columns([1, 6, 1])
    with preview2:
        frames_to_preview = st_memory.slider(
            "Frames to preview:",
            min_value=1,
            max_value=len(img_list),
            value=current_preview_range,
            key=f"frames_to_preview_{shot_uuid}",
        )

    if frames_to_preview != (1, len(img_list)):
        st.session_state[f"{shot_uuid}_preview_mode"] = True
    else:
        st.session_state[f"{shot_uuid}_preview_mode"] = False

    total_number_of_frames = len(img_list)

    def start_preview(shot_uuid, idx, total_number_of_frames):
        current_preview = st.session_state.get(f"frames_to_preview_{shot_uuid}", (1, len(img_list)))
        if idx > 2 or idx > current_preview[0] + 3:
            st.session_state.jump_to_animation = True
        st.session_state[f"{shot_uuid}_preview_mode"] = True
        if idx == 0:
            st.session_state[f"frames_to_preview_{shot_uuid}"] = (
                idx + 1,
                len(img_list) - 1,
            )
        else:
            st.session_state[f"frames_to_preview_{shot_uuid}"] = (
                idx + 1,
                len(img_list),
            )

    def close_preview_mode(shot_uuid, total_number_of_frames):
        st.session_state[f"frames_to_preview_{shot_uuid}"] = (1, total_number_of_frames)
        st.session_state[f"{shot_uuid}_preview_mode"] = False

    def open_preview_mode(shot_uuid):
        st.session_state[f"frames_to_preview_{shot_uuid}"] = (1, 3)
        st.session_state[f"{shot_uuid}_preview_mode"] = True

    def shift_preview_window(shot_uuid, total_number_of_frames):
        frames_to_preview = st.session_state[f"frames_to_preview_{shot_uuid}"]
        new_start = min(frames_to_preview[0] + 2, total_number_of_frames - 2)
        new_end = min(frames_to_preview[1] + 2, total_number_of_frames)
        st.session_state[f"frames_to_preview_{shot_uuid}"] = (new_start, new_end)

    def extend_preview_window(shot_uuid, total_number_of_frames, extend_by=3):
        frames_to_preview = st.session_state[f"frames_to_preview_{shot_uuid}"]
        new_end = min(frames_to_preview[1] + extend_by, total_number_of_frames)
        st.session_state[f"frames_to_preview_{shot_uuid}"] = (
            frames_to_preview[0],
            new_end,
        )

    if st.session_state.get(f"{shot_uuid}_preview_mode", False):

        with preview1:
            st.write("")
            if st.button(
                "Close preview mode",
                key=f"close_preview_mode_{shot_uuid}",
                on_click=close_preview_mode,
                args=(shot_uuid, total_number_of_frames),
            ):
                refresh_app()

        with preview3:
            st.write("")
            if st.button(
                "Shift forward",
                key=f"shift_forward_{shot_uuid}",
                on_click=shift_preview_window,
                args=(shot_uuid, total_number_of_frames),
            ):
                refresh_app()
    else:
        with preview1:
            st.write("")
            if st.button(
                "Open preview mode",
                key=f"open_preview_mode_{shot_uuid}",
                on_click=open_preview_mode,
                args=(shot_uuid,),
            ):
                st.write("Opening preview mode")
                refresh_app()

        with preview3:
            st.write("")
            st.info("Showing all")

    start_frame, end_frame = frames_to_preview
    img_list = img_list[start_frame - 1 : end_frame]

    if len(img_list) <= 1:
        st.error("You need at least 2 frames to preview")

    cumulative_seconds = 0.0
    for i in range(0, len(img_list), items_per_row):
        prev_frame_settings = None
        with st.container():

            grid = st.columns([2 if j % 2 == 0 else 2 for j in range(2 * items_per_row)])

            for j in range(items_per_row):
                idx = i + j
                img = img_list[idx] if idx < len(img_list) else None

                if img and img.location:
                    with grid[2 * j]:
                        actual_frame_number = start_frame + idx
                        st.info(f"**Frame {actual_frame_number} - {cumulative_seconds:.2f}s**")
                        st.image(img.location, use_column_width=True)

                        preview_start, preview_end = st.session_state[f"frames_to_preview_{shot_uuid}"]
                        if actual_frame_number > preview_start and actual_frame_number < preview_end:

                            def end_preview(preview_start, shot_uuid, idx):
                                st.session_state[f"frames_to_preview_{shot_uuid}"] = (
                                    preview_start,
                                    idx,
                                )

                            if type_of_settings == "Advanced":
                                btn1, btn2 = st.columns([1, 1])
                                with btn1:
                                    if st.button(
                                        "▶️",
                                        key=f"start_here_{shot_uuid}_{img.uuid}",
                                        use_container_width=True,
                                        on_click=start_preview,
                                        args=(shot_uuid, actual_frame_number - 1, total_number_of_frames),
                                        help="This will start a preview of the animation from this frame",
                                    ):
                                        refresh_app()
                                with btn2:
                                    if st.button(
                                        "⏹️",
                                        key=f"end_preview_{shot_uuid}_{img.uuid}",
                                        use_container_width=True,
                                        on_click=end_preview,
                                        args=(preview_start, shot_uuid, actual_frame_number),
                                        help="This will end the preview of the animation at this frame",
                                    ):
                                        refresh_app()

                        elif (
                            actual_frame_number == preview_end
                            and actual_frame_number != total_number_of_frames
                        ):
                            btn1, btn2 = st.columns([1, 1])
                            with btn1:
                                if st.button(
                                    "+1",
                                    use_container_width=True,
                                    on_click=extend_preview_window,
                                    args=(shot_uuid, total_number_of_frames, 1),
                                    help="This will extend the preview window by 1 frame",
                                ):
                                    refresh_app()
                            with btn2:
                                if st.button(
                                    "+3",
                                    use_container_width=True,
                                    on_click=extend_preview_window,
                                    args=(shot_uuid, total_number_of_frames, 3),
                                    help="This will extend the preview window by 3 frames",
                                ):
                                    refresh_app()

                                # Check if we need to jump and create an anchor

                    # Create a new grid for each row of images
                    if j == 0:
                        cols2 = st.columns(3)  # Create a grid with 3 columns

                    with cols2[j]:
                        # setting default values for frames (if they are newly added or settings is not present in the session_state)
                        if f"distance_to_next_frame_{shot_uuid}_{img.uuid}" not in st.session_state:
                            # for newly created frames we apply prev frame settings if available
                            settings_to_apply = prev_frame_settings or DEFAULT_SHOT_MOTION_VALUES
                            cur_settings = {}
                            for k, v in settings_to_apply.items():
                                st.session_state[f"{k}_{shot_uuid}_{img.uuid}"] = v
                                cur_settings[f"{k}"] = v
                            prev_frame_settings = cur_settings
                        else:
                            cur_settings = {}
                            for k, v in DEFAULT_SHOT_MOTION_VALUES.items():
                                t_key = f"{k}_{shot_uuid}_{img.uuid}"
                                cur_settings[k] = (
                                    v if t_key not in st.session_state else st.session_state[t_key]
                                )
                            prev_frame_settings = cur_settings
                        if type_of_settings == "Advanced":
                            sub1, sub2 = st.columns([1, 1])
                            with sub1:
                                individual_prompt = st.text_input(
                                    "Frame prompt:",
                                    key=f"individual_prompt_widget_{img.uuid}",
                                    value=st.session_state[f"individual_prompt_{shot_uuid}_{img.uuid}"],
                                    help="This wll bias the video towards the words you enter for this segment. Lowering the strength will increase the impact of the prompt.",
                                )
                            with sub2:

                                individual_negative_prompt = st.text_input(
                                    "Frame negative prompt:",
                                    key=f"negative_prompt_widget_{img.uuid}",
                                    value=st.session_state[
                                        f"individual_negative_prompt_{shot_uuid}_{img.uuid}"
                                    ],
                                    help="This will bias the video away from the words you enter for this segment.",
                                )

                        else:
                            individual_prompt = ""
                            individual_negative_prompt = ""
                        individual_prompts.append(individual_prompt)
                        individual_negative_prompts.append(individual_negative_prompt)
                        advanced1, advanced2, _ = st.columns([1, 1, 0.5])
                        with advanced1:

                            def create_slider(
                                label,
                                min_value,
                                max_value,
                                step,
                                key_suffix,
                                default_value,
                                img_uuid,
                                uuid,
                                help_text=None,
                                img_list=img_list,
                                help=None,
                            ):
                                value_key = f"{key_suffix}_{uuid}_{img_uuid}"
                                widget_key = f"{key_suffix}_widget_{uuid}_{img_uuid}"

                                if value_key not in st.session_state:
                                    st.session_state[value_key] = default_value
                                if st.session_state[f"type_of_selector"] == "number_input":
                                    slider_value = st.number_input(
                                        label,
                                        min_value=min_value,
                                        max_value=max_value,
                                        step=step,
                                        key=widget_key,
                                        value=st.session_state[value_key],
                                        help=help,
                                    )
                                else:
                                    slider_value = st.slider(
                                        label,
                                        min_value=min_value,
                                        max_value=max_value,
                                        step=step,
                                        key=widget_key,
                                        value=st.session_state[value_key],
                                        help=help,
                                    )

                                if slider_value != st.session_state[value_key]:
                                    st.session_state[value_key] = slider_value
                                    update_last_changed(value_key, slider_value)
                                    refresh_app()

                                return slider_value

                            def update_last_changed(key, value):
                                st.session_state["last_frame_changed"] = key
                                st.session_state["last_value_set"] = value

                            if type_of_settings == "Simple":
                                strength_of_frame = 0.85
                            else:
                                strength_of_frame = create_slider(
                                    label="Strength of frame:",
                                    min_value=0.0,
                                    max_value=1.0,
                                    step=0.01,
                                    key_suffix="strength_of_frame",
                                    default_value=st.session_state[
                                        f"strength_of_frame_{shot_uuid}_{img.uuid}"
                                    ],
                                    img_uuid=img.uuid,
                                    uuid=shot_uuid,
                                    img_list=img_list,
                                    help="How strongly the frame will influence the animation at its peak.",
                                )

                            strength_of_frames.append(strength_of_frame)

                            bulk_updater(
                                "Strength of frames", "strength_of_frame", img.uuid, shot_uuid, img_list
                            )

                        with advanced2:
                            if type_of_settings == "Simple":
                                motion_during_frame = 1.25
                            else:
                                motion_during_frame = create_slider(
                                    label="Motion during frame:",
                                    min_value=0.5,
                                    max_value=1.5,
                                    step=0.01,
                                    key_suffix="motion_during_frame",
                                    default_value=st.session_state[
                                        f"motion_during_frame_{shot_uuid}_{img.uuid}"
                                    ],
                                    img_uuid=img.uuid,
                                    uuid=shot_uuid,
                                    img_list=img_list,
                                    help="How much the frame will move during this frame's influence.",
                                )

                            motions_during_frames.append(motion_during_frame)

                            bulk_updater(
                                "Motion during frames", "motion_during_frame", img.uuid, shot_uuid, img_list
                            )

                    with grid[2 * j + 1]:
                        if idx < len(img_list) - 1:
                            distance_to_next_frame = create_slider(
                                label="Seconds to next frame:",
                                min_value=0.25,
                                max_value=(
                                    12.00 if st.session_state[f"type_of_selector"] == "slider" else 600.00
                                ),
                                step=0.25,
                                key_suffix="distance_to_next_frame",
                                default_value=st.session_state[
                                    f"distance_to_next_frame_{shot_uuid}_{img.uuid}"
                                ],
                                img_uuid=img.uuid,
                                uuid=shot_uuid,
                                img_list=img_list,
                                help="How long until the peak of the next frame.To set the time to more than 12 seconds, you can use the number input.",
                            )

                            distances_to_next_frames.append(distance_to_next_frame)
                            bulk_updater(
                                "Distance to next frames",
                                "distance_to_next_frame",
                                img.uuid,
                                shot_uuid,
                                img_list,
                            )
                            if type_of_settings == "Simple":
                                speed_of_transition = 0.5
                            else:
                                speed_of_transition = create_slider(
                                    label="Speed of transition:",
                                    min_value=0.20,
                                    max_value=0.80,
                                    step=0.01,
                                    key_suffix="speed_of_transition",
                                    default_value=st.session_state[
                                        f"speed_of_transition_{shot_uuid}_{img.uuid}"
                                    ],
                                    img_uuid=img.uuid,
                                    uuid=shot_uuid,
                                    img_list=img_list,
                                    help="How fast the transition from this frame to the next. Lower values will cause the frames to blend together more.",
                                )

                            bulk_updater(
                                "Speed of transitions", "speed_of_transition", img.uuid, shot_uuid, img_list
                            )
                            speeds_of_transitions.append(speed_of_transition)

                            if type_of_settings == "Simple":
                                freedom_between_frames = 0.25
                            else:
                                freedom_between_frames = create_slider(
                                    label="Freedom between frames:",
                                    min_value=0.0,
                                    max_value=1.0,
                                    step=0.01,
                                    key_suffix="freedom_between_frames",
                                    default_value=st.session_state[
                                        f"freedom_between_frames_{shot_uuid}_{img.uuid}"
                                    ],
                                    img_uuid=img.uuid,
                                    uuid=shot_uuid,
                                    img_list=img_list,
                                    help="How much the animation can run free between frames. Lower values will constrain the animation to the frames.",
                                )

                            bulk_updater(
                                "Freedom between frames",
                                "freedom_between_frames",
                                img.uuid,
                                shot_uuid,
                                img_list,
                            )
                            freedoms_between_frames.append(freedom_between_frames)
                            cumulative_seconds += distance_to_next_frame

            if i + items_per_row >= len(img_list):

                if st.session_state.get(f"{shot_uuid}_preview_mode", False):
                    st.markdown("***")
                    btn1, btn2, btn3 = st.columns([1, 1, 3])
                    with btn1:

                        st.success(
                            f"Preview mode is on - showing frames {frames_to_preview[0]} to {frames_to_preview[1]} out of {total_number_of_frames}."
                        )
                        if st.button(
                            "Close preview mode",
                            key=f"close_preview_mode_2_{shot_uuid}",
                            on_click=close_preview_mode,
                            args=(shot_uuid, total_number_of_frames),
                            use_container_width=True,
                        ):
                            refresh_app()
                    with btn2:
                        if st.button(
                            "Shift preview window",
                            key=f"shift_preview_window_{shot_uuid}",
                            on_click=shift_preview_window,
                            args=(shot_uuid, total_number_of_frames),
                        ):
                            refresh_app()

                        if st.button(
                            "+3 preview window",
                            key=f"extend_preview_window_{shot_uuid}",
                            on_click=extend_preview_window,
                            args=(shot_uuid, total_number_of_frames),
                        ):
                            refresh_app()

            if (i < len(img_list) - 1) or (len(img_list) % items_per_row != 0):
                st.markdown("***")

    return (
        strength_of_frames,
        distances_to_next_frames,
        speeds_of_transitions,
        freedoms_between_frames,
        individual_prompts,
        individual_negative_prompts,
        motions_during_frames,
    )
