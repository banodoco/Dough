import os
import time
import streamlit as st

from shared.constants import COMFY_BASE_PATH, GPU_INFERENCE_ENABLED_KEY, ConfigManager
from ui_components.widgets.download_file_progress_bar import download_file_widget
from utils.constants import T2IModel
from utils.state_refresh import refresh_app


config_manager = ConfigManager()
gpu_enabled = config_manager.get(GPU_INFERENCE_ENABLED_KEY, False)
checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")

SDXL_MODEL_DOWNLOAD_LIST = {
    "SDXL-Base": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "filename": "sd_xl_base_1.0.safetensors",
        "desc": "Base SDXL model",
        "dest": checkpoints_dir,
    },
    "SDXL-Base-Lightning": {
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step.safetensors",
        "filename": "sdxl_lightning_4step.safetensors",
        "desc": "Faster version of SDXL",
        "dest": checkpoints_dir,
    },
    "Juggernaut-XL_v9": {
        "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "filename": "Juggernaut-XL_v9_v2.safetensors",
        "desc": "Good general purpose model",
        "dest": checkpoints_dir,
    },
    "Juggernaut-XL-Lightning": {
        "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/resolve/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors",
        "filename": "Juggernaut-XL_Lightning_4Steps.safetensors",
        "desc": "Faster version of Juggernaut-XL",
        "dest": checkpoints_dir,
    },
}

SD3_MODEL_DOWNLOAD_LIST = {
    "SD3 Medium fp8 clip + T5xxl": {
        "url": "https://huggingface.co/lone682/sd3/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors?download=true",
        "filename": "sd3_medium_incl_clips_t5xxlfp8.safetensors",
        "desc": "",
        "dest": checkpoints_dir,
    },
    "SD3 Medium fp16 clip + T5xxl": {
        "url": "https://huggingface.co/lone682/sd3/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors?download=true",
        "filename": "sd3_medium_incl_clips_t5xxlfp16.safetensors",
        "desc": "",
        "dest": checkpoints_dir,
    },
}


# TODO: make all the file access methods in a single interface
def list_dir_files(directory, depth=0):
    all_items = []

    def explore(current_dir, current_depth):
        if current_depth > depth:
            return

        for entry in os.listdir(current_dir):
            full_path = os.path.join(current_dir, entry)
            rel_path = os.path.relpath(full_path, directory)

            if os.path.isfile(full_path):
                all_items.append(rel_path)
            elif os.path.isdir(full_path):
                all_items.append(rel_path + "/")
                if current_depth < depth:
                    explore(full_path, current_depth + 1)

    explore(directory, 0)
    return sorted(all_items)


def model_selector_online_element(type=T2IModel.SDXL.value, position="explorer", selected_model=None):
    default_model = (
        "sd_xl_base_1.0.safetensors"
        if type == T2IModel.SDXL.value
        else "sd3_medium_incl_clips_t5xxlfp16.safetensors"
    )

    info_msg = (
        "SDXL base will be selected as default. This tends to work best with style transfer."
        if type == T2IModel.SDXL.value
        else "Default base SD3 medium will be selected."
    )

    col1, col2 = st.columns([1, 0.25])
    with col1:
        explorer_gen_model = ""

        default_model_list = (
            [v["filename"] for v in SDXL_MODEL_DOWNLOAD_LIST.values()]
            if type == T2IModel.SDXL.value
            else [v["filename"] for v in SD3_MODEL_DOWNLOAD_LIST.values()]
        )
        model_files = [
            file for file in default_model_list if file.endswith(".safetensors") or file.endswith(".ckpt")
        ]

        cur_model = selected_model if (selected_model and selected_model in model_files) else None
        current_model_index = model_files.index(cur_model) if (cur_model and cur_model in model_files) else 0

        if model_files and len(model_files):
            explorer_gen_model = st.selectbox(
                label="Styling model:",
                options=model_files,
                key=f"{position}_{type}_explorer_gen_model_video",
                index=current_model_index,
                # on_change=update_model,
            )

            st.session_state["insp_lightning_mode"] = "lightning" in explorer_gen_model

        else:
            st.write("")
            st.info(info_msg)
            explorer_gen_model = default_model

        return explorer_gen_model


def model_selector_offline_element(type=T2IModel.SDXL.value, position="explorer", selected_model=None):
    tab1, tab2 = st.tabs(["Choose Model", "Download Models"])
    default_model = (
        "sd_xl_base_1.0.safetensors"
        if type == T2IModel.SDXL.value
        else "sd3_medium_incl_clips_t5xxlfp16.safetensors"
    )

    info_msg = (
        "SDXL base will be selected as default. This tends to work best with style transfer."
        if type == T2IModel.SDXL.value
        else "Default base SD3 medium will be selected."
    )

    with tab1:
        col1, col2 = st.columns([1, 0.25])
        with col1:
            explorer_gen_model = ""

            # TODO: make a common interface for accessing different types of files
            # read the metadata of the models and sort them into sd/sd3/sdxl categories during startup
            all_files = list_dir_files(checkpoints_dir, 1)
            default_model_list = (
                [v["filename"] for v in SDXL_MODEL_DOWNLOAD_LIST.values()]
                if type == T2IModel.SDXL.value
                else [v["filename"] for v in SD3_MODEL_DOWNLOAD_LIST.values()]
            )
            all_files += default_model_list
            all_files = list(set(all_files))
            ignored_model_list = [
                "dynamicrafter_512_interp_v1.ckpt",
                "sd_xl_refiner_1.0.safetensors",
                "sd_xl_refiner_1.0_0.9vae.safetensors",
            ]
            model_files = [
                file for file in all_files if file.endswith(".safetensors") or file.endswith(".ckpt")
            ]

            if type == T2IModel.SDXL.value:
                match_condition = lambda file: file  # and "xl" in file.lower()
            else:
                match_condition = lambda file: file  # and "sd3" in file.lower()

            model_files = [
                file for file in model_files if match_condition(file) and file not in ignored_model_list
            ]

            # if len(model_files) == 0:
            #     model_files = [default_model]

            cur_model = selected_model if (selected_model and selected_model in model_files) else None
            current_model_index = (
                model_files.index(cur_model) if (cur_model and cur_model in model_files) else 0
            )

            if model_files and len(model_files):
                explorer_gen_model = st.selectbox(
                    label="Styling model:",
                    options=model_files,
                    key=f"{position}_{type}_explorer_gen_model_video",
                    index=current_model_index,
                    # on_change=update_model,
                )

                if type == T2IModel.SDXL.value:
                    st.write("")
                    lightning = st.checkbox(
                        "Lightning Model",
                        help="Generate images faster with less quality.",
                        value=st.session_state["insp_lightning_mode"],
                    )

                    if st.session_state["insp_lightning_mode"] != lightning:
                        st.session_state["insp_lightning_mode"] = lightning
                        refresh_app()

                (
                    st.info("Please only select SDXL based models. Default models will be auto-downloaded.")
                    if type == T2IModel.SDXL.value
                    else st.info(
                        "Please only select SD3 based models. Default models will be auto-downloaded."
                    )
                )
            else:
                st.write("")
                st.info(info_msg)
                explorer_gen_model = default_model

    with tab2:
        # NOTE: makes sure to add 'xl' in these filenames because that is the only filter rn for sdxl models (will update in the future)

        extra_model_list = (
            SDXL_MODEL_DOWNLOAD_LIST if type == T2IModel.SDXL.value else SD3_MODEL_DOWNLOAD_LIST
        )

        select_col1, select_col2 = st.columns([1, 1])
        with select_col1:
            download_model = st.selectbox(
                label="Download model:",
                options=extra_model_list.keys(),
                key=f"{position}_{type}_explorer_gen_model_download",
                index=0,
                # on_change=update_model,
            )

        if extra_model_list[download_model]["desc"]:
            with select_col2:
                st.write("")
                st.info(
                    extra_model_list[download_model]["desc"],
                )

        if st.button("Download"):
            download_file_widget(
                extra_model_list[download_model]["url"],
                extra_model_list[download_model]["filename"],
                checkpoints_dir,
            )
            refresh_app()

    return explorer_gen_model


def model_selector_element(type=T2IModel.SDXL.value, position="explorer", selected_model=None):
    if gpu_enabled:
        return model_selector_offline_element(
            type=type, position="explorer", selected_model=None
        )
    else:
        return model_selector_online_element(
            type=type, position="explorer", selected_model=None
        )
