import os
import time
import streamlit as st

from shared.constants import COMFY_BASE_PATH
from ui_components.widgets.download_file_progress_bar import download_file_widget
from utils.constants import T2IModel


def model_selector_element(type=T2IModel.SDXL.value, position="explorer", selected_model=None):
    tab1, tab2 = st.tabs(["Choose Model", "Download Models"])
    default_model = (
        "Juggernaut-XL_v9_v2.safetensors"
        if type == T2IModel.SDXL.value
        else "sd3_medium_incl_clips.safetensors"
    )

    info_msg = (
        "Juggernaut-XL_v9 will be selected as default"
        if type == T2IModel.SDXL.value
        else "Default model base SD3 medium fp8 would be selected"
    )
    checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            explorer_gen_model = ""

            # TODO: make a common interface for accessing different types of files
            all_files = os.listdir(checkpoints_dir)
            ignored_model_list = [
                "dynamicrafter_512_interp_v1.ckpt",
                "sd_xl_refiner_1.0.safetensors",
                "sd_xl_refiner_1.0_0.9vae.safetensors",
            ]
            model_files = [
                file for file in all_files if file.endswith(".safetensors") or file.endswith(".ckpt")
            ]

            if type == T2IModel.SDXL.value:
                match_condition = lambda file: file and "xl" in file.lower()
            else:
                match_condition = lambda file: file and "sd3" in file.lower()

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
            else:
                st.write("")
                st.info(info_msg)
                explorer_gen_model = default_model

        with col2:
            if len(all_files) == 0:
                st.write("")
                st.info("This is the default model - to download more, go to the Download Models tab.")
            else:
                st.write("")
                st.info("To download more models, go to the Download Models tab.")

    with tab2:
        # NOTE: makes sure to add 'xl' in these filenames because that is the only filter rn for sdxl models (will update in the future)
        sdxl_model_download_list = {
            "Juggernaut-XL_v9": {
                "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
                "filename": "Juggernaut-XL_v9_v2.safetensors",
                "desc": "Good general purpose model",
            },
            "Juggernaut-XL-Lightning": {
                "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/resolve/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors",
                "filename": "Juggernaut-XL_Lightning_4Steps.safetensors",
                "desc": "Faster version of Juggernaut-XL",
            },
        }

        sd3_model_download_list = {
            "SD3 Medium fp8 clip + T5xxl": {
                "url": "https://huggingface.co/lone682/sd3/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors?download=true",
                "filename": "sd3_medium_incl_clips_t5xxlfp8.safetensors",
                "desc": "",
            },
            "SD3 Medium fp16 clip + T5xxl": {
                "url": "https://huggingface.co/lone682/sd3/resolve/main/sd3_medium_incl_clips_t5xxlfp16.safetensors?download=true",
                "filename": "sd3_medium_incl_clips_t5xxlfp16.safetensors",
                "desc": "",
            },
        }

        extra_model_list = (
            sdxl_model_download_list if type == T2IModel.SDXL.value else sd3_model_download_list
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
            st.rerun()

    return explorer_gen_model
