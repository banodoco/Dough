import os
import time
import streamlit as st

from shared.constants import COMFY_BASE_PATH
from ui_components.widgets.download_file_progress_bar import download_file_widget


def model_selector_element():
    tab1, tab2 = st.tabs(["Choose Model", "Download Models"])
    default_model = "sd_xl_base_1.0.safetensors"
    model_data = None
    checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            explorer_gen_model = ""

            # TODO: make a common interface for accessing different types of files
            all_files = os.listdir(checkpoints_dir)
            ignored_model_list = ["dynamicrafter_512_interp_v1.ckpt", "sd_xl_refiner_1.0.safetensors"]
            model_files = [
                file for file in all_files if file.endswith(".safetensors") or file.endswith(".ckpt")
            ]

            model_files = [
                file for file in model_files if "xl" in file.lower() and file not in ignored_model_list
            ]

            # if len(model_files) == 0:
            #     model_files = [default_model]

            cur_model = st.session_state.get(f"ckpt_explorer_model", default_model)
            current_model_index = (
                model_files.index(cur_model) if (cur_model and cur_model in model_files) else 0
            )

            if model_files and len(model_files):
                explorer_gen_model = st.selectbox(
                    label="Styling model:",
                    options=model_files,
                    key="explorer_gen_model_video",
                    index=current_model_index,
                    # on_change=update_model,
                )
            else:
                st.write("")
                st.info("Default model base SDXL would be selected")
                explorer_gen_model = default_model

        with col2:
            if len(all_files) == 0:
                st.write("")
                st.info("This is the default model - to download more, go to the Download Models tab.")
            else:
                st.write("")
                st.info("To download more models, go to the Download Models tab.")

    with tab2:
        extra_model_list = {
            "Base SDXL": {
                "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true",
                "filename": "sd_xl_base_1.0.safetensors",
                "desc": "Base SDXL model",
            },
            "Anima Pencil XL": {
                "url": "https://civitai.com/api/download/models/505691",
                "filename": "animaPencilXL_v400.safetensors",
                "desc": "Good for anime art",
            },
            "Araminta XL": {
                "url": "https://civitai.com/api/download/models/561766",
                "filename": "theAramintaxl_cv5.safetensors",
                "desc": "General purpose model",
            },
            "MKLAN Art XL": {
                "url": "https://civitai.com/api/download/models/528345",
                "filename": "mklanArtVersion_mklan2311art.safetensors",
                "desc": "Suited for creative outputs",
            },
            "MKLAN Realistic XL": {
                "url": "https://civitai.com/api/download/models/528311",
                "filename": "mklanRealistic_mklan230realistic.safetensors",
                "desc": "Realistic model",
            },
            "Traditional Painting XL": {
                "url": "https://civitai.com/api/download/models/529269",
                "filename": "traditionalPainting_v02.safetensors",
                "desc": "Traditional paiting style",
            },
        }

        select_col1, select_col2 = st.columns([1, 1])
        with select_col1:
            download_model = st.selectbox(
                label="Download model:",
                options=extra_model_list.keys(),
                key="explorer_gen_model_download",
                index=0,
                # on_change=update_model,
            )

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

    # print("returning: // ", explorer_gen_model)
    return explorer_gen_model
