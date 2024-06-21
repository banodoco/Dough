import io
import time
from typing import List
import zipfile
from shared.constants import COMFY_BASE_PATH, InferenceLogTag, InternalFileTag, InternalFileType, SortOrder
import streamlit as st
import os
import requests
import shutil
from zipfile import ZipFile
from io import BytesIO
from ui_components.constants import CreativeProcessType
from ui_components.methods.file_methods import get_file_bytes_and_extension
from ui_components.methods.video_methods import upscale_video
from ui_components.models import InternalFileObject, InternalProjectObject, InternalShotObject
from ui_components.widgets.inspiration_engine import inspiration_engine_element
from ui_components.widgets.shot_view import create_video_download_button
from ui_components.widgets.sm_animation_style_element import video_shortlist_btn
from ui_components.widgets.timeline_view import timeline_view
from ui_components.components.explorer_page import gallery_image_view
from ui_components.widgets.variant_comparison_grid import get_video_upscale_dict, uspcale_expander_element
from utils import st_memory
from utils.data_repo.data_repo import DataRepo

from ui_components.widgets.sidebar_logger import sidebar_logger
from ui_components.components.explorer_page import generate_images_element


def upscaling_page(project_uuid: str):
    video_list: List[InternalFileObject] = get_final_video_list(project_uuid)
    upscale_in_progress_arr = get_video_upscale_dict(project_uuid)

    st.markdown(f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}]")
    st.markdown("***")

    # -------------- sidebar -------------------
    with st.sidebar:
        with st.expander("🔍 Generation log", expanded=True):
            sidebar_logger(st.session_state["shot_uuid"])

    # ------------- page header ----------------
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

    with header_col1:
        st.markdown("### ✨ Upscale videos")
        st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")
    st.write("")
    st.write("")

    main_clip_list = [1, 2, 3, 4, 5]

    # upscaling
    with header_col2:
        uspcale_expander_element(
            [v.uuid for v in video_list],
            heading="Bulk upscale",
            btn_text="Upscale All shortlisted clips",
            ui_key=str(project_uuid),
        )

    # export all files
    with header_col3:
        with st.expander("Export all shortlisted videos", expanded=False):
            if not len(main_clip_list):
                st.info("No videos available in the project.")

            else:
                create_multi_video_download_button([v.location for v in video_list], ui_key="upscaling_page")

    # -------------- video grid --------------------
    if video_list:
        ## display 2 per row
        for i in range(0, len(video_list), 2):
            col1, col2 = st.columns(2)
            with col1:
                display_video(video_list[i], video_list[i].uuid in upscale_in_progress_arr)
            with col2:
                if i + 1 < len(video_list):
                    display_video(video_list[i + 1], video_list[i + 1].uuid in upscale_in_progress_arr)
            st.markdown("***")

    else:
        st.info("You need to shortlist videos in the Animate Shot view for them to appear here.")


def display_video(video_file: InternalFileObject, upscale_in_progress=False):
    data_repo = DataRepo()
    video_shot: InternalShotObject = data_repo.get_shot_from_uuid(video_file.origin_shot_uuid)
    upscaled_video = video_file.inference_log.generation_tag == InferenceLogTag.UPSCALED_VIDEO.value

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"#### From shot '{video_shot.name}'")

        if not upscaled_video and not upscale_in_progress:
            video_shortlist_btn(video_file.uuid, "remove_from_shortlist")
        st.markdown("***")

        if upscale_in_progress:
            st.info("Upscale pending")
        elif upscaled_video:
            st.success("Upscaled video")
        else:
            uspcale_expander_element(
                [video_file.uuid],
                heading="Upscale settings",
                btn_text="Queue for upscaling",
                ui_key=str(video_file.uuid),
                default_expanded=True,
            )

    with col2:
        st.video(video_file.location)
        create_video_download_button(video_file.location, ui_key="upscale_page")


def get_final_video_list(project_uuid):
    """
    returns the list of the shortlisted + upscaled videos
    """
    data_repo = DataRepo()

    page_number = 1
    num_items_per_page = 100
    shortlisted_file_filter = {
        "file_type": InternalFileType.VIDEO.value,
        "project_id": project_uuid,
        "page": page_number,
        "data_per_page": num_items_per_page,
        "sort_order": SortOrder.DESCENDING.value,
    }

    video_list, _ = data_repo.get_all_file_list(**shortlisted_file_filter)
    final_list = []

    for video in video_list:
        if (
            video.tag == InternalFileTag.SHORTLISTED_VIDEO.value
            or video.inference_log.generation_tag == InferenceLogTag.UPSCALED_VIDEO.value
        ):
            final_list.append(video)

    return final_list


def create_multi_video_download_button(video_location_list, ui_key="temp"):
    if st.button("Prepare videos for download", use_container_width=True, key=ui_key + "_download_button"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as zip_file:
            for video_location in video_location_list:
                file_name = os.path.basename(video_location)
                file_bytes, _ = get_file_bytes_and_extension(video_location)
                zip_file.writestr(file_name, file_bytes)

        zip_buffer.seek(0)

        st.download_button(
            label="Download videos as ZIP",
            data=zip_buffer,
            file_name="videos.zip",
            mime="application/zip",
            key=ui_key + "_download_gen",
            use_container_width=True,
        )
