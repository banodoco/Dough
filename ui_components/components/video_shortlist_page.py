import time
from shared.constants import COMFY_BASE_PATH
import streamlit as st
import os
import requests
import shutil
from zipfile import ZipFile
from io import BytesIO
from ui_components.constants import CreativeProcessType
from ui_components.methods.video_methods import upscale_video
from ui_components.widgets.sm_animation_style_element import video_shortlist_view
from ui_components.widgets.timeline_view import timeline_view
from ui_components.components.explorer_page import gallery_image_view
from utils import st_memory
from utils.data_repo.data_repo import DataRepo

from ui_components.widgets.sidebar_logger import sidebar_logger
from ui_components.components.explorer_page import generate_images_element


def video_shortlist_page(shot_uuid: str, h2):

    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    video_shortlist_view(shot.project.uuid)