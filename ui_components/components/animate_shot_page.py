import json
import streamlit as st
from shared.constants import InternalFileType
from ui_components.components.video_rendering_page import sm_video_rendering_page
from ui_components.models import InternalShotObject
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from utils.data_repo.data_repo import DataRepo
from ui_components.widgets.sidebar_logger import sidebar_logger

def animate_shot_page(shot_uuid: str, h2):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    st.session_state['project_uuid'] = str(shot.project.uuid)

    with st.sidebar:
        frame_selector_widget(show_frame_selector=False)

        st.write("")                    
        with st.expander("🔍 Generation log", expanded=True):
            sidebar_logger(shot_uuid)
        
        st.write("")

    st.markdown(f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}] > :blue[{shot.name}]")
    st.markdown("***")
    
    selected_variant = variant_comparison_grid(shot_uuid, stage="Shots")
    file_uuid_list = []
    # loading images from a particular video variant
    if selected_variant:
        log = data_repo.get_inference_log_from_uuid(selected_variant)
        shot_data = json.loads(log.input_params)
        file_uuid_list = shot_data.get('origin_data', json.dumps({})).get('settings', {}).get('file_uuid_list', [])
    # picking current images if no variant is selected
    else:
        for timing in shot.timing_list:
            if timing.primary_image and timing.primary_image.location:
                file_uuid_list.append(timing.primary_image.uuid)
    
    img_list = data_repo.get_all_file_list(uuid__in=file_uuid_list, file_type=InternalFileType.IMAGE.value)[0]
    sm_video_rendering_page(shot_uuid, img_list)

    st.markdown("***")