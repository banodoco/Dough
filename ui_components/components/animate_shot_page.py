import json
import time
import streamlit as st
from shared.constants import InferenceParamType, InferenceStatus, InferenceType, InternalFileType
from ui_components.components.video_rendering_page import sm_video_rendering_page, two_img_realistic_interpolation_page
from ui_components.models import InternalShotObject
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from utils import st_memory
from utils.constants import AnimateShotMethod
from utils.data_repo.data_repo import DataRepo
from ui_components.widgets.sidebar_logger import sidebar_logger
from utils.enum import ExtendedEnum


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
    video_rendering_page(shot_uuid, selected_variant)


            

def video_rendering_page(shot_uuid, selected_variant):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    file_uuid_list = []
    if f"type_of_animation_{shot.uuid}" not in st.session_state:
        st.session_state[f"type_of_animation_{shot.uuid}"] = 0
    if st.session_state[f"type_of_animation_{shot.uuid}"] == 0:   # AnimateShotMethod.BATCH_CREATIVE_INTERPOLATION.value
        # loading images from a particular video variant
        if selected_variant:
            log = data_repo.get_inference_log_from_uuid(selected_variant)
            shot_data = json.loads(log.input_params)
            file_uuid_list = shot_data.get('origin_data', json.dumps({})).get('settings', {}).get('file_uuid_list', [])

    else:
        # hackish sol, will fix later
        for idx in range(2):
            if f'img{idx+1}_uuid_{shot_uuid}' in st.session_state and st.session_state[f'img{idx+1}_uuid_{shot_uuid}']:
                file_uuid_list.append(st.session_state[f'img{idx+1}_uuid_{shot_uuid}'])            
                
        if not (f'video_desc_{shot_uuid}' in st.session_state and st.session_state[f'video_desc_{shot_uuid}']):
            st.session_state[f'video_desc_{shot_uuid}'] = ""
    
    # picking current images if no file_uuids are found 
    # (either no variant was selected or no prev img in session_state was present)
    if not (file_uuid_list and len(file_uuid_list)):
        for timing in shot.timing_list:
            if timing.primary_image and timing.primary_image.location:
                file_uuid_list.append(timing.primary_image.uuid)
    
    img_list = data_repo.get_all_file_list(uuid__in=file_uuid_list, file_type=InternalFileType.IMAGE.value)[0]    
    
    headline1, _, headline3 = st.columns([1, 1, 1])
    with headline1:
        st.markdown("### 🎥 Generate animations")  
        st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")
    with headline3:
        with st.expander("Type of animation", expanded=False):
            type_of_animation = st_memory.radio("What type of animation would you like to generate?", \
                options=AnimateShotMethod.value_list(), horizontal=True, \
                    help="**Batch Creative Interpolaton** lets you input multple images and control the motion and style of each frame - resulting in a fluid, surreal and highly-controllable motion. \n\n **2-Image Realistic Interpolation** is a simpler way to generate animations - it generates a video by interpolating between two images, and is best for realistic motion.",key=f"type_of_animation_{shot.uuid}")
    
    if type_of_animation == AnimateShotMethod.BATCH_CREATIVE_INTERPOLATION.value:
        sm_video_rendering_page(shot_uuid, img_list)
    else:        
        two_img_realistic_interpolation_page(shot_uuid, img_list)

    st.markdown("***")