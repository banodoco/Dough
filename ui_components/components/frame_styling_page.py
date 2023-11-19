import streamlit as st
from shared.constants import ViewType

from ui_components.widgets.cropping_element import cropping_selector_element
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.add_key_frame_element import add_key_frame, add_key_frame_element
from ui_components.widgets.timeline_view import timeline_view
from ui_components.widgets.explorer_element import generate_images_element
from ui_components.widgets.animation_style_element import animation_style_element
from ui_components.widgets.video_cropping_element import video_cropping_element
from ui_components.widgets.inpainting_element import inpainting_element
from ui_components.widgets.drawing_element import drawing_element
from ui_components.widgets.sidebar_logger import sidebar_logger
from ui_components.widgets.explorer_element import explorer_element,gallery_image_view
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from utils import st_memory

from ui_components.constants import CreativeProcessType, DefaultProjectSettingParams, DefaultTimingStyleParams

from utils.data_repo.data_repo import DataRepo


def frame_styling_page(shot_uuid: str):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)
    project_settings = data_repo.get_project_setting(shot.project.uuid)

    if "strength" not in st.session_state:
        st.session_state['strength'] = DefaultProjectSettingParams.batch_strength
        st.session_state['prompt_value'] = DefaultProjectSettingParams.batch_prompt
        st.session_state['model'] = None
        st.session_state['negative_prompt_value'] = DefaultProjectSettingParams.batch_negative_prompt
        st.session_state['guidance_scale'] = DefaultProjectSettingParams.batch_guidance_scale
        st.session_state['seed'] = DefaultProjectSettingParams.batch_seed
        st.session_state['num_inference_steps'] = DefaultProjectSettingParams.batch_num_inference_steps
        st.session_state['transformation_stage'] = DefaultProjectSettingParams.batch_transformation_stage
        
    if "current_frame_uuid" not in st.session_state:        
        timing = data_repo.get_timing_list_from_shot(shot_uuid)[0]
        st.session_state['current_frame_uuid'] = timing.uuid
        st.session_state['current_frame_index'] = timing.aux_frame_index + 1
    
    if 'frame_styling_view_type' not in st.session_state:
        st.session_state['frame_styling_view_type'] = "Individual"
        st.session_state['frame_styling_view_type_index'] = 0

    if st.session_state['change_view_type'] == True:  
        st.session_state['change_view_type'] = False
    
    if st.session_state['frame_styling_view_type'] == "Timeline" or st.session_state['frame_styling_view_type'] == "Explorer":
        st.markdown(
            f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{st.session_state['page']}]")
    else:
        if st.session_state['page'] == "Key Frames":
            st.markdown(
                f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{shot.name}] > :blue[Frame #{st.session_state['current_frame_index']}]")
        else:
            st.markdown(f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{st.session_state['page']}] > :blue[{shot.name}]")

    project_settings = data_repo.get_project_setting(shot.project.uuid)

    if st.session_state['frame_styling_view_type'] == "Explorer":

        explorer_element(shot.project.uuid)

    # -------------------- INDIVIDUAL VIEW ----------------------
    elif st.session_state['frame_styling_view_type'] == "Individual":
        with st.sidebar:
            frame_selector_widget()
                
        if st.session_state['page'] == CreativeProcessType.MOTION.value:
            variant_comparison_grid(shot_uuid, stage=CreativeProcessType.MOTION.value)

            st.session_state['shot_view'] = st_memory.menu('',\
                        ["Animate Frames", "Basic Cropping"], \
                            icons=['film', 'crop', "paint-bucket", 'pencil'], \
                                menu_icon="cast", default_index=st.session_state.get('styling_view_index', 0), \
                                    key="styling_view_selector", orientation="horizontal", \
                                        styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#66A9BE"}})

            if st.session_state['shot_view'] == "Animate Frames":
                with st.expander("🎬 Choose Animation Style & Create Variants", expanded=True):
                    animation_style_element(shot_uuid)

            elif st.session_state['shot_view'] == "Basic Cropping":
                with st.expander("🤏 Crop, Move & Rotate Image", expanded=True):
                    video_cropping_element(shot_uuid)

        elif st.session_state['page'] == CreativeProcessType.STYLING.value:

            variant_comparison_grid(st.session_state['current_frame_uuid'], stage=CreativeProcessType.STYLING.value)

            st.markdown("***")
            st.session_state['styling_view'] = st_memory.menu('',\
                                    ["Generate Variants", "Crop, Move & Rotate Image", "Inpainting & BG Removal","Draw On Image"], \
                                        icons=['magic', 'crop', "paint-bucket", 'pencil'], \
                                            menu_icon="cast", default_index=st.session_state.get('styling_view_index', 0), \
                                                key="styling_view_selector", orientation="horizontal", \
                                                    styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#66A9BE"}})


            if st.session_state['styling_view'] == "Generate Variants":
                with st.expander("🛠️ Generate Variants + Prompt Settings", expanded=True):
                
                    generate_images_element(shot.project.uuid,data_repo, position='individual')
                                                
            elif st.session_state['styling_view'] == "Crop, Move & Rotate Image":
                with st.expander("🤏 Crop, Move & Rotate Image", expanded=True):                    
                    cropping_selector_element(shot_uuid)

            elif st.session_state['styling_view'] == "Inpainting & BG Removal":
                with st.expander("🌌 Inpainting, Background Removal & More", expanded=True):
                    inpainting_element(st.session_state['current_frame_uuid'])

            elif st.session_state['styling_view'] == "Draw On Image":
                with st.expander("📝 Draw On Image", expanded=True):
                    drawing_element(timing_list,project_settings, shot_uuid)
            st.markdown("***")                       
            with st.expander("➕ Add Key Frame", expanded=True):
                selected_image, inherit_styling_settings  = add_key_frame_element(shot_uuid)
                if st.button(f"Add key frame",type="primary",use_container_width=True):
                    add_key_frame(selected_image, inherit_styling_settings, shot_uuid)
                    st.rerun()

    # -------------------- TIMELINE VIEW --------------------------       
    elif st.session_state['frame_styling_view_type'] == "Timeline":
        if st.session_state['page'] == "Key Frames":

            with st.sidebar:
                with st.expander("📋 Explorer Shortlist",expanded=True):
                    if st_memory.toggle("Open", value=True, key="explorer_shortlist_toggle"):
                        project_setting = data_repo.get_project_setting(shot.project.uuid)
                        page_number = st.radio("Select page", options=range(1, project_setting.total_gallery_pages), horizontal=True)
                        gallery_image_view(shot.project.uuid, page_number=page_number, num_items_per_page=10, open_detailed_view_for_all=False, shortlist=True, num_columns=2,view="sidebar")
                                
            timeline_view(shot_uuid, "Key Frames")
        elif st.session_state['page'] == "Shots":
            timeline_view(shot_uuid, "Shots")
    
    # -------------------- SIDEBAR NAVIGATION --------------------------
    with st.sidebar:
        # with st.expander("🔍 Generation Log", expanded=True):    
            # sidebar_logger(shot_uuid)        
        
        with st.expander("🔍 Generation Log", expanded=True):
            if st_memory.toggle("Open", value=True, key="generaton_log_toggle"):
                sidebar_logger(shot_uuid)
        st.markdown("***")

