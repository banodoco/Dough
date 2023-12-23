import streamlit as st
from shared.constants import ViewType
from streamlit_option_menu import option_menu

from ui_components.widgets.cropping_element import cropping_selector_element
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.add_key_frame_element import add_key_frame, add_key_frame_element
from ui_components.widgets.timeline_view import timeline_view
from ui_components.components.explorer_page import generate_images_element
from ui_components.widgets.animation_style_element import animation_style_element
from ui_components.widgets.video_cropping_element import video_cropping_element
from ui_components.widgets.inpainting_element import inpainting_element
from ui_components.widgets.drawing_element import drawing_element
from ui_components.widgets.sidebar_logger import sidebar_logger
# from ui_components.components.explorer_page import explorer_element,gallery_image_view
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from ui_components.widgets.shot_view import shot_keyframe_element
from utils import st_memory


from ui_components.constants import CreativeProcessType, DefaultProjectSettingParams, DefaultTimingStyleParams

from utils.data_repo.data_repo import DataRepo


def frame_styling_page(shot_uuid: str, h2,data_repo,shot,timing_list, project_settings):

    if len(timing_list) == 0:
        with h2:         
            frame_selector_widget(show=['shot_selector','frame_selector'])
        
        st.markdown("#### There are no frames present in this shot yet.")
        


    
    else:
        with st.sidebar:     
            with h2:

                frame_selector_widget(show=['shot_selector','frame_selector'])
                                            
                st.session_state['styling_view'] = st_memory.menu('',\
                                        ["Generate", "Crop/Move", "Inpainting","Scribbling"], \
                                            icons=['magic', 'crop', "paint-bucket", 'pencil'], \
                                                menu_icon="cast", default_index=st.session_state.get('styling_view_index', 0), \
                                                    key="styling_view_selector", orientation="horizontal", \
                                                        styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "orange"}})

        st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['frame_styling_view_type']}] > :orange[{st.session_state['styling_view']}] > :blue[{shot.name} - #{st.session_state['current_frame_index']}]")

            
        if st.session_state['styling_view'] == "Generate":
            variant_comparison_grid(st.session_state['current_frame_uuid'], stage=CreativeProcessType.STYLING.value)
            with st.expander("🛠️ Generate Variants + Prompt Settings", expanded=True):
                generate_images_element(position='individual', project_uuid=shot.project.uuid, timing_uuid=st.session_state['current_frame_uuid'])
                                            
        elif st.session_state['styling_view'] == "Crop/Move":
            with st.expander("🤏 Crop, Move & Rotate", expanded=True):                    
                cropping_selector_element(shot_uuid)

        elif st.session_state['styling_view'] == "Inpainting":
            with st.expander("🌌 Inpainting", expanded=True):
                inpainting_element(st.session_state['current_frame_uuid'])

        elif st.session_state['styling_view'] == "Scribbling":
            with st.expander("📝 Draw On Image", expanded=True):
                drawing_element(timing_list,project_settings, shot_uuid)

            