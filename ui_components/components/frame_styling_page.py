import streamlit as st

from ui_components.widgets.cropping_element import cropping_selector_element
from ui_components.widgets.frame_selector import frame_selector_widget, frame_view
from ui_components.widgets.add_key_frame_element import add_key_frame, add_key_frame_element
from ui_components.widgets.timeline_view import timeline_view
from ui_components.components.explorer_page import generate_images_element
from ui_components.widgets.inpainting_element import inpainting_element
from ui_components.widgets.drawing_element import drawing_element
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from utils import st_memory

from ui_components.constants import CreativeProcessType
from utils.data_repo.data_repo import DataRepo


def frame_styling_page(shot_uuid: str, h2):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)


    if len(timing_list) == 0:
  
        st.markdown("#### There are no frames present in this shot yet.")

    else:
        with st.sidebar:     
                                                                
            st.session_state['styling_view'] = st_memory.menu('',\
                                    ["Crop","Generate"], \
                                        icons=['magic', 'crop', "paint-bucket", 'pencil'], \
                                            menu_icon="cast", default_index=st.session_state.get('styling_view_index', 0), \
                                                key="styling_view_selector", orientation="horizontal", \
                                                    styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "orange"}})
            
            frame_view(view="Key Frame")

        st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['frame_styling_view_type']}] > :orange[{shot.name} - #{st.session_state['current_frame_index']}] > :blue[{st.session_state['styling_view']}]")

        variant_comparison_grid(st.session_state['current_frame_uuid'], stage=CreativeProcessType.STYLING.value)    

        
        if st.session_state['styling_view'] == "Generate":
            
            with st.expander("🛠️ Generate Variants", expanded=True):
                generate_images_element(position='individual', project_uuid=shot.project.uuid, timing_uuid=st.session_state['current_frame_uuid'])
                                            
        elif st.session_state['styling_view'] == "Crop":
            with st.expander("🤏 Crop, Move & Rotate", expanded=True):                    
                cropping_selector_element(shot_uuid)

        elif st.session_state['styling_view'] == "Inpaint":
            with st.expander("🌌 Inpainting", expanded=True):
                inpainting_element(st.session_state['current_frame_uuid'])

        elif st.session_state['styling_view'] == "Scribble":
            with st.expander("📝 Draw On Image", expanded=True):
                drawing_element(shot_uuid)

        st.markdown("***")

            