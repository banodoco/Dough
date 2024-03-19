import streamlit as st
from ui_components.widgets.frame_selector import frame_selector_widget, frame_view
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from ui_components.widgets.animation_style_element import animation_style_element
from utils.data_repo.data_repo import DataRepo
from ui_components.widgets.sidebar_logger import sidebar_logger

def animate_shot_page(shot_uuid: str, h2):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    with st.sidebar:
        frame_selector_widget(show_frame_selector=False)

        st.write("")                    
        with st.expander("🔍 Generation log", expanded=True):
            # if st_memory.toggle("Open", value=True, key="generaton_log_toggle"):
            sidebar_logger(st.session_state["shot_uuid"])
        
        st.write("")
        # frame_view(view='Video',show_current_frames=False)

    st.markdown(f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}] > :blue[{shot.name}]")
    st.markdown("***")
    
    variant_comparison_grid(st.session_state['shot_uuid'], stage="Shots")
    animation_style_element(st.session_state['shot_uuid'])

    st.markdown("***")
    