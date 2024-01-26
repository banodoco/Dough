import streamlit as st
from ui_components.widgets.frame_selector import frame_selector_widget, frame_view
from ui_components.widgets.variant_comparison_grid import variant_comparison_grid
from ui_components.widgets.animation_style_element import animation_style_element
from utils.data_repo.data_repo import DataRepo

def animate_shot_page(shot_uuid: str, h2):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    with h2:
        frame_selector_widget(show_frame_selector=False)
    with st.sidebar:
        frame_view(view='Video')

    st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}] > :orange[{shot.name}]")
    st.markdown("***")
    variant_comparison_grid(st.session_state['shot_uuid'], stage="Shots")
    with st.expander("🎬 Choose Animation Style & Create Variants", expanded=True):
        animation_style_element(st.session_state['shot_uuid'])