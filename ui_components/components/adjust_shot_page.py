import streamlit as st
from ui_components.widgets.shot_view import shot_keyframe_element
from ui_components.components.explorer_page import gallery_image_view
from ui_components.components.explorer_page import generate_images_element
from ui_components.components.frame_styling_page import frame_styling_page
from ui_components.widgets.frame_selector import frame_selector_widget
from utils import st_memory
from ui_components.widgets.sidebar_logger import sidebar_logger
from utils.data_repo.data_repo import DataRepo


def adjust_shot_page(shot_uuid: str, h2):
    with st.sidebar:
        frame_selection = frame_selector_widget(show_frame_selector=True)

    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    if frame_selection == "":
        with st.sidebar:
            st.write("")

            with st.expander("🔍 Generation log", expanded=True):
                # if st_memory.toggle("Open", value=True, key="generaton_log_toggle"):
                sidebar_logger(st.session_state["shot_uuid"])

        st.markdown(
            f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}] > :blue[{shot.name}]"
        )
        st.markdown("***")

        column1, column2 = st.columns([2, 1.35])
        with column1:
            st.markdown(f"### 🎬 '{shot.name}' frames")
            st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")
            items_per_row = st_memory.slider("Items per row:", 1, 10, 6, key="items_per_row")

        shot_keyframe_element(st.session_state["shot_uuid"], items_per_row, column2, position="Individual")

    else:
        frame_styling_page(st.session_state["shot_uuid"])
