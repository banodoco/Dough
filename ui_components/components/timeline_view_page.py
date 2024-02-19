import streamlit as st
from ui_components.constants import CreativeProcessType
from ui_components.widgets.timeline_view import timeline_view
from ui_components.components.explorer_page import gallery_image_view
from streamlit_option_menu import option_menu
from utils import st_memory
from utils.data_repo.data_repo import DataRepo

def timeline_view_page(shot_uuid: str, h2):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    with st.sidebar:
        views = CreativeProcessType.value_list()

        if "view" not in st.session_state:
            st.session_state["view"] = views[0]
            st.session_state["manual_select"] = None

        st.write("")         
        with st.expander("📋 Explorer Shortlist",expanded=True):
            if st_memory.toggle("Open", value=True, key="explorer_shortlist_toggle"):
                gallery_image_view(shot.project.uuid, shortlist=True, view=["add_and_remove_from_shortlist","add_to_any_shot"], shot=None, sidebar=True)
    
    st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}] > :orange[{st.session_state['view']}]")

    st.markdown("***")
    slider1, slider2 = st.columns([1,5])
    with slider1:
        show_video = st.toggle("Show Video:", value=False, key="show_video")
    if show_video:
        st.session_state["view"] = "Shots"
    else:
        st.session_state["view"] = "Key Frames"
    timeline_view(st.session_state["shot_uuid"], st.session_state['view'])