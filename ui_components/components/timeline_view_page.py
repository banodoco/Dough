
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
                    
        with st.expander("📋 Explorer Shortlist",expanded=True):

            if st_memory.toggle("Open", value=True, key="explorer_shortlist_toggle"):
                project_setting = data_repo.get_project_setting(shot.project.uuid)
                # page_number = st.radio("Select page:", options=range(1, project_setting.total_shortlist_gallery_pages + 1), horizontal=True)
                gallery_image_view(shot.project.uuid, shortlist=True,view=["add_and_remove_from_shortlist","add_to_any_shot"], shot=shot,sidebar=True)
                        
    with h2:
        st.session_state['view'] = option_menu(None, views, icons=['palette', 'camera-reels', "hourglass", 'stopwatch'], menu_icon="cast", orientation="vertical", key="secti2on_selector", styles={
                                                "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "orange"}}, manual_select=st.session_state["manual_select"])
        
        if st.session_state["manual_select"] != None:
            st.session_state["manual_select"] = None

    st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}] > :orange[{st.session_state['view']}]")

    st.markdown("***")
    timeline_view(st.session_state["shot_uuid"], st.session_state['view'])