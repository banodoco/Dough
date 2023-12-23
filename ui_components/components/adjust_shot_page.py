import streamlit as st
from ui_components.widgets.shot_view import shot_keyframe_element
from ui_components.components.explorer_page import gallery_image_view
from ui_components.components.explorer_page import generate_images_element
from ui_components.widgets.frame_selector import frame_selector_widget
from utils import st_memory



def adjust_shot_page(shot_uuid: str, h2,data_repo,shot,timing_list, project_settings):
    with h2:
        frame_selector_widget(show=['shot_selector'])

    st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}] > :orange[{shot.name}]")

    st.markdown("***")

    shot_keyframe_element(st.session_state["shot_uuid"], 4, position="Individual")
    # with st.expander("📋 Explorer Shortlist",expanded=True):
    shot_explorer_view = st_memory.menu('',["Shortlist", "Explore"],                        
        icons=['grid-3x3','airplane'],
        menu_icon="cast", 
        default_index=st.session_state.get('shot_explorer_view', 0),
        key="shot_explorer_view", orientation="horizontal",
        styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#868c91"}})
    
    st.markdown("***")

    if shot_explorer_view == "Shortlist":                    
        project_setting = data_repo.get_project_setting(shot.project.uuid)
        page_number = st.radio("Select page:", options=range(1, project_setting.total_shortlist_gallery_pages + 1), horizontal=True)                        
        st.markdown("***")
        gallery_image_view(shot.project.uuid, page_number=page_number, num_items_per_page=8, open_detailed_view_for_all=False, shortlist=True, num_columns=4,view="individual_shot", shot=shot)
    elif shot_explorer_view == "Explore":
        project_setting = data_repo.get_project_setting(shot.project.uuid)
        page_number = st.radio("Select page:", options=range(1, project_setting.total_shortlist_gallery_pages + 1), horizontal=True)
        generate_images_element(position='explorer', project_uuid=shot.project.uuid, timing_uuid=st.session_state['current_frame_uuid'])
        st.markdown("***")
        gallery_image_view(shot.project.uuid, page_number=page_number, num_items_per_page=8, open_detailed_view_for_all=False, shortlist=False, num_columns=4,view="individual_shot", shot=shot)