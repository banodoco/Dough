import streamlit as st
from ui_components.widgets.shot_view import shot_keyframe_element
from ui_components.components.explorer_page import gallery_image_view
from ui_components.components.explorer_page import generate_images_element
from ui_components.components.frame_styling_page import frame_styling_page
from ui_components.widgets.frame_selector import frame_selector_widget, frame_view
from utils import st_memory
from utils.data_repo.data_repo import DataRepo


def adjust_shot_page(shot_uuid: str, h2):
    with st.sidebar:
        frame_selection = frame_selector_widget(show_frame_selector=True)
    
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    if frame_selection == "":
        with st.sidebar:       
            # frame_view(view='Video',show_current_frames=False)
            st.write("")
            with st.expander("📋 Explorer Shortlist",expanded=True):
                if st_memory.toggle("Open", value=True, key="explorer_shortlist_toggle"):
                    project_setting = data_repo.get_project_setting(shot.project.uuid)
                    number_of_pages = project_setting.total_shortlist_gallery_pages
                    page_number = 0
                    gallery_image_view(shot.project.uuid, shortlist=True,view=['add_and_remove_from_shortlist','add_to_this_shot'], shot=shot, sidebar=True)
        
        st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}] > :orange[{shot.name}]")
        st.markdown("***")
        shot_keyframe_element(st.session_state["shot_uuid"], 4, position="Individual")

        st.markdown("### ✨ Generate Images ----------")        
        with st.expander("", expanded=True):
            generate_images_element(position='explorer', project_uuid=shot.project.uuid, timing_uuid=None, shot_uuid=shot.uuid)
        st.markdown("***")

        st.markdown("***")
        gallery_image_view(shot.project.uuid, shortlist=False,view=['add_and_remove_from_shortlist','add_to_this_shot','view_inference_details','shot_chooser'], shot=shot,sidebar=False)
    else:
        frame_styling_page(st.session_state["shot_uuid"], h2)