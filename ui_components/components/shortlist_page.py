import streamlit as st
from ui_components.components.explorer_page import gallery_image_view
from utils.data_repo.data_repo import DataRepo
from utils import st_memory


def shortlist_page(project_uuid):

    st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}]")    

    data_repo = DataRepo()
    project_setting = data_repo.get_project_setting(project_uuid)       
    # columnn_selecter()
    k1,k2 = st.columns([5,1])
    # shortlist_page_number = k1.radio("Select page", options=range(1, project_setting.total_shortlist_gallery_pages), horizontal=True, key="shortlist_gallery")
    # with k2:
      #  open_detailed_view_for_all = st_memory.toggle("Open prompt details for all:", key='shortlist_gallery_toggle',value=False)
    st.markdown("***")
    gallery_image_view(project_uuid, True,view=['view_inference_details','add_to_any_shot','add_and_remove_from_shortlist'])
    