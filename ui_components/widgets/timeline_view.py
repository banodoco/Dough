import streamlit as st
from ui_components.methods.common_methods import add_new_shot
from ui_components.widgets.shot_view import shot_keyframe_element, shot_video_element
from utils.data_repo.data_repo import DataRepo
from utils import st_memory


def timeline_view(shot_uuid, stage):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    shot_list = data_repo.get_shot_list(shot.project.uuid)
    
    st.markdown("***")
    
    _, header_col_2 = st.columns([5.5,1.5])
            
    with header_col_2:
        items_per_row = st_memory.slider("How many frames per row?", min_value=3, max_value=7, value=5, step=1, key="items_per_row_slider")

    if stage == 'Key Frames':
        for shot in shot_list:
            shot_keyframe_element(shot.uuid, items_per_row)
            st.markdown("***")
        if st.button('Add new shot', type="primary"):
            add_new_shot(shot.project.uuid)
            st.rerun()
        
    else:
        grid = st.columns(items_per_row)
        for idx, shot in enumerate(shot_list):
            with grid[idx%items_per_row]:
                shot_video_element(shot.uuid)
