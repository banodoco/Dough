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
        items_per_row = st_memory.slider("How many frames per row?", min_value=3, max_value=7, value=4, step=1, key="items_per_row_slider")

    if stage == 'Key Frames':
        for shot in shot_list:
            shot_keyframe_element(shot.uuid, items_per_row)
            st.markdown("***")
        st.markdown("### Add new shot")
        shot1,shot2,shot3 = st.columns([1,1,3])
        with shot1:
            new_shot_name = st.text_input("Shot Name:",max_chars=25)
        with shot2:
            st.write("")
            st.write("")
            if st.button('Add new shot', type="primary"):
                new_shot = add_new_shot(shot.project.uuid)                
                if new_shot_name != "":
                    data_repo.update_shot(uuid=new_shot.uuid, name=new_shot_name)                                        
                st.rerun()
        
    else:
        for idx, shot in enumerate(shot_list):
            if idx % items_per_row == 0:
                grid = st.columns(items_per_row)
            with grid[idx % items_per_row]:
                shot_video_element(shot.uuid)
            if (idx + 1) % items_per_row == 0 or idx == len(shot_list) - 1:
                st.markdown("***")