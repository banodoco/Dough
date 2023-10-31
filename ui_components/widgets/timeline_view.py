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
        items_per_row = st_memory.slider("How many frames per row?", min_value=1, max_value=10, value=5, step=1, key="items_per_row_slider")

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


    
    # for i in range(0, total_count, items_per_row):  # Step of items_per_row for grid
    #     grid = st.columns(items_per_row)  # Create items_per_row columns for grid
    #     for j in range(items_per_row):
    #         if i + j < total_count:  # Check if index is within range
    #             with grid[j]:
    #                 display_number = i + j + 1                            
    #                 if stage == 'Key Frames':
    #                     display_image(timing_uuid=shot_list[i + j].uuid, stage=WorkflowStageType.STYLED.value, clickable=False)
    #                 elif stage == 'Videos':
    #                     if shot_list[i + j].main_clip:
    #                         st.video(shot_list[i + j].main_clip.location)
    #                     else:
    #                         st.error("No video found for this frame.")
    #                 with st.expander(f'Frame #{display_number}', True):    
    #                     timeline_view_buttons(i, j, shot_list, time_setter_toggle, replace_image_widget_toggle, duration_setter_toggle, copy_frame_toggle, move_frames_toggle, delete_frames_toggle, change_position_toggle, project_uuid)


