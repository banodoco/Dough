import time
from typing import List
import streamlit as st
from ui_components.constants import WorkflowStageType
from ui_components.methods.file_methods import generate_pil_image

from ui_components.models import InternalFrameTimingObject, InternalShotObject
from ui_components.widgets.add_key_frame_element import add_key_frame,add_key_frame_section
from ui_components.widgets.frame_movement_widgets import change_frame_shot, delete_frame_button, jump_to_single_frame_view_button, move_frame_back_button, move_frame_forward_button, replace_image_widget
from utils.data_repo.data_repo import DataRepo
from utils import st_memory

def shot_keyframe_element(shot_uuid, items_per_row, **kwargs):
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    
    if "open_shot" not in st.session_state:
        st.session_state["open_shot"] = None

    # st.markdown(f"### {shot.name}", expanded=True)

    timing_list: List[InternalFrameTimingObject] = shot.timing_list
    
    with st.expander(f"_-_-_-_", expanded=True):
        # st.info(f"##### {shot.name}")    
        header_col_1, header_col_0, header_col_2, header_col_3 = st.columns([1.5, 1,1,4])

        with header_col_0:
            if st.session_state["open_shot"] != shot.uuid:
                if st.toggle("Open shot", key=f"shot_{shot.uuid}"):
                    st.session_state["open_shot"] = shot.uuid
                    st.rerun()
            else:
                if not st.toggle("Open shot", key=f"close_shot_{shot.uuid}", value=True):
                    st.session_state["open_shot"] = None
                    st.rerun()

        if st.session_state["open_shot"] == shot.uuid:
                    
            with header_col_1:        
                update_shot_name(shot, data_repo)
                    
            with header_col_2:
                update_shot_duration(shot, data_repo)

            with header_col_3:
                col2, col3, col4 = st.columns(3)
    
                with col2:
                    delete_frames_toggle = st_memory.toggle("Delete Frames", value=True, key="delete_frames_toggle")
                    copy_frame_toggle = st_memory.toggle("Copy Frame", value=True, key="copy_frame_toggle")
                with col3:
                    move_frames_toggle = st_memory.toggle("Move Frames", value=True, key="move_frames_toggle")
                    replace_image_widget_toggle = st_memory.toggle("Replace Image", value=False, key="replace_image_widget_toggle")
                    
                with col4:
                    change_shot_toggle = st_memory.toggle("Change Shot", value=False, key="change_shot_toggle")
                
            st.markdown("***")
        else:
            with header_col_1:
                st.info(f"**{shot.name}**")    


        grid = st.columns(items_per_row)
        # if timing_list and len(timing_list):
        for idx in range(len(timing_list) + 1):
            with grid[idx%items_per_row]:
                if idx == len(timing_list):
                    if st.session_state["open_shot"] == shot.uuid:
                        st.info("**Add new frame to shot**")
                        selected_image, inherit_styling_settings, _  =  add_key_frame_section(shot_uuid, False)                           
                        if st.button(f"Add key frame",type="primary",use_container_width=True):
                            add_key_frame(selected_image, inherit_styling_settings, shot_uuid)
                            st.rerun()                         
                else:
                    timing = timing_list[idx]
                    if timing.primary_image and timing.primary_image.location:
                        st.image(timing.primary_image.location, use_column_width=True)
                        if st.session_state["open_shot"] == shot.uuid:
                            timeline_view_buttons(idx, shot_uuid, replace_image_widget_toggle, copy_frame_toggle, move_frames_toggle,delete_frames_toggle, change_shot_toggle)
                    else:
                        
                        st.warning("No primary image present")
        # else:
          #  st.warning("No keyframes present")

        st.markdown("***")     


        if st.session_state["open_shot"] == shot.uuid:
            bottom1, bottom2, bottom3 = st.columns([1,2,1])
            with bottom1:            
                confirm_delete = st.checkbox("I know that this will delete all the frames and videos within")
                help = "Check the box above to enable the delete bottom." if confirm_delete else ""
                if st.button("Delete shot", disabled=(not confirm_delete), help=help, key=shot_uuid):
                    data_repo.delete_shot(shot_uuid)
                    st.success("Done!")
                    time.sleep(0.3)
                    st.rerun()
            
            with bottom3:
                if st.button("Move shot up", key=f'shot_up_movement_{shot.uuid}'):
                    if shot.shot_idx > 0:
                        data_repo.update_shot(shot_uuid, shot_idx=shot.shot_idx-1)
                    else:
                        st.error("This is the first shot")
                        time.sleep(0.3)
                    st.rerun()
                if st.button("Move shot down", key=f'shot_down_movement_{shot.uuid}'):
                    shot_list = data_repo.get_shot_list(shot.project.uuid)
                    if shot.shot_idx < len(shot_list):
                        data_repo.update_shot(shot_uuid, shot_idx=shot.shot_idx+1)
                    else:
                        st.error("This is the last shot")
                        time.sleep(0.3)
                    st.rerun()
                
def update_shot_name(shot, data_repo):
    name = st.text_area("Name:", value=shot.name, max_chars=40, height=15)
    if name != shot.name:
        data_repo.update_shot(shot.uuid, name=name)
        st.success("Success")
        time.sleep(0.3)
        st.rerun()

def update_shot_duration(shot, data_repo):
    duration = st.number_input("Duration:", value=shot.duration)
    if duration != shot.duration:
        data_repo.update_shot(shot.uuid, duration=duration)
        st.success("Success")
        time.sleep(0.3)
        st.rerun()

def shot_video_element(shot_uuid):
    data_repo = DataRepo()
    
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)

    st.markdown(f"### {shot.name}")
    if shot.main_clip and shot.main_clip.location:
        st.video(shot.main_clip.location)
    else:
        st.warning('''No video present''')

    if st.button(f"Jump to {shot.name}", key=f"btn_{shot_uuid}"):
        st.success("Coming soon")
    

        

def timeline_view_buttons(idx, shot_uuid, replace_image_widget_toggle, copy_frame_toggle, move_frames_toggle, delete_frames_toggle, change_shot_toggle):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = shot.timing_list

    if replace_image_widget_toggle:
        replace_image_widget(timing_list[idx].uuid, stage=WorkflowStageType.STYLED.value, options=["Uploaded Frame"])
    
    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
    
    if move_frames_toggle:
        with btn1:                                            
            move_frame_back_button(timing_list[idx].uuid, "side-to-side")
        with btn2:   
            move_frame_forward_button(timing_list[idx].uuid, "side-to-side")
    
    if copy_frame_toggle:
        with btn3:
            if st.button("🔁", key=f"copy_frame_{timing_list[idx].uuid}", use_container_width=True):
                pil_image = generate_pil_image(timing_list[idx].primary_image.location)
                add_key_frame(pil_image, False, st.session_state['shot_uuid'], timing_list[idx].aux_frame_index+1, refresh_state=False)
                st.rerun()

    if delete_frames_toggle:
        with btn4:
            delete_frame_button(timing_list[idx].uuid)
    
    if change_shot_toggle:
        change_frame_shot(timing_list[idx].uuid, "side-to-side")
    
    jump_to_single_frame_view_button(idx + 1, timing_list, 'timeline_btn_'+str(timing_list[idx].uuid))        

