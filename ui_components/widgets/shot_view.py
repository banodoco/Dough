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
        header_col_0, header_col_1, header_col_2, header_col_3 = st.columns([1.75, 1,1,3])

        if st.session_state["open_shot"] == shot.uuid:
            with header_col_0:
                update_shot_name(shot.uuid)                
                if not st.toggle("Expand", key=f"close_shot_{shot.uuid}", value=True):
                    st.session_state["open_shot"] = None
                    st.rerun()
                    
            with header_col_1:   
                update_shot_duration(shot.uuid)

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
        else:
            with header_col_0:      
                st.info(f"##### {shot.name}")
                if st.toggle("Expand", key=f"shot_{shot.uuid}"):
                    st.session_state["open_shot"] = shot.uuid
                    st.rerun()

            with header_col_1:                
                st.info(f"**{shot.duration} secs**")

        st.markdown("***")

        for i in range(0, len(timing_list) + 1, items_per_row):
            with st.container():
                grid = st.columns(items_per_row)
                for j in range(items_per_row):
                    idx = i + j
                    if idx <= len(timing_list):
                        with grid[j]:
                            if idx == len(timing_list):
                                if st.session_state["open_shot"] == shot.uuid:
                                    st.info("**Add new frame to shot**")
                                    selected_image, inherit_styling_settings =  add_key_frame_section(shot_uuid, False)                           
                                    if st.button(f"Add key frame",type="primary",use_container_width=True):
                                        add_key_frame(selected_image, "No", shot_uuid)
                                        st.rerun()                         
                            else:
                                timing = timing_list[idx]
                                if timing.primary_image and timing.primary_image.location:
                                    st.image(timing.primary_image.location, use_column_width=True)
                                else:                        
                                    st.warning("No primary image present")        
                                if st.session_state["open_shot"] == shot.uuid:
                                    timeline_view_buttons(idx, shot_uuid, replace_image_widget_toggle, copy_frame_toggle, move_frames_toggle,delete_frames_toggle, change_shot_toggle)
                if (i < len(timing_list) - 1) or (st.session_state["open_shot"] == shot.uuid) or (len(timing_list) % items_per_row != 0 and st.session_state["open_shot"] != shot.uuid):
                    st.markdown("***")
        # st.markdown("***")

        if st.session_state["open_shot"] == shot.uuid:
            st.markdown("##### Admin stuff:")
            bottom1, bottom2, bottom3, _ = st.columns([1,1,1,3])
            with bottom1:    
                st.error("Delete:")
                delete_shot_button(shot.uuid)
                                
            with bottom2:
                st.warning("Duplicate:")
                duplicate_shot_button(shot.uuid)       
            
            with bottom3:
                st.info("Move:")
                move1, move2 = st.columns(2)
                with move1:
                    if st.button("⬆️", key=f'shot_up_movement_{shot.uuid}', help="Move shot up", use_container_width=True):
                        if shot.shot_idx > 0:
                            data_repo.update_shot(uuid=shot_uuid, shot_idx=shot.shot_idx-1)
                        else:
                            st.error("This is the first shot")
                            time.sleep(0.3)
                        st.rerun()
                with move2:
                    if st.button("⬇️", key=f'shot_down_movement_{shot.uuid}', help="Move shot down", use_container_width=True):
                        shot_list = data_repo.get_shot_list(shot.project.uuid)
                        if shot.shot_idx < len(shot_list):
                            data_repo.update_shot(uuid=shot_uuid, shot_idx=shot.shot_idx+1)
                        else:
                            st.error("This is the last shot")
                            time.sleep(0.3)
                        st.rerun()

def duplicate_shot_button(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    if st.button("Duplicate shot", key=f"duplicate_btn_{shot.uuid}", help="This will duplicate this shot.", use_container_width=True):
        data_repo.duplicate_shot(shot.uuid)
        st.success("Shot duplicated successfully")
        time.sleep(0.3)
        st.rerun()

def delete_shot_button(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    confirm_delete = st.checkbox("I know that this will delete all the frames and videos within")    
    help_text = "Check the box above to enable the delete button." if not confirm_delete else "This will this shot and all the frames and videos within."
    if st.button("Delete shot", disabled=(not confirm_delete), help=help_text, key=shot.uuid, use_container_width=True):
        data_repo.delete_shot(shot.uuid)
        st.success("Shot deleted successfully")
        time.sleep(0.3)
        st.rerun()

def update_shot_name(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    name = st.text_input("Name:", value=shot.name, max_chars=25)
    if name != shot.name:
        data_repo.update_shot(uuid=shot.uuid, name=name)
        st.success("Name updated!")
        time.sleep(0.3)
        st.rerun()

def update_shot_duration(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    duration = st.number_input("Duration:", value=shot.duration)
    if duration != shot.duration:
        data_repo.update_shot(uuid=shot.uuid, duration=duration)
        st.success("Duration updated!")
        time.sleep(0.3)
        st.rerun()

def shot_video_element(shot_uuid):
    data_repo = DataRepo()
    
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    
    st.info(f"##### {shot.name}")
    if shot.main_clip and shot.main_clip.location:
        st.video(shot.main_clip.location)
    else:
        st.warning('''No video present''')

    if st.button(f"Jump to shot", key=f"btn_{shot_uuid}", use_container_width=True):
        st.session_state["shot_uuid"] = shot.uuid
        st.session_state["frame_styling_view_type_manual_select"] = 2
        st.rerun()
    

        

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

