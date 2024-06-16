from typing import List
import streamlit as st
from utils.data_repo.data_repo import DataRepo
from utils import st_memory
from ui_components.methods.common_methods import add_new_shot


def frame_selector_widget(show_frame_selector=True):
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(st.session_state["shot_uuid"])
    shot = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
    shot_list = data_repo.get_shot_list(shot.project.uuid)
    len_timing_list = len(timing_list) if len(timing_list) > 0 else 1.0
    project_uuid = shot.project.uuid

    if "prev_shot_index" not in st.session_state:
        st.session_state["prev_shot_index"] = shot.shot_idx
    if "shot_name" not in st.session_state:
        st.session_state["shot_name"] = shot.name
    shot1, shot2 = st.columns([1, 1])
    with shot1:
        shot_names = [s.name for s in shot_list]
        shot_names.append("**Create New Shot**")
        current_shot_name = st.selectbox(
            "Shot:", shot_names, key="current_shot_sidebar_selector", index=shot_names.index(shot.name)
        )
        if current_shot_name != "**Create New Shot**":
            if current_shot_name != st.session_state["shot_name"]:
                st.session_state["shot_name"] = current_shot_name
                st.rerun()

        if current_shot_name == "**Create New Shot**":
            new_shot_name = st.text_input(
                "New shot name:", max_chars=40, key=f"shot_name_sidebar_{st.session_state['shot_name']}"
            )
            if st.button("Create new shot", key=f"create_new_shot_{st.session_state['shot_name']}"):
                new_shot = add_new_shot(project_uuid, name=new_shot_name)
                st.session_state["shot_name"] = new_shot_name
                st.session_state["shot_uuid"] = new_shot.uuid
                st.rerun()

    # find shot index based on shot name
    st.session_state["current_shot_index"] = shot_names.index(st.session_state["shot_name"]) + 1

    if st.session_state["shot_name"] != shot.name:
        st.session_state["shot_uuid"] = shot_list[shot_names.index(st.session_state["shot_name"])].uuid
        st.rerun()

    if not ("current_shot_index" in st.session_state and st.session_state["current_shot_index"]):
        st.session_state["current_shot_index"] = shot_names.index(st.session_state["shot_name"]) + 1
        update_current_shot_index(st.session_state["current_shot_index"])

    if st.session_state["page"] == "Key Frames":
        if st.session_state["current_frame_index"] > len_timing_list:
            update_current_frame_index(len_timing_list)

    elif st.session_state["page"] == "Shots":
        if st.session_state["current_shot_index"] > len(shot_list):
            update_current_shot_index(len(shot_list))

    if show_frame_selector:
        if len(timing_list):
            if "prev_frame_index" not in st.session_state or st.session_state["prev_frame_index"] > len(
                timing_list
            ):
                st.session_state["prev_frame_index"] = 1

            # Create a list of frames with a blank value as the first item
            frame_list = [""] + [f"{i+1}" for i in range(len(timing_list))]
            with shot2:
                frame_selection = st_memory.selectbox(
                    "Frame:", frame_list, key="current_frame_sidebar_selector"
                )

            # only trigger the frame number extraction and current frame index update if a non-empty value is selected
            if frame_selection != "":
                if st.button("Jump to shot view", use_container_width=True):
                    st.session_state["current_frame_sidebar_selector"] = 0
                    st.rerun()

                st.session_state["current_frame_index"] = int(frame_selection.split(" ")[-1])
                update_current_frame_index(st.session_state["current_frame_index"])
        else:
            frame_selection = ""
            with shot2:
                st.write("")
                st.error("No frames present")

        return frame_selection


def update_current_frame_index(index):
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(st.session_state["shot_uuid"])
    st.session_state["current_frame_uuid"] = timing_list[index - 1].uuid
    if st.session_state["prev_frame_index"] != index or st.session_state["current_frame_index"] != index:
        st.session_state["prev_frame_index"] = index
        st.session_state["current_frame_index"] = index
        st.session_state["current_frame_uuid"] = timing_list[index - 1].uuid
        st.session_state["reset_canvas"] = True
        st.session_state["frame_styling_view_type_index"] = 0
        st.session_state["frame_styling_view_type"] = "Generate View"

        st.rerun()


def update_current_shot_index(index):
    data_repo = DataRepo()
    shot_list = data_repo.get_shot_list(st.session_state["project_uuid"])
    st.session_state["shot_uuid"] = shot_list[index - 1].uuid
    if st.session_state["prev_shot_index"] != index or st.session_state["current_shot_index"] != index:
        st.session_state["current_shot_index"] = index
        st.session_state["prev_shot_index"] = index
        st.session_state["shot_uuid"] = shot_list[index - 1].uuid
        st.session_state["reset_canvas"] = True
        st.session_state["frame_styling_view_type_index"] = 0
        st.session_state["frame_styling_view_type"] = "Individual View"

        st.rerun()
