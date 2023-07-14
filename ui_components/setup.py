import time
import streamlit as st
import os
from moviepy.editor import *
from ui_components.common_methods import create_working_assets

from ui_components.components.app_settings_page import app_settings_page
from ui_components.components.batch_action_page import batch_action_page
from ui_components.components.custom_models_page import custom_models_page
from ui_components.components.frame_styling_page import frame_styling_page
from ui_components.components.key_frame_selection import key_frame_selection_page
from ui_components.components.new_project_page import new_project_page
from ui_components.components.project_settings_page import project_settings_page
from ui_components.components.video_rendering_page import video_rendering_page
from ui_components.components.welcome_page import welcome_page
# from ui_components.components.motion_page import guidance_page
# from ui_components.components.motion_page import styling_page
# from ui_components.components.motion_page import motion_page
from streamlit_option_menu import option_menu
from ui_components.models import InternalAppSettingObject
from utils.common_methods import get_current_user_uuid
import utils.local_storage.local_storage as local_storage

from utils.data_repo.data_repo import DataRepo

# TODO: CORRECT-CODE


def setup_app_ui():
    data_repo = DataRepo()

    app_settings: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()

    with st.sidebar:

        h1, h2 = st.columns([1, 3])

        with h1:
            st.markdown("#:red[Ba]:green[no]:orange[do]:blue[co]")

        sections = ["Open Project", "App Settings", "New Project"]

        if "section" not in st.session_state:
            st.session_state["section"] = sections[0]
            st.session_state['change_section'] = False

        if st.session_state['change_section'] == True:
            st.session_state['section_index'] = sections.index(
                st.session_state["section"])
        else:
            st.session_state['section_index'] = None

        with h2:
            st.write("")
            st.session_state["section"] = option_menu(
                "",
                sections,
                icons=['cog', 'cog', 'cog'],
                menu_icon="ellipsis-v",
                orientation="horizontal",
                key="app_settings",
                styles={
                    "nav-link": {"font-size": "12px", "margin": "0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "grey"}
                },
                manual_select=st.session_state['section_index']
            )

        if st.session_state['change_section'] == True:
            st.session_state['change_section'] = False

    if int(st.session_state["welcome_state"]) in [0, 1, 2, 3, 4] and st.session_state["online"] == False:
        welcome_page()
    else:
        project_list = data_repo.get_all_project_list(
            user_id=get_current_user_uuid())

        if st.session_state["section"] == "Open Project":

            if "index_of_project_name" not in st.session_state:
                if app_settings.previous_project:
                    st.session_state["project_uuid"] = app_settings.previous_project
                    st.session_state["index_of_project_name"] = next((i for i, p in enumerate(
                        project_list) if p.uuid == app_settings.previous_project), None)
                    
                    # if index is not found (project deleted or data mismatch) assigning the first project as default
                    if not st.session_state["index_of_project_name"]:
                        st.session_state["index_of_project_name"] = 0
                        st.session_state["project_uuid"] = project_list[0].uuid
                else:
                    st.session_state["index_of_project_name"] = 0

            selected_project_name = st.sidebar.selectbox("Select which project you'd like to work on:", [
                                                                p.name for p in project_list], index=st.session_state["index_of_project_name"])
        
            selected_index = next(i for i, p in enumerate(project_list) if p.name == selected_project_name)
            st.session_state["project_uuid"] = project_list[selected_index].uuid

            if "current_frame_index" not in st.session_state:
                st.session_state['current_frame_index'] = 0

            if st.session_state["index_of_project_name"] != next((i for i, p in enumerate(
                    project_list) if p.uuid == st.session_state["project_uuid"]), None):
                st.write("Project changed")
                st.session_state["index_of_project_name"] = next((i for i, p in enumerate(
                    project_list) if p.uuid == st.session_state["project_uuid"]), None)
                data_repo.update_app_setting(previous_project=st.session_state["project_uuid"])
                st.experimental_rerun()

            if st.session_state["project_uuid"] == "":
                st.info(
                    "No projects found - create one in the 'New Project' section")
            else:

                if not os.path.exists("videos/" + st.session_state["project_uuid"] + "/assets"):
                    create_working_assets(st.session_state["project_uuid"])

                if "index_of_section" not in st.session_state:
                    st.session_state["index_of_section"] = 0
                    st.session_state["index_of_page"] = 0

                with st.sidebar:
                    main_view_types = ["Creative Process", "Tools & Settings", "Video Rendering"]
                    st.session_state['main_view_type'] = option_menu(None, main_view_types, icons=['magic', 'tools', "play-circle", 'stopwatch'], menu_icon="cast", default_index=0, key="main_view_type_name", orientation="horizontal", styles={
                                                                     "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "red"}})

                mainheader1, mainheader2 = st.columns([3, 2])
                # with mainheader1:
                # st.header(st.session_state["page"])


                if st.session_state["main_view_type"] == "Creative Process":

                    with st.sidebar:

                        pages = ["Guidance", "Styling", "Motion"]

                        if 'page' not in st.session_state:
                            st.session_state["page"] = pages[0]
                            st.session_state["manual_select"] = None

                        if st.session_state["page"] not in pages:
                            st.session_state["page"] = pages[0]
                            st.session_state["manual_select"] = None

                        st.session_state['page'] = option_menu(None, pages, icons=['pencil', 'palette', "hourglass", 'stopwatch'], menu_icon="cast", orientation="horizontal", key="secti2on_selector", styles={
                                                               "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "orange"}}, manual_select=st.session_state["manual_select"])

                    frame_styling_page(
                        mainheader2, st.session_state["project_uuid"])

                elif st.session_state["main_view_type"] == "Tools & Settings":

                    with st.sidebar:

                        pages = ["Custom Models",
                                 "Batch Actions", "Project Settings"]

                        if st.session_state["page"] not in pages:
                            st.session_state["page"] = pages[0]
                            st.session_state["manual_select"] = None

                        st.session_state['page'] = option_menu(None, pages, icons=['pencil', 'palette', "hourglass", 'stopwatch'], menu_icon="cast", orientation="horizontal", key="secti2on_selector", styles={
                                                               "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "orange"}}, manual_select=st.session_state["manual_select"])

                    if st.session_state["page"] == "Custom Models":
                        custom_models_page(st.session_state["project_uuid"])
                    elif st.session_state["page"] == "Batch Actions":
                        batch_action_page(st.session_state["project_uuid"])
                    elif st.session_state["page"] == "Project Settings":
                        project_settings_page(st.session_state["project_uuid"])

                elif st.session_state["main_view_type"] == "Video Rendering":

                    video_rendering_page(
                        mainheader2, st.session_state["project_uuid"])

        elif st.session_state["section"] == "App Settings":
            app_settings_page()

        elif st.session_state["section"] == "New Project":

            new_project_page()

        else:
            st.info("You haven't added any prompts yet. Add an image to get started.")
