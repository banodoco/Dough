import time
import streamlit as st
import os
import math
from moviepy.editor import *
from shared.constants import SERVER, ServerType

from ui_components.components.app_settings_page import app_settings_page
from ui_components.components.custom_models_page import custom_models_page
from ui_components.components.frame_styling_page import frame_styling_page
from ui_components.components.new_project_page import new_project_page
from ui_components.components.project_settings_page import project_settings_page
from ui_components.components.video_rendering_page import video_rendering_page
from streamlit_option_menu import option_menu
from ui_components.constants import CreativeProcessType
from ui_components.models import InternalAppSettingObject
from utils.common_utils import create_working_assets, get_current_user, get_current_user_uuid, reset_project_state
from utils import st_memory

from utils.data_repo.data_repo import DataRepo

# TODO: CORRECT-CODE


def setup_app_ui():
    data_repo = DataRepo()

    app_settings: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()

    if SERVER != ServerType.DEVELOPMENT.value:
        current_user = get_current_user()
        user_credits = current_user.total_credits if (current_user and current_user.total_credits > 0) else 0
        if user_credits < 0.5:
            st.error(f"You have {user_credits} credits left - please go to App Settings to add more credits")

    with st.sidebar:
        h1, h2 = st.columns([1, 3])
        with h1:
            st.markdown("# :red[ba]:green[no]:orange[do]:blue[co]")

        sections = ["Open Project", "App Settings", "New Project"]
        with h2:
            st.write("")
            st.session_state["section"] = st_memory.menu(
                "",
                sections,
                icons=['cog', 'cog', 'cog'],
                menu_icon="ellipsis-v",
                default_index=st.session_state.get('section_index', 0),
                key="app_settings",
                orientation="horizontal",
                styles={
                    "nav-link": {"font-size": "12px", "margin": "0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "grey"}
                }
            )
    
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
        project_changed = False
        if 'project_uuid' in st.session_state and st.session_state['project_uuid'] != project_list[selected_index].uuid:
            project_changed = True
        
        if project_changed:
            reset_project_state()
        
        st.session_state["project_uuid"] = project_list[selected_index].uuid

        if "current_frame_index" not in st.session_state:
            st.session_state['current_frame_index'] = 1

        if st.session_state["index_of_project_name"] != next((i for i, p in enumerate(
                project_list) if p.uuid == st.session_state["project_uuid"]), None):
            st.write("Project changed")
            st.session_state["index_of_project_name"] = next((i for i, p in enumerate(
                project_list) if p.uuid == st.session_state["project_uuid"]), None)
            data_repo.update_app_setting(previous_project=st.session_state["project_uuid"])
            st.rerun()

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
                st.session_state['main_view_type'] = st_memory.menu(None, main_view_types, icons=['search-heart', 'tools', "play-circle", 'stopwatch'], menu_icon="cast", default_index=0, key="main_view_type_name", orientation="horizontal", styles={
                                                                    "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "red"}})

            mainheader1, mainheader2 = st.columns([3, 2])
            # with mainheader1:
            # st.header(st.session_state["page"])

        


            if st.session_state["main_view_type"] == "Creative Process":

                with st.sidebar:

                    view_types = ["Explorer","Timeline","Individual"]

                    if 'frame_styling_view_type_index' not in st.session_state:
                        st.session_state['frame_styling_view_type_index'] = 0
                        st.session_state['frame_styling_view_type'] = "Explorer"
                        st.session_state['change_view_type'] = False

                    if 'change_view_type' not in st.session_state:
                        st.session_state['change_view_type'] = False

                    if st.session_state['change_view_type'] == True:
                        st.session_state['frame_styling_view_type_index'] = view_types.index(
                            st.session_state['frame_styling_view_type'])
                    else:
                        st.session_state['frame_styling_view_type_index'] = None


                    # Option menu
                    st.session_state['frame_styling_view_type'] = option_menu(
                        None,
                        view_types,
                        icons=['compass', 'bookshelf','aspect-ratio', "hourglass", 'stopwatch'],
                        menu_icon="cast",
                        orientation="horizontal",
                        key="section-selecto1r",
                        styles={"nav-link": {"font-size": "15px", "margin":"0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "green"}},
                        manual_select=st.session_state['frame_styling_view_type_index']                        
                    )

                    if st.session_state['frame_styling_view_type'] != "Explorer":
                        pages = CreativeProcessType.value_list()
                    else:
                        pages = ["Key Frames"]
                    
                    if 'page' not in st.session_state:
                        st.session_state["page"] = pages[0]
                        st.session_state["manual_select"] = None

                    if st.session_state["page"] not in pages:
                        st.session_state["page"] = pages[0]
                        st.session_state["manual_select"] = None

                    st.session_state['page'] = option_menu(None, pages, icons=['palette', 'camera-reels', "hourglass", 'stopwatch'], menu_icon="cast", orientation="horizontal", key="secti2on_selector", styles={
                                                            "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "orange"}}, manual_select=st.session_state["manual_select"])

                    # TODO: CORRECT-CODE

                                

                frame_styling_page(
                    mainheader2, st.session_state["project_uuid"])

            elif st.session_state["main_view_type"] == "Tools & Settings":

                with st.sidebar:
                    tool_pages = ["Query Logger",  "Custom Models", "Project Settings"]

                    if st.session_state["page"] not in tool_pages:
                        st.session_state["page"] = tool_pages[0]
                        st.session_state["manual_select"] = None

                    st.session_state['page'] = option_menu(None, tool_pages, icons=['pencil', 'palette', "hourglass", 'stopwatch'], menu_icon="cast", orientation="horizontal", key="secti2on_selector", styles={
                                                            "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "green"}}, manual_select=st.session_state["manual_select"])
                if st.session_state["page"] == "Query Logger":
                    st.info("Query Logger will appear here.")
                if st.session_state["page"] == "Custom Models":
                    custom_models_page(st.session_state["project_uuid"])                
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
