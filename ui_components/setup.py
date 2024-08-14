import streamlit as st
import os
from moviepy.editor import *
from shared.constants import SERVER, AppSubPage, CreativeProcessPage, ServerType
from ui_components.methods.common_methods import check_project_meta_data
from ui_components.components.app_settings_page import app_settings_page
from ui_components.components.timeline_view_page import timeline_view_page
from ui_components.components.inspiraton_engine_page import inspiration_engine_page
from ui_components.components.adjust_shot_page import adjust_shot_page
from ui_components.components.animate_shot_page import animate_shot_page
from ui_components.components.upscaling_page import upscaling_page

from ui_components.components.new_project_page import new_project_page
from ui_components.components.project_settings_page import project_settings_page
from streamlit_option_menu import option_menu
from utils.common_utils import set_default_values
from utils.state_refresh import refresh_app

from ui_components.models import InternalAppSettingObject
from utils.common_utils import (
    create_working_assets,
    get_current_user,
    get_current_user_uuid,
    reset_project_state,
)
from utils import st_memory

from utils.data_repo.data_repo import DataRepo


def setup_app_ui():
    data_repo = DataRepo()

    app_settings: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()

    if SERVER != ServerType.DEVELOPMENT.value:
        current_user = get_current_user()
        user_credits = current_user.total_credits if (current_user and current_user.total_credits > 0) else 0
        if user_credits < 0.5:
            st.error(f"You have {user_credits} credits left - please go to App Settings to add more credits")

    hide_img = """
        <style>
        button[title="View fullscreen"]{
            display: none;}
        </style>
        """
    st.markdown(hide_img, unsafe_allow_html=True)

    with st.sidebar:
        h1, h2 = st.columns([1, 2])
        with h1:
            # st.markdown("# :red[ba]:green[no]:orange[do]:blue[co]")
            st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")
            st.caption("by Banodoco")

        sections = ["Open Project", "App Settings", "New Project"]
        with h2:
            st.write("")
            st.session_state["section"] = st_memory.menu(
                "",
                sections,
                icons=["cog", "cog", "cog"],
                menu_icon="ellipsis-v",
                default_index=st.session_state.get("section_index", 0),
                key="app_settings",
                orientation="horizontal",
                styles={
                    "nav-link": {"font-size": "12px", "margin": "0px", "--hover-color": "#5c5c5c"},
                    "nav-link-selected": {"background-color": "grey"},
                },
            )

    project_list = data_repo.get_all_project_list(user_id=get_current_user_uuid())

    if st.session_state["section"] == "Open Project":
        if "index_of_project_name" not in st.session_state:
            if app_settings.previous_project:
                st.session_state["project_uuid"] = app_settings.previous_project
                st.session_state["index_of_project_name"] = next(
                    (
                        i
                        for i, p in enumerate(project_list)
                        if str(p.uuid) == str(app_settings.previous_project)
                    ),
                    None,
                )

                # if index is not found (project deleted or data mismatch) assigning the first project as default
                if not st.session_state["index_of_project_name"]:
                    st.session_state["index_of_project_name"] = 0
                    st.session_state["project_uuid"] = project_list[0].uuid
            else:
                st.session_state["index_of_project_name"] = 0

        selected_project_name = st.sidebar.selectbox(
            "Project:", [p.name for p in project_list], index=st.session_state["index_of_project_name"]
        )

        selected_index = next(i for i, p in enumerate(project_list) if p.name == selected_project_name)
        project_changed = False
        if (
            "project_uuid" in st.session_state
            and st.session_state["project_uuid"] != project_list[selected_index].uuid
        ):
            project_changed = True

        if project_changed:
            reset_project_state()

        st.session_state["project_uuid"] = project_list[selected_index].uuid
        data_repo.update_app_setting(previous_project_id=st.session_state["project_uuid"])
        if "maintain_state" not in st.session_state:
            st.session_state["maintain_state"] = False

        if not st.session_state["maintain_state"]:
            check_project_meta_data(st.session_state["project_uuid"])

        if "shot_uuid" not in st.session_state:
            shot_list = data_repo.get_shot_list(st.session_state["project_uuid"])
            st.session_state["shot_uuid"] = shot_list[0].uuid

        if "last_shot_number" not in st.session_state:
            st.session_state["last_shot_number"] = 0

        # print uuids of shots
        if "current_frame_index" not in st.session_state:
            st.session_state["current_frame_index"] = 1

        if st.session_state["index_of_project_name"] != next(
            (i for i, p in enumerate(project_list) if p.uuid == st.session_state["project_uuid"]), None
        ):
            st.success("Project changed!")
            st.session_state["index_of_project_name"] = next(
                (i for i, p in enumerate(project_list) if p.uuid == st.session_state["project_uuid"]), None
            )
            data_repo.update_app_setting(previous_project=st.session_state["project_uuid"])

            refresh_app()

        if st.session_state["project_uuid"] == "":
            st.info("No projects found - create one in the 'New Project' section")

        else:
            if not os.path.exists("videos/" + st.session_state["project_uuid"] + "/assets"):
                create_working_assets(st.session_state["project_uuid"])

            if "index_of_section" not in st.session_state:
                st.session_state["index_of_section"] = 0
                st.session_state["index_of_page"] = 0

            with st.sidebar:
                main_view_types = ["Creative Process", "Project Settings"]
                st.session_state["main_view_type"] = st_memory.menu(
                    None,
                    main_view_types,
                    icons=["search-heart", "tools", "play-circle", "stopwatch"],
                    menu_icon="cast",
                    default_index=0,
                    key="main_view_type_name",
                    orientation="horizontal",
                    styles={
                        "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#014001"},
                        "nav-link-selected": {"background-color": "green"},
                    },
                )

            if st.session_state["main_view_type"] == "Creative Process":
                set_default_values(st.session_state["shot_uuid"])

                with st.sidebar:
                    creative_process_pages = CreativeProcessPage.value_list()

                    # mapping subpages to their main page
                    subpage_page_map = {
                        # timeline
                        AppSubPage.SHOTS.value: CreativeProcessPage.SHOTS.value,
                        AppSubPage.INSPIRATION_ENGINE.value: CreativeProcessPage.INSPIRATION_ENGINE.value,
                        # adjust shot
                        AppSubPage.ADJUST_SHOT.value: CreativeProcessPage.ADJUST_SHOT.value,
                        AppSubPage.KEYFRAME.value: CreativeProcessPage.ADJUST_SHOT.value,
                        # animate shot
                        AppSubPage.ANIMATE_SHOT.value: CreativeProcessPage.ANIMATE_SHOT.value,
                        AppSubPage.UPSCALING.value: CreativeProcessPage.UPSCALING.value,
                    }

                    if "current_subpage" not in st.session_state:
                        st.session_state["current_subpage"] = st.session_state["prev_subpage"] = (
                            AppSubPage.SHOTS.value
                        )

                    if "page" not in st.session_state:
                        st.session_state["page"] = st.session_state["prev_page"] = (
                            CreativeProcessPage.SHOTS.value
                        )

                    if "selected_page_idx" not in st.session_state:
                        st.session_state["selected_page_idx"] = creative_process_pages.index(
                            st.session_state["page"]
                        )

                    # checking if the subpage has changed
                    if (
                        "prev_subpage" in st.session_state
                        and st.session_state["prev_subpage"] != st.session_state["current_subpage"]
                    ):
                        main_page = subpage_page_map[st.session_state["current_subpage"]]
                        st.session_state["prev_subpage"] = st.session_state["current_subpage"]
                        st.session_state["page"] = st.session_state["prev_page"] = main_page
                        st.session_state["selected_page_idx"] = creative_process_pages.index(
                            st.session_state["page"]
                        )
                        refresh_app()

                    # 'page' state randomly resets therefore binding it to 'selected_page_idx'
                    st.session_state["page"] = creative_process_pages[st.session_state["selected_page_idx"]]

                    def change_page(key):
                        page = st.session_state[key]
                        st.session_state["page"] = page

                        for k, v in subpage_page_map.items():
                            if v == st.session_state["page"]:
                                st.session_state["current_subpage"] = st.session_state["prev_subpage"] = k
                                st.session_state["prev_page"] = st.session_state["page"]
                                st.session_state["selected_page_idx"] = creative_process_pages.index(
                                    st.session_state["page"]
                                )
                                break

                    _ = option_menu(
                        None,
                        creative_process_pages,
                        icons=["bookshelf", "lightning-charge", "crop", "film", "aspect-ratio"],
                        menu_icon="cast",
                        orientation="vertical",
                        key="page_opt_menu",
                        styles={
                            "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#bd3737"},
                            "nav-link-selected": {"background-color": "#ff4b4b"},
                        },
                        manual_select=st.session_state["selected_page_idx"],
                        default_index=st.session_state["selected_page_idx"],
                        on_change=change_page,
                    )

                    # TODO: this is a hacky fix (related to jump_to_single_frame_view_button)
                    if st.session_state["page"] != CreativeProcessPage.ADJUST_SHOT.value:
                        st.session_state["current_frame_sidebar_selector"] = 0

                # NOTE: not is use
                # if st.session_state["page"] == "Explore":
                #     explorer_page(st.session_state["project_uuid"])
                # elif st.session_state["page"] == "Shortlist":
                #     shortlist_page(st.session_state["project_uuid"])

                if st.session_state["page"] == CreativeProcessPage.SHOTS.value:
                    timeline_view_page(st.session_state["shot_uuid"], h2)

                elif st.session_state["page"] == CreativeProcessPage.INSPIRATION_ENGINE.value:
                    inspiration_engine_page(st.session_state["shot_uuid"], h2)

                elif st.session_state["page"] == CreativeProcessPage.ADJUST_SHOT.value:
                    adjust_shot_page(st.session_state["shot_uuid"], h2)

                elif st.session_state["page"] == CreativeProcessPage.ANIMATE_SHOT.value:
                    animate_shot_page(st.session_state["shot_uuid"], h2)

                elif st.session_state["page"] == CreativeProcessPage.UPSCALING.value:
                    upscaling_page(st.session_state["project_uuid"])

            elif st.session_state["main_view_type"] == "Project Settings":
                project_settings_page(st.session_state["project_uuid"])

    elif st.session_state["section"] == "App Settings":
        app_settings_page()

    elif st.session_state["section"] == "New Project":
        new_project_page()

    else:
        st.info("You haven't added any prompts yet. Add an image to get started.")

    with st.sidebar:
        st.caption(
            "Want to join [a community](https://discord.gg/acg8aNBTxd) that's pushing AI to its technical and artistic limits or help build a [next-generation artistic tool](https://banodoco.ai/Plan) and economic engine for the open source AI art ecosystem?"
        )
