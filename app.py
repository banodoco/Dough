import time
import streamlit as st
from moviepy.editor import *
import subprocess
import os
import django
import sentry_sdk

from utils.data_repo.api_repo import APIRepo

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
django.setup()
st.session_state["django_init"] = True

from shared.constants import (
    GPU_INFERENCE_ENABLED_KEY,
    HOSTED_BACKGROUND_RUNNER_MODE,
    OFFLINE_MODE,
    SERVER,
    ServerType,
    ConfigManager,
)

from shared.logging.logging import AppLogger
from ui_components.components.user_login_page import user_login_ui
from ui_components.models import InternalUserObject
from utils.app_update_utils import apply_updates, check_and_pull_changes, load_save_checkpoint
from utils.common_decorators import update_refresh_lock
from utils.common_utils import is_process_active, refresh_process_active
from utils.state_refresh import refresh_app

from utils.constants import (
    REFRESH_PROCESS_PORT,
    RUNNER_PROCESS_NAME,
)
from streamlit_server_state import server_state_lock
from utils.refresh_target import SAVE_STATE
from banodoco_settings import project_init
from utils.data_repo.data_repo import DataRepo


config_manager = ConfigManager()
RUNNER_PROCESS_PORT = config_manager.get("runner_process_port")


def start_runner():
    if SERVER != ServerType.DEVELOPMENT.value and HOSTED_BACKGROUND_RUNNER_MODE in [False, "False"]:
        return

    with server_state_lock["runner"]:
        app_logger = AppLogger()

        if not is_process_active(RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT):
            app_logger.info("Starting runner")
            python_executable = sys.executable
            _ = subprocess.Popen([python_executable, "banodoco_runner.py"])
            max_retries = 6
            while not is_process_active(RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT) and max_retries:
                time.sleep(0.1)
                max_retries -= 1

            # refreshing the app if the runner port has changed
            old_port = RUNNER_PROCESS_PORT
            new_port = config_manager.get("runner_process_port", fresh_pull=True)
            if old_port != new_port:
                st.rerun()
        else:
            # app_logger.debug("Runner already running")
            pass


def start_project_refresh():
    if SERVER != ServerType.DEVELOPMENT.value:
        return

    with server_state_lock["refresh_app"]:
        app_logger = AppLogger()

        if not refresh_process_active(REFRESH_PROCESS_PORT):
            python_executable = sys.executable
            _ = subprocess.Popen([python_executable, "auto_refresh.py"])
            max_retries = 6
            while not refresh_process_active(REFRESH_PROCESS_PORT) and max_retries:
                time.sleep(1)
                max_retries -= 1
            app_logger.info("Auto refresh enabled")
        else:
            # app_logger.debug("refresh process already running")
            pass


def main():
    st.set_page_config(page_title="Dough", layout="wide", page_icon="🎨")
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    update_refresh_lock(False)

    # if it's the first time,
    if "first_load" not in st.session_state:
        if not is_process_active(RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT):
            if not load_save_checkpoint():
                check_and_pull_changes()  # enabling auto updates only for local version
            else:
                apply_updates()
                refresh_app()
        st.session_state["first_load"] = True

    start_runner()
    start_project_refresh()
    project_init()

    from ui_components.setup import setup_app_ui
    from ui_components.components.welcome_page import welcome_page

    data_repo = DataRepo()
    api_repo = APIRepo()
    config_manager = ConfigManager()
    gpu_enabled = config_manager.get(GPU_INFERENCE_ENABLED_KEY, False)

    app_setting = data_repo.get_app_setting_from_uuid()
    if app_setting.welcome_state == 2:
        # api/online inference mode
        if not gpu_enabled:
            if not api_repo.is_user_logged_in():
                # user not logged in
                user_login_ui()
            else:
                setup_app_ui()
        else:
            # gpu/offline inference mode
            setup_app_ui()
    else:
        welcome_page()

    st.session_state["maintain_state"] = False


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise e
