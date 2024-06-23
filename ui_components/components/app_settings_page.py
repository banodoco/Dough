import os
import subprocess
import sys
import time
from git import Repo
import streamlit as st
from dataclasses import dataclass, field
from shared.constants import SERVER, ServerType
from ui_components.methods.file_methods import delete_from_env, load_from_env, save_to_env
from utils.common_utils import get_current_user, get_toml_config
from ui_components.components.query_logger_page import query_logger_page

from utils.constants import TomlConfig
from utils.data_repo.data_repo import DataRepo
from utils.encryption import generate_file_hash
from utils.enum import ExtendedEnum
from ui_components.widgets.base_theme import BaseTheme as theme


class ErrorLevel(ExtendedEnum):
    SEVERE = "severe"  # breaking error
    MILD = "mild"  # can be breaking but not 100% sure
    WARNING = "warning"  # non-breaking


@dataclass
class ErrorPayload:
    error: str
    error_level: str
    resolution: str


def app_settings_page():
    data_repo = DataRepo()

    app_version = None
    with open("scripts/app_version.txt", "r") as file:
        app_version = file.read()

    st.markdown("#### App Settings" + ("" if not app_version else f" (v{app_version})"))
    st.markdown("***")

    if SERVER != ServerType.DEVELOPMENT.value:
        with st.expander("Purchase Credits", expanded=True):
            user_credits = get_current_user(invalidate_cache=True).total_credits
            user_credits = round(user_credits, 2) if user_credits else 0
            st.write(f"Total Credits: {user_credits}")
            c1, c2 = st.columns([1, 1])
            with c1:
                if "input_credits" not in st.session_state:
                    st.session_state["input_credits"] = 10

                credits = st.number_input(
                    "Credits (1 credit = $1)", value=st.session_state["input_credits"], step=10
                )
                if credits != st.session_state["input_credits"]:
                    st.session_state["input_credits"] = credits
                    st.rerun()

                if st.button("Generate payment link"):
                    if credits < 10:
                        st.error("Minimum credit value should be atleast 10")
                        time.sleep(0.7)
                        st.rerun()
                    else:
                        payment_link = data_repo.generate_payment_link(credits)
                        payment_link = f"""<a target='_self' href='{payment_link}'> PAYMENT LINK </a>"""
                        st.markdown(payment_link, unsafe_allow_html=True)

    # TODO: rn storing 'update_state' in replicate_username inside app_setting to bypass db changes, will change this later
    app_setting = data_repo.get_app_setting_from_uuid()
    update_enabled = (
        True
        if app_setting.replicate_username and app_setting.replicate_username in ["update", "bn"]
        else False
    )
    with st.expander("App Update", expanded=True):

        # st.info("We recommend auto-updating the app to get the latest features and bug fixes. However, if you'd like to update manually, you can turn this off and use './scripts/entrypoint.sh --update' when you're starting the app to update.")
        st.toggle(
            "Auto-update app upon restart",
            key="enable_app_update",
            value=update_enabled,
            on_change=update_toggle,
            help="This will update the app automatically when a new version is available.",
        )

    with st.expander("Custom ComfyUI Path", expanded=True):
        custom_comfy_input_component()

    with st.expander("API Keys", expanded=False):
        api_key_input_component()

    with st.expander("Health Check", expanded=False):
        health_check_component()

    with st.expander("Inference Logs", expanded=False):
        query_logger_page()


def custom_comfy_input_component():
    custom_comfy_key = "COMFY_MODELS_BASE_PATH"
    custom_comfy_path = load_from_env(custom_comfy_key)
    if not custom_comfy_path:
        st.info(
            """
            Please enter your custom ComfyUI path below. Dough will use the models present in your personal ComfyUI
            for inference. It will also download the new models in your personal ComfyUI instance. Please note that 
            Dough will maintain it's own copy of the nodes and packages to not cause any issues with the packages 
            that might already be installed on your personal ComfyUI.
            
            """
        )
    h1, _ = st.columns([1, 1])
    with h1:
        updated_path = st.text_input(
            "Custom ComfyUI path:",
            custom_comfy_path,
            key="app_settings_custom_comfy",
        )

    if st.button("Update", key="app_settings_update_path"):
        if updated_path != custom_comfy_path:
            if not updated_path:
                delete_from_env(custom_comfy_key)
            else:
                updated_path = os.path.join(updated_path, "")
                save_to_env(
                    key=custom_comfy_key,
                    value=updated_path,
                )
            theme.success_msg("Successfully updated the ComfyUI path")
            st.rerun()


def health_check_component():
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Start check"):
            res = run_health_check()
            err_list = []
            for err in res:
                err_list.append(
                    {"Error": err.error, "Severity": err.error_level, "Resolution": err.resolution}
                )

            if err_list and len(err_list):
                st.error("Health check has identified the errors listed below")
                st.table(data=err_list)
            else:
                st.success("No errors found")
    with c2:
        st.info(
            "This checks Dough for common issues like incorrect package installation, corrupt/missing files and invalid config"
        )


def run_health_check():
    """
    1. check model locations - severe error
    2. check model hash - mild error
    3. check node folder is properly downloaded - severe error
    4. check node hash - severe error
    5. check python version - severe error
    6. check python packages - mild error
    """

    def get_file_inside_comfy(filename):
        folder_path = "ComfyUI/"
        file_path = None
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file == filename:
                    file_path = os.path.join(root, file)

        file_hash = None if not file_path else generate_file_hash(file_path)
        return file_path, file_hash

    error_list = []

    st.write("Checking files... Don't refresh the page, this can take a couple of minutes")
    file_hash_dict = get_toml_config(TomlConfig.FILE_HASH.value)
    for file, val in file_hash_dict.items():
        filepath, file_hash = get_file_inside_comfy(file)  # this will use the BASE_COMFY_PATH
        filepath = "/".join(filepath.split("/")[1:-1]) + "/" if filepath else None
        if filepath and val["location"] and filepath != val["location"]:
            error_list.append(
                ErrorPayload(
                    error=f"{file} should be at {val['location']}",
                    error_level=ErrorLevel.SEVERE.value,
                    resolution=f"Move {file} to {val['location']}",
                )
            )

        if file_hash and file_hash not in val["hash"]:
            error_list.append(
                ErrorPayload(
                    error=f"{file} is either not downloaded properly or is a different variant",
                    error_level=ErrorLevel.MILD.value,
                    resolution=f"Try deleting {file}, Dough will auto-download it",
                )
            )

    st.write("Checking nodes...")
    node_commit_dict = get_toml_config(TomlConfig.NODE_VERSION.value)
    for node, val in node_commit_dict.items():
        node_path = os.path.join("ComfyUI", "custom_nodes", node)
        # TODO: replace this with a more robust hash check
        # checking if the node is properly installed
        if os.path.exists(node_path):
            if not len(os.listdir(node_path)) >= 3:
                error_list.appned(
                    ErrorPayload(
                        error=f"{node} not installed properly",
                        error_level=ErrorLevel.SEVERE.value,
                        resolution=f"Try deleting {node} folder, Dough will auto-download it",
                    )
                )

            else:
                # TODO: consolidate all the git methods in a single class
                repo = Repo(node_path)
                current_hash = repo.head.commit.hexsha if repo.head.is_detached else repo.rev_parse("HEAD")
                if current_hash != val["commit_hash"]:
                    error_list.append(
                        ErrorPayload(
                            error=f"{node} is a different version than what is expected",
                            error_level=ErrorLevel.SEVERE.value,
                            resolution=f"Either enable automatic update and restart the app or delete the {node} folder",
                        )
                    )

    st.write("Checking packages...")
    python_version = sys.version.split()[0]
    if not str(python_version).startswith("3.10"):
        error_list.append(
            ErrorPayload(
                error="Wrong python version installed",
                error_level=ErrorLevel.SEVERE.value,
                resolution="Python 3.10 is needed",
            )
        )
    missing_packages = check_python_and_packages()
    if missing_packages and len(missing_packages):
        missing_str = ""
        for pkg in missing_packages:
            missing_str += f"{pkg}, "
        error_list.append(
            ErrorPayload(
                error=f"{len(missing_packages)} missing packages",
                error_level=ErrorLevel.SEVERE.value,
                resolution="Re-install missing packages " + missing_str,
            )
        )

    return error_list


def check_python_and_packages():
    from packaging import version

    missing_packages = []
    try:
        with open("requirements.txt", "r") as file:
            required_packages = file.read().splitlines()

        installed_packages = subprocess.check_output(["pip", "freeze"]).decode("utf-8").splitlines()
        installed_packages_dict = {pkg.split("==")[0]: pkg.split("==")[1] for pkg in installed_packages}

        for package in required_packages:
            if "==" in package:
                package_name, package_version = package.split("==")
                if package_name not in installed_packages_dict:
                    missing_packages.append(package)
                else:
                    installed_version = installed_packages_dict[package_name]
                    if version.parse(installed_version) < version.parse(package_version):
                        missing_packages.append(package)
            else:
                package_name = package.strip()
                if package_name not in installed_packages_dict:
                    missing_packages.append(package)

    except FileNotFoundError:
        print("requirements.txt file not found.")

    ignored_packages = ["mesa"]
    return [pkg for pkg in missing_packages if pkg not in ignored_packages]


def api_key_input_component():
    data_repo = DataRepo()
    app_secrets = data_repo.get_app_secrets_from_user_uuid()
    if "replicate_key" in app_secrets and app_secrets["replicate_key"]:
        st.session_state["replicate_key"] = app_secrets["replicate_key"]
    else:
        st.session_state["replicate_key"] = ""
    if (
        st.session_state["replicate_key"] is None
        or st.session_state["replicate_key"] == ""
        or "replicate_key" not in st.session_state
    ):
        st.info(
            """
            Please enter your Replicate API key below to use prompt generation in Inspiration Engine. To get your API key, you’ll need to:

            1) Sign up for the Replicate platform **[here](https://replicate.com/)**.
            2) Create your API key by going into the "API tokens" section.
            3) Enter this key into the field.

            
            """
        )
    h1, _ = st.columns([1, 1])
    with h1:
        replicate_key = st.text_input("Replicate API Key:", st.session_state["replicate_key"])

    if st.button("Update"):
        if replicate_key != None and replicate_key != st.session_state["replicate_key"]:
            data_repo.update_app_setting(replicate_key=replicate_key)
            st.session_state["replicate_key"] = replicate_key
            st.success("API Key updated successfully.")
            time.sleep(0.7)
            st.rerun()


def update_toggle():
    data_repo = DataRepo()
    data_repo.update_app_setting(
        replicate_username="update" if st.session_state["enable_app_update"] else "no_update"
    )
