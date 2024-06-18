import glob
import hashlib
import os
import shutil
import subprocess
from dotenv import dotenv_values
import requests
import streamlit as st
import threading
import sys
from git import Repo
from streamlit_server_state import server_state_lock

from utils.common_utils import get_toml_config
from utils.constants import TomlConfig
from utils.data_repo.data_repo import DataRepo

update_event = threading.Event()
dough_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
comfy_runner_dir = os.path.join(dough_dir, "comfy_runner")
comfy_ui_dir = os.path.join(dough_dir, "ComfyUI")


def check_for_updates():
    if not os.path.exists("banodoco_local.db"):
        return

    with server_state_lock["update_process"]:
        global update_event
        data_repo = DataRepo()
        app_setting = data_repo.get_app_setting_from_uuid()
        update_enabled = (
            True
            if app_setting.replicate_username and app_setting.replicate_username in ["bn", "update"]
            else False
        )
        current_version = get_local_version()
        remote_version = get_remote_version()
        if (
            current_version
            and remote_version
            and compare_versions(remote_version, current_version) == 1
            and update_enabled
            and not st.session_state.get("update_in_progress", False)
        ):
            st.info("Checking for updates...")
            st.session_state["update_in_progress"] = True
            update_thread = threading.Thread(target=update_app)
            update_thread.start()
            update_thread.join()
            update_event.wait()
            st.session_state["update_in_progress"] = False
            st.session_state["first_load"] = True
            st.rerun()


def update_app():
    try:
        update_dough()
        update_comfy_runner()
        update_comfy_ui()
    except Exception as e:
        print("Update failed:", str(e))


def update_comfy_runner():
    if os.path.exists(comfy_runner_dir):
        os.chdir(comfy_runner_dir)
        try:
            repo = Repo(comfy_runner_dir)
            current_branch = repo.active_branch

            # deleting the folder if it's on some other branch and cloning
            # a fresh copy
            if current_branch != "main":
                shutil.rmtree(comfy_runner_dir)
                move_to_root()
                Repo.clone_from("https://github.com/piyushK52/comfy_runner", "./comfy_runner")
            # updating if it's already on the main branch
            else:
                update_git_repo(comfy_runner_dir)
        except Exception as e:
            print(f"Error occured: {str(e)}")

        print("Comfy runner updated")
        move_to_root()


def update_dough():
    print("Updating the app...")

    try:
        update_git_repo(dough_dir)
    except Exception as e:
        print(f"Error occurred: {str(e)}")

    # performing db migrations if any
    if os.path.exists("banodoco_local.db"):
        python_executable = sys.executable
        completed_process = subprocess.run(
            [python_executable, "manage.py", "migrate"], capture_output=True, text=True
        )
        if completed_process.returncode == 0:
            print("Database migration successful")

    # updating env file
    if os.path.exists(".env"):
        sample_env = dotenv_values(".env.sample")
        env = dotenv_values(".env")
        missing_keys = [key for key in sample_env if key not in env]
        for key in missing_keys:
            env[key] = sample_env[key]
        with open(".env", "w") as f:
            for key, value in env.items():
                f.write(f"{key}={value}\n")
        print("env update successful")

    move_to_root()


def update_comfy_ui():
    global update_event
    custom_nodes_dir = os.path.join(comfy_ui_dir, "custom_nodes")
    node_commit_dict = get_toml_config(TomlConfig.NODE_VERSION.value)

    if os.path.exists(custom_nodes_dir):
        initial_dir = dough_dir
        for folder in os.listdir(custom_nodes_dir):
            folder_path = os.path.join(custom_nodes_dir, folder)
            if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, ".git")):
                print(f"Updating {folder}")
                os.chdir(folder_path)

                old_hash, new_hash = None, None
                requirements_files = glob.glob("requirements*.txt")
                if requirements_files:
                    requirements_file = requirements_files[0]
                    try:
                        with open(requirements_file, "rb") as f:
                            old_hash = hashlib.sha256(f.read()).hexdigest()
                    except FileNotFoundError:
                        print(f"Requirements file not found for {folder}")

                # moving to a stable commit version for this node and installing
                # deps only if they have changed
                try:
                    commit_hash = node_commit_dict.get(folder, {}).get("commit_hash", None)
                    update_git_repo(folder_path, commit_hash)
                    print(f"{folder} update successful")

                    if requirements_files:
                        with open(requirements_file, "rb") as f:
                            new_hash = hashlib.sha256(f.read()).hexdigest()
                except subprocess.CalledProcessError as e:
                    print(f"Error updating {folder}: {e}")

                if old_hash and new_hash and old_hash != new_hash:
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
                        )
                        print(f"{folder} requirements installed successfully")
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing requirements for {folder}: {e}")

                os.chdir(initial_dir)

    update_event.set()
    move_to_root()


# TODO: move all the git methods into a single class
def update_git_repo(git_dir, commit_hash=None):
    repo = Repo(git_dir)

    repo.git.stash()
    repo.remotes.origin.fetch()

    if repo.head.is_detached:
        current_hash = repo.head.commit.hexsha
        # print(f"Current repository is in detached HEAD state, current commit: {current_hash}")

        if commit_hash:
            if current_hash != commit_hash:
                print(f"Checking out stable commit: {commit_hash}")
                repo.git.checkout(commit_hash)
            else:
                print("Already at the stable commit")
        else:
            print("No commit hash provided. Skipping checkout.")
    else:
        current_branch = repo.active_branch
        print(f"Current branch: {current_branch.name}")

        if commit_hash:
            current_hash = repo.rev_parse("HEAD")
            if current_hash != commit_hash:
                print(f"Checking out stable commit: {commit_hash}")
                repo.git.checkout(commit_hash)
            else:
                print("Already at the stable commit")
        else:
            repo.remotes.origin.pull(current_branch.name)


def get_local_version():
    file_path = "./scripts/app_version.txt"
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except Exception as e:
        return None


def get_current_branch(git_dir):
    if not is_git_initialized(git_dir):
        init_git("../", "https://github.com/banodoco/Dough.git")

    try:
        completed_process = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=git_dir,
        )
        current_branch = completed_process.stdout.strip()
    except Exception as e:
        print("------ exception occured: ", str(e))
        current_branch = "main"

    return current_branch


def get_remote_version():
    current_branch = get_current_branch(dough_dir)

    url = f"https://raw.githubusercontent.com/banodoco/Dough/{current_branch}/scripts/app_version.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return None


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode("utf-8"), error.decode("utf-8")


def is_git_initialized(repo_folder):
    os.chdir(repo_folder)
    output, error = run_command("git rev-parse --is-inside-work-tree")
    os.chdir(dough_dir)
    return output.strip() == "true"


def init_git(repo_folder, repo_url):
    os.chdir(repo_folder)
    run_command("git init")
    run_command(f"git remote add origin {repo_url}")
    run_command("git fetch origin")
    run_command("git reset --hard origin/main")

    os.chdir("..")


def compare_versions(version1, version2):
    ver1 = [int(x) for x in version1.split(".")]
    ver2 = [int(x) for x in version2.split(".")]

    max_len = max(len(ver1), len(ver2))
    ver1 += [0] * (max_len - len(ver1))
    ver2 += [0] * (max_len - len(ver2))

    for i in range(len(ver1) - 1):
        if ver1[i] > ver2[i]:
            return 1
        elif ver1[i] < ver2[i]:
            return -1

    if ver1[-1] > ver2[-1]:
        return 1
    elif ver1[-1] < ver2[-1]:
        return -1
    else:
        return 0


def move_to_root():
    os.chdir(dough_dir)
    # current_dir = os.getcwd()
    # while os.path.basename(current_dir) != "Dough":
    #     current_dir = os.path.dirname(current_dir)
    #     os.chdir(current_dir)
