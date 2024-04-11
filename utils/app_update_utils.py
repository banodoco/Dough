import glob
import hashlib
import os
import subprocess
from dotenv import dotenv_values
import requests
import streamlit as st
import threading
import sys
from streamlit_server_state import server_state_lock

from utils.data_repo.data_repo import DataRepo

update_event = threading.Event()
def check_for_updates():
    if not os.path.exists('banodoco_local.db'):
        return
    
    with server_state_lock["update_process"]:
        global update_event
        data_repo = DataRepo()
        app_setting = data_repo.get_app_setting_from_uuid()
        update_enabled = True if app_setting.replicate_username and app_setting.replicate_username == 'update' else False
        current_version = get_local_version()
        remote_version = get_remote_version()
        if current_version and remote_version and compare_versions(remote_version, current_version) == 1 and update_enabled\
            and not st.session_state.get("update_in_progress", False):
            st.info("Checking for updates...")
            st.session_state['update_in_progress'] = True
            update_thread = threading.Thread(target=update_app)
            update_thread.start()
            update_thread.join()
            update_event.wait()
            st.session_state['update_in_progress'] = False
            st.rerun()
    
def update_app():
    try:
        update_dough()
        update_comfy_runner()
        update_comfy_ui()
    except subprocess.CalledProcessError as e:
        print("Update failed:", str(e))

def update_comfy_runner():
    if os.path.exists("comfy_runner/"):
        os.chdir("comfy_runner/")
        subprocess.run(["git", "stash"], check=True)
        completed_process = subprocess.run(["git", "pull", "origin", "feature/package"], check=True)
        if completed_process.returncode == 0:
            print("Comfy runner updated")
        move_to_root()

def update_dough():
    print("Updating the app...")
    subprocess.run(["git", "stash"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    completed_process = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=True, capture_output=True, text=True)
    current_branch = completed_process.stdout.strip()
    subprocess.run(["git", "pull", "origin", current_branch], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists("banodoco_local.db"):
        python_executable = sys.executable
        completed_process = subprocess.run([python_executable, 'manage.py', 'migrate'], capture_output=True, text=True)
        if completed_process.returncode == 0:
            print("Database migration successful")
    
    if os.path.exists(".env"):
        sample_env = dotenv_values('.env.sample')
        env = dotenv_values('.env')
        missing_keys = [key for key in sample_env if key not in env]

        for key in missing_keys:
            env[key] = sample_env[key]

        with open(".env", 'w') as f:
            for key, value in env.items():
                f.write(f"{key}={value}\n")
                
        print("env update successful")
    
    move_to_root()

def update_comfy_ui():
    global update_event
    custom_nodes_dir = "ComfyUI/custom_nodes"
    if os.path.exists(custom_nodes_dir):
        initial_dir = os.getcwd()
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
                try:
                    subprocess.run(["git", "pull", "origin", "main"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"{folder} update successful")
                    
                    if requirements_files:
                        with open(requirements_file, "rb") as f:
                            new_hash = hashlib.sha256(f.read()).hexdigest()
                except subprocess.CalledProcessError as e:
                    print(f"Error updating {folder}: {e}")

                if old_hash and new_hash and old_hash != new_hash:
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                        print(f"{folder} requirements installed successfully")
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing requirements for {folder}: {e}")
                
                os.chdir(initial_dir)
                
    update_event.set()
    move_to_root()
    
def get_local_version():
    file_path = "./scripts/app_version.txt"
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except Exception as e:
        return None
    
def get_remote_version():
    completed_process = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=True, capture_output=True, text=True)
    current_branch = completed_process.stdout.strip()
    url = f"https://raw.githubusercontent.com/banodoco/Dough/{current_branch}/scripts/app_version.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return None
    
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
    current_dir = os.getcwd()
    while os.path.basename(current_dir) != "Dough":
        current_dir = os.path.dirname(current_dir)
        os.chdir(current_dir)