import importlib
import os
import sys
import subprocess
import time
from git import Repo
from shared.constants import COMFY_BASE_PATH
from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from utils.common_utils import get_toml_config
from utils.constants import TomlConfig


COMFY_RUNNER_PATH = "./comfy_runner"


def predict_gpu_output(
    workflow: str,
    file_path_list=[],
    output_node=None,
    extra_model_list=[],
    ignore_model_list=[],
    log_tag=None,
) -> str:
    # spec = importlib.util.spec_from_file_location('my_module', f'{COMFY_RUNNER_PATH}/inf.py')
    # comfy_runner = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(comfy_runner)

    # hackish sol.. waiting for comfy repo to be cloned
    while not is_comfy_runner_present():
        time.sleep(2)

    sys.path.append(str(os.getcwd()) + COMFY_RUNNER_PATH[1:])
    from comfy_runner.inf import ComfyRunner

    comfy_commit_hash = get_toml_config(TomlConfig.COMFY_VERSION.value)["commit_hash"]
    node_commit_dict = get_toml_config(TomlConfig.NODE_VERSION.value)
    pkg_versions = get_toml_config(TomlConfig.PKG_VERSIONS.value)
    extra_node_urls = []
    for k, v in node_commit_dict.items():
        v["title"] = k
        extra_node_urls.append(v)

    comfy_runner = ComfyRunner()
    output = comfy_runner.predict(
        workflow_input=workflow,
        file_path_list=file_path_list,
        stop_server_after_completion=False,
        output_node_ids=output_node,
        extra_models_list=extra_model_list,
        ignore_model_list=ignore_model_list,
        client_id=log_tag,
        extra_node_urls=extra_node_urls,
        comfy_commit_hash=comfy_commit_hash,
        strict_dep_list=pkg_versions
    )

    return output["file_paths"]  # ignoring text output for now {"file_paths": [], "text_content": []}


def is_comfy_runner_present():
    return os.path.exists(COMFY_RUNNER_PATH)  # hackish sol, will fix later


# TODO: convert comfy_runner into a package for easy import
def setup_comfy_runner():
    if is_comfy_runner_present():
        update_comfy_runner_env()
        return

    app_logger.log(LoggingType.INFO, "cloning comfy runner")
    comfy_repo_url = "https://github.com/piyushK52/comfy-runner"
    Repo.clone_from(comfy_repo_url, COMFY_RUNNER_PATH[2:], single_branch=True, branch="main")

    # installing dependencies
    subprocess.run(["pip", "install", "-r", COMFY_RUNNER_PATH + "/requirements.txt"], check=True)
    update_comfy_runner_env()


def find_comfy_runner():
    # just keep going up the directory tree, till we find comfy_runner
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        if os.path.exists(os.path.join(current_path, '.git')):
            comfy_runner_path = os.path.join(current_path, 'comfy_runner')
            if os.path.exists(comfy_runner_path):
                return comfy_runner_path
            else:
                return None  # comfy_runner not found in the project root
        
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            return None
        
        current_path = parent_path

def update_comfy_runner_env():
    comfy_base_path = os.getenv("COMFY_MODELS_BASE_PATH", "ComfyUI")
    comfy_runner_path = find_comfy_runner()
    if not comfy_runner_path:
        print("comfy_runner not present")
        return
    
    if comfy_base_path != "ComfyUI":
        env_file_path = os.path.join(comfy_runner_path, ".env")
        try:
            os.makedirs(os.path.dirname(env_file_path), exist_ok=True)

            with open(env_file_path, "w", encoding="utf-8") as f:
                f.write(f"COMFY_RUNNER_MODELS_BASE_PATH={comfy_base_path}")
            
            with open(env_file_path, "r", encoding="utf-8") as f:
                written_content = f.read()
            
            if written_content != f"COMFY_RUNNER_MODELS_BASE_PATH={comfy_base_path}":
                print(f"File was written, but content doesn't match. Expected: {comfy_base_path}, Got: {written_content}")

        except IOError as e:
            print(f"IOError occurred while writing to {env_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    else:
        with open("comfy_runner/.env", "w", encoding="utf-8") as f:
            f.write("")
