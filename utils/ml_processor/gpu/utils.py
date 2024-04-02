import importlib
import os
import sys
import subprocess
import time
from git import Repo
from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger


COMFY_RUNNER_PATH = "./comfy_runner"

def predict_gpu_output(workflow: str, file_path_list=[], output_node=None, extra_model_list=[], ignore_model_list=[]) -> str:
    # spec = importlib.util.spec_from_file_location('my_module', f'{COMFY_RUNNER_PATH}/inf.py')
    # comfy_runner = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(comfy_runner)
    
    # hackish sol.. waiting for comfy repo to be cloned
    while not is_comfy_runner_present():
        time.sleep(2)

    sys.path.append(str(os.getcwd()) + COMFY_RUNNER_PATH[1:])
    from comfy_runner.inf import ComfyRunner
    
    comfy_runner = ComfyRunner()
    output = comfy_runner.predict(
        workflow_input=workflow,
        file_path_list=file_path_list,
        stop_server_after_completion=False,
        output_node_ids=output_node,
        extra_models_list=extra_model_list,
        ignore_model_list=ignore_model_list
    )

    return output['file_paths']   # ignoring text output for now {"file_paths": [], "text_content": []}

def is_comfy_runner_present():
    return os.path.exists(COMFY_RUNNER_PATH)     # hackish sol, will fix later

# TODO: convert comfy_runner into a package for easy import
def setup_comfy_runner():
    if is_comfy_runner_present():
        return
    
    app_logger.log(LoggingType.INFO, 'cloning comfy runner')
    comfy_repo_url = "https://github.com/piyushK52/comfy-runner"
    Repo.clone_from(comfy_repo_url, COMFY_RUNNER_PATH[2:], single_branch=True, branch='feature/package')

    # installing dependencies
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', COMFY_RUNNER_PATH + '/requirements.txt'], check=True)