import json
import os
import shutil
import signal
import sys
import time
import uuid
import psutil
import requests
import socket
import platform
import traceback
import sentry_sdk
import setproctitle
from dotenv import load_dotenv
import django
from shared.constants import COMFY_PORT, LOCAL_DATABASE_NAME, OFFLINE_MODE, InferenceParamType, InferenceStatus, InferenceType, ProjectMetaData, HOSTED_BACKGROUND_RUNNER_MODE
from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from shared.utils import get_file_type
from ui_components.methods.file_methods import get_file_bytes_and_extension, load_from_env, save_or_host_file_bytes, save_to_env
from utils.common_utils import acquire_lock, release_lock
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ComfyWorkflow, replicate_status_map

from utils.constants import RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT, AUTH_TOKEN, REFRESH_AUTH_TOKEN
from utils.ml_processor.gpu.utils import is_comfy_runner_present, predict_gpu_output, setup_comfy_runner
from utils.ml_processor.sai.utils import predict_sai_output


load_dotenv()
setproctitle.setproctitle(RUNNER_PROCESS_NAME)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
django.setup()
SERVER = os.getenv('SERVER', 'development')

REFRESH_FREQUENCY = 2   # refresh every 2 seconds
MAX_APP_RETRY_CHECK = 3  # if the app is not running after 3 retries then the script will stop

TERMINATE_SCRIPT = False

# sentry init
if OFFLINE_MODE:
    SENTRY_DSN = os.getenv('SENTRY_DSN', '')
    SENTRY_ENV = os.getenv('SENTRY_ENV', '')
else:
    import boto3
    ssm = boto3.client("ssm", region_name="ap-south-1")

    SENTRY_ENV = ssm.get_parameter(Name='/banodoco-fe/sentry/environment')['Parameter']['Value']
    SENTRY_DSN = ssm.get_parameter(Name='/banodoco-fe/sentry/dsn')['Parameter']['Value']

sentry_sdk.init(
    environment=SENTRY_ENV,
    dsn=SENTRY_DSN,
    traces_sample_rate=0
)

def handle_termination(signal, frame):
    print("Received termination signal. Cleaning up...")
    global TERMINATE_SCRIPT
    TERMINATE_SCRIPT = True
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_termination)

def main():
    if SERVER != 'development' and HOSTED_BACKGROUND_RUNNER_MODE in [False, 'False']:
        return
    
    retries = MAX_APP_RETRY_CHECK

    # in case of windows opening a dummy socket (to signal that the process has started)
    if platform.system() == "Windows" and OFFLINE_MODE:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("localhost", RUNNER_PROCESS_PORT))
        server_socket.listen(100)     # hacky fix
    
    print('runner running')
    while True:
        if TERMINATE_SCRIPT:
            stop_server(COMFY_PORT)
            return

        if SERVER == 'development':
            if not is_app_running():
                if retries <=  0:
                    stop_server(COMFY_PORT)
                    print('runner stopped')
                    return
                retries -= 1
            else:
                retries = min(retries + 1, MAX_APP_RETRY_CHECK)
        
        time.sleep(REFRESH_FREQUENCY)
        if HOSTED_BACKGROUND_RUNNER_MODE not in [False, 'False']:
            validate_admin_auth_token()
        check_and_update_db()

# creates a 
def validate_admin_auth_token():
    data_repo = DataRepo()
    # check if a valid token is present
    auth_token = load_from_env(AUTH_TOKEN)
    refresh_token = load_from_env(REFRESH_AUTH_TOKEN)
    user, token = None, None
    if auth_token and valid_token(auth_token):
        return

    # check if a valid refresh_token is present
    elif refresh_token:
        user, token, refresh_token = data_repo.refresh_auth_token(refresh_token)

    # fetch fresh token and refresh_token
    if not (user and token):
        email = os.getenv('admin_email', '')
        password = os.getenv('admin_password')
        user, token, refresh_token = data_repo.user_password_login(email=email, password=password)

    if token:
        save_to_env(AUTH_TOKEN, token)
        save_to_env(REFRESH_AUTH_TOKEN, refresh_token)

def valid_token(token):
    data_repo = DataRepo()
    try:
        user = data_repo.get_first_active_user()
    except Exception as e:
        print("invalid token: ", str(e))
        return False

    return True if user else False

def is_app_running():
    url = 'http://localhost:5500/healthz'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            print(f"server not running")
            return False
    except requests.exceptions.RequestException as e:
        print("server not running")
        return False
    
def update_cache_dict(inference_type, log, timing_uuid, shot_uuid, timing_update_list, shot_update_list, gallery_update_list):
    if inference_type in [InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value, \
                                        InferenceType.FRAME_INPAINTING.value]:
        if str(log.project.uuid) not in timing_update_list:
            timing_update_list[str(log.project.uuid)] = []
        timing_update_list[str(log.project.uuid)].append(timing_uuid)

    elif inference_type == InferenceType.GALLERY_IMAGE_GENERATION.value:
        gallery_update_list[str(log.project.uuid)] = True
    
    elif inference_type == InferenceType.FRAME_INTERPOLATION.value:
        if str(log.project.uuid) not in shot_update_list:
            shot_update_list[str(log.project.uuid)] = []
        shot_update_list[str(log.project.uuid)].append(shot_uuid)
        
def find_process_by_port(port):
        pid = None
        for proc in psutil.process_iter(attrs=['pid', 'name', 'connections']):
            try:
                if proc and 'connections' in proc.info and proc.info['connections']:
                    for conn in proc.info['connections']:
                        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                            app_logger.log(LoggingType.DEBUG, f"Process {proc.info['pid']} (Port {port})")
                            pid = proc.info['pid']
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return pid
    
def stop_server(port):
        pid = find_process_by_port(port)
        if pid:
            app_logger.log(LoggingType.DEBUG, "comfy server stopped")
            process = psutil.Process(pid)
            process.terminate()
            process.wait()
            
def format_model_output(output, model_display_name):
    if model_display_name and model_display_name == ComfyWorkflow.MOTION_LORA.value:
        return output
    else:
        return [output[-1]]

def check_and_update_db():
    # print("updating logs")
    from backend.models import InferenceLog, AppSetting, User
    
    # waiting for db (hackish sol)
    while not os.path.exists(LOCAL_DATABASE_NAME):
        time.sleep(2)

    # returning if db creation and migrations are pending
    try:
        user = User.objects.filter(is_disabled=False).first()
    except Exception as e:
        app_logger.log(LoggingType.DEBUG, "db creation pending..")
        time.sleep(3)
        return

    if not user:
        return

    app_setting = AppSetting.objects.filter(user_id=user.id, is_disabled=False).first()
    replicate_key = app_setting.replicate_key_decrypted
    if not replicate_key:
        # app_logger.log(LoggingType.ERROR, "Replicate key not found")
        return
    
    log_list = InferenceLog.objects.filter(status__in=[InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value],
                                           is_disabled=False).all()
    
    # these items will updated in the cache when the app refreshes the next time
    timing_update_list = {}     # {project_id: [timing_uuids]}
    gallery_update_list = {}    # {project_id: True/False}
    shot_update_list = {}       # {project_id: [shot_uuids]}

    for log in log_list:
        input_params = json.loads(log.input_params)
        replicate_data = input_params.get(InferenceParamType.REPLICATE_INFERENCE.value, None)
        local_gpu_data = input_params.get(InferenceParamType.GPU_INFERENCE.value, None)
        sai_data = input_params.get(InferenceParamType.SAI_INFERENCE.value, None)
        if replicate_data:
            prediction_id = replicate_data['prediction_id']

            url = "https://api.replicate.com/v1/predictions/" + prediction_id
            headers = {
                "Authorization": f"Token {replicate_key}"
            }

            try:
                response = requests.get(url, headers=headers)
            except Exception as e:
                sentry_sdk.capture_exception(e)
                response = None

            if response and response.status_code in [200, 201]:
                # print("response: ", response)
                result = response.json()
                log_status = replicate_status_map[result['status']] if result['status'] in replicate_status_map else InferenceStatus.IN_PROGRESS.value
                output_details = json.loads(log.output_details)
                
                if log_status == InferenceStatus.COMPLETED.value:
                    if 'output' in result and result['output']:
                        output_details['output'] = result['output'] if (output_details['version'] == \
                            "a4a8bafd6089e1716b06057c42b19378250d008b80fe87caa5cd36d40c1eda90" or \
                                isinstance(result['output'], str)) else [result['output'][-1]]
                        
                        # updating the output url (to prevent file path errors in the runtime)
                        output = output_details['output']
                        output = output[0] if isinstance(output, list) else output
                        file_bytes, file_ext = get_file_bytes_and_extension(output)
                        file_path = "videos/temp/" + str(uuid.uuid4()) + "." + file_ext
                        file_path = save_or_host_file_bytes(file_bytes, file_path, file_ext) or file_path
                        output_details['output'] = file_path
                        
                        update_data = {
                            "status" : log_status,
                            "output_details" : json.dumps(output_details)
                        }
                        if 'metrics' in result and result['metrics'] and 'predict_time' in result['metrics']:
                            update_data['total_inference_time'] = float(result['metrics']['predict_time'])

                        InferenceLog.objects.filter(id=log.id).update(**update_data)
                        origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, {})
                        if origin_data and log_status == InferenceStatus.COMPLETED.value:
                            from ui_components.methods.common_methods import process_inference_output

                            try:
                                origin_data['output'] = output_details['output']
                                origin_data['log_uuid'] = log.uuid
                                print("processing inference output")
                                process_inference_output(**origin_data)
                                timing_uuid, shot_uuid = origin_data.get('timing_uuid', None), origin_data.get('shot_uuid', None)
                                update_cache_dict(origin_data.get('inference_type', ""), log, timing_uuid, shot_uuid, timing_update_list, shot_update_list, gallery_update_list)

                            except Exception as e:
                                app_logger.log(LoggingType.ERROR, f"Error: {e}")
                                output_details['error'] = str(e)
                                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.FAILED.value, output_details=json.dumps(output_details))
                                sentry_sdk.capture_exception(e)

                    else:
                        log_status = InferenceStatus.FAILED.value
                        InferenceLog.objects.filter(id=log.id).update(status=log_status, output_details=json.dumps(output_details))
                
                else:
                    InferenceLog.objects.filter(id=log.id).update(status=log_status)
            else:
                if response:
                    app_logger.log(LoggingType.DEBUG, f"Error: {response.content}")
                    sentry_sdk.capture_exception(response.content)
        elif local_gpu_data:
            data = json.loads(local_gpu_data)
            try:
                setup_comfy_runner()
                
                # fetching the current status again (as this could have been cancelled)
                log = InferenceLog.objects.filter(id=log.id).first()
                cur_status = log.status
                if cur_status in [InferenceStatus.FAILED.value, InferenceStatus.CANCELED.value]:
                    return
                
                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.IN_PROGRESS.value)
                start_time = time.time()
                output = predict_gpu_output(data['workflow_input'], data['file_path_list'], \
                    data['output_node_ids'], data.get("extra_model_list", []), data.get("ignore_model_list", []))
                end_time = time.time()

                res_output = format_model_output(output, log.model_name)
                destination_path_list = []
                for output in res_output:
                    destination_path = "./videos/temp/" + str(uuid.uuid4()) + "." + output.split(".")[-1]
                    shutil.copy2("./output/" + output, destination_path)
                    destination_path_list.append(destination_path)
                    
                output_details = json.loads(log.output_details)
                output_details['output'] = destination_path_list[0] if len(destination_path_list) == 1 else destination_path_list
                
                log = InferenceLog.objects.filter(id=log.id).first()
                cur_status = log.status
                if cur_status in [InferenceStatus.FAILED.value, InferenceStatus.CANCELED.value]:
                    return

                update_data = {
                    "status" : InferenceStatus.COMPLETED.value,
                    "output_details" : json.dumps(output_details),
                    "total_inference_time" : end_time - start_time,
                }

                InferenceLog.objects.filter(id=log.id).update(**update_data)
                origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, {})
                origin_data['output'] = destination_path_list[0] if len(destination_path_list) == 1 else destination_path_list
                origin_data['log_uuid'] = log.uuid
                print("processing inference output")

                from ui_components.methods.common_methods import process_inference_output
                process_inference_output(**origin_data)
                timing_uuid, shot_uuid = origin_data.get('timing_uuid', None), origin_data.get('shot_uuid', None)
                update_cache_dict(origin_data.get('inference_type', ''), log, timing_uuid, shot_uuid, timing_update_list, shot_update_list, gallery_update_list)

            except Exception as e:
                print("error occured: ", str(e))
                # sentry_sdk.capture_exception(e)
                traceback.print_exc()
                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.FAILED.value)
        elif sai_data:
            # TODO: a lot of code is being repeated in the different types of inference, will fix this later
            try:
                data = sai_data
                log = InferenceLog.objects.filter(id=log.id).first()
                cur_status = log.status
                if cur_status in [InferenceStatus.FAILED.value, InferenceStatus.CANCELED.value]:
                    return
                
                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.IN_PROGRESS.value)
                start_time = time.time()
                output = predict_sai_output(data)
                end_time = time.time()
                
                destination_path_list = []
                destination_path = "./videos/temp/" + str(uuid.uuid4()) + "." + output.split(".")[-1]
                shutil.copy2(output, destination_path)
                destination_path_list.append(destination_path)
                
                output_details = json.loads(log.output_details)
                output_details['output'] = destination_path_list[0] if len(destination_path_list) == 1 else destination_path_list
                update_data = {
                    "status" : InferenceStatus.COMPLETED.value,
                    "output_details" : json.dumps(output_details),
                    "total_inference_time" : end_time - start_time,
                }
                
                InferenceLog.objects.filter(id=log.id).update(**update_data)
                origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, {})
                origin_data['output'] = destination_path_list[0] if len(destination_path_list) == 1 else destination_path_list
                origin_data['log_uuid'] = log.uuid
                print("processing inference output")

                from ui_components.methods.common_methods import process_inference_output
                process_inference_output(**origin_data)
                timing_uuid, shot_uuid = origin_data.get('timing_uuid', None), origin_data.get('shot_uuid', None)
                update_cache_dict(origin_data.get('inference_type', ''), log, timing_uuid, shot_uuid, timing_update_list, shot_update_list, gallery_update_list)
            except Exception as e:
                print("error occured: ", str(e))
                # sentry_sdk.capture_exception(e)
                traceback.print_exc()
                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.FAILED.value)
        else:
            # if replicate/gpu data is not present then removing the status
            InferenceLog.objects.filter(id=log.id).update(status="")

    # adding update_data in the project
    from backend.models import Project

    final_res = {}
    for project_uuid, val in timing_update_list.items():
        final_res[project_uuid] = {ProjectMetaData.DATA_UPDATE.value: list(set(val))}

    for project_uuid, val in gallery_update_list.items():
        if project_uuid not in final_res:
            final_res[project_uuid] = {}
        
        final_res[project_uuid].update({f"{ProjectMetaData.GALLERY_UPDATE.value}": val})

    for project_uuid, val in shot_update_list.items():
        final_res[project_uuid] = {ProjectMetaData.SHOT_VIDEO_UPDATE.value: list(set(val))}
    

    for project_uuid, val in final_res.items():
        key = str(project_uuid)
        if acquire_lock(key):
            _ = Project.objects.filter(uuid=project_uuid).update(meta_data=json.dumps(val))
            release_lock(key)

    if not len(log_list):
        # app_logger.log(LoggingType.DEBUG, f"No logs found")
        pass

    return

main()