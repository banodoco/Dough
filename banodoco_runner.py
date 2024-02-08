import json
import os
import signal
import sys
import time
import requests
import traceback
import sentry_sdk
import setproctitle
from dotenv import load_dotenv
import django
from shared.constants import OFFLINE_MODE, InferenceParamType, InferenceStatus, InferenceType, ProjectMetaData, HOSTED_BACKGROUND_RUNNER_MODE
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.methods.file_methods import load_from_env, save_to_env
from utils.common_utils import acquire_lock, release_lock
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import replicate_status_map

from utils.constants import RUNNER_PROCESS_NAME, AUTH_TOKEN, REFRESH_AUTH_TOKEN
from utils.ml_processor.gpu.utils import is_comfy_runner_present, predict_gpu_output, setup_comfy_runner


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
    
    print('runner running')
    while True:
        if TERMINATE_SCRIPT:
            return

        if SERVER == 'development':
            if not is_app_running():
                if retries <=  0:
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

def check_and_update_db():
    # print("updating logs")
    from backend.models import InferenceLog, AppSetting, User
    
    app_logger = AppLogger()

    user = User.objects.filter(is_disabled=False).first()
    app_setting = AppSetting.objects.filter(user_id=user.id, is_disabled=False).first()
    replicate_key = app_setting.replicate_key_decrypted
    if not replicate_key:
        app_logger.log(LoggingType.ERROR, "Replicate key not found")
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
                                update_cache_dict(origin_data['inference_type'], log, timing_uuid, shot_uuid, timing_update_list, shot_update_list, gallery_update_list)

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
            print("here")
            data = json.loads(local_gpu_data)
            print("data 1")
            try:
                setup_comfy_runner()
                print("comfy setup")
                start_time = time.time()
                output = predict_gpu_output(data['workflow_input'], data['file_path_list'], data['output_node_ids'])
                end_time = time.time()

                # TODO: copy the output inside videos folder
                update_data = {
                    "status" : InferenceStatus.COMPLETED.value,
                    "output_details" : json.dumps(output[0]),
                    "total_inference_time" : end_time - start_time,
                }

                InferenceLog.objects.filter(id=log.id).update(**update_data)
                origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, {})
                origin_data['output'] = output[0]
                origin_data['log_uuid'] = log.uuid
                print("processing inference output")
                process_inference_output(**origin_data)
                timing_uuid, shot_uuid = origin_data.get('timing_uuid', None), origin_data.get('shot_uuid', None)
                update_cache_dict(origin_data['inference_type'], log, timing_uuid, shot_uuid, timing_update_list, shot_update_list, gallery_update_list)

            except Exception as e:
                print("error occured: ", str(e))
                # sentry_sdk.capture_exception(e)
                traceback.print_exc()
                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.FAILED.value)
        else:
            # if not replicate data is present then removing the status
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