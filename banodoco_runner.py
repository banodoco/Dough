import json
import os
import time
import requests
import setproctitle
from dotenv import load_dotenv
import django
from shared.constants import InferenceParamType, InferenceStatus, InferenceType, ProjectMetaData, HOSTED_BACKGROUND_RUNNER_MODE
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.methods.file_methods import load_from_env, save_to_env
from utils.common_utils import acquire_lock, release_lock
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.replicate.constants import replicate_status_map

from utils.constants import RUNNER_PROCESS_NAME, AUTH_TOKEN, REFRESH_AUTH_TOKEN


load_dotenv()
setproctitle.setproctitle(RUNNER_PROCESS_NAME)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
django.setup()
SERVER = os.getenv('SERVER', 'development')

REFRESH_FREQUENCY = 2   # refresh every 2 seconds
MAX_APP_RETRY_CHECK = 3  # if the app is not running after 3 retries then the script will stop

def main():
    if SERVER != 'development' and not HOSTED_BACKGROUND_RUNNER_MODE:
        return
    
    retries = MAX_APP_RETRY_CHECK
    
    print('runner running')
    while True:
        if SERVER == 'development':
            if not is_app_running():
                if retries <=  0:
                    print('runner stopped')
                    return
                retries -= 1
            else:
                retries = min(retries + 1, MAX_APP_RETRY_CHECK)
        
        time.sleep(REFRESH_FREQUENCY)
        if HOSTED_BACKGROUND_RUNNER_MODE:
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

def check_and_update_db():
    # print("updating logs")
    from backend.models import InferenceLog, AppSetting
    
    app_logger = AppLogger()
    app_setting = AppSetting.objects.filter(is_disabled=False).first()
    replicate_key = app_setting.replicate_key_decrypted
    log_list = InferenceLog.objects.filter(status__in=[InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value],
                                           is_disabled=False).all()
    
    # these items will updated in the cache when the app refreshes the next time
    timing_update_list = {}     # {project_id: [timing_uuids]}
    gallery_update_list = {}    # {project_id: True/False}
    shot_update_list = {}       # {project_id: [shot_uuids]}

    for log in log_list:
        input_params = json.loads(log.input_params)
        replicate_data = input_params.get(InferenceParamType.REPLICATE_INFERENCE.value, None)
        if replicate_data:
            prediction_id = replicate_data['prediction_id']

            url = "https://api.replicate.com/v1/predictions/" + prediction_id
            headers = {
                "Authorization": f"Token {replicate_key}"
            }

            response = requests.get(url, headers=headers)
            if response.status_code in [200, 201]:
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
                        origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, None)
                        if origin_data and log_status == InferenceStatus.COMPLETED.value:
                            from ui_components.methods.common_methods import process_inference_output

                            try:
                                origin_data['output'] = output_details['output']
                                origin_data['log_uuid'] = log.uuid
                                print("processing inference output")
                                process_inference_output(**origin_data)

                                if origin_data['inference_type'] in [InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value, \
                                                                    InferenceType.FRAME_INPAINTING.value]:
                                    if str(log.project.uuid) not in timing_update_list:
                                        timing_update_list[str(log.project.uuid)] = []
                                    timing_update_list[str(log.project.uuid)].append(origin_data['timing_uuid'])

                                elif origin_data['inference_type'] == InferenceType.GALLERY_IMAGE_GENERATION.value:
                                    gallery_update_list[str(log.project.uuid)] = True
                                
                                elif origin_data['inference_type'] == InferenceType.FRAME_INTERPOLATION.value:
                                    if str(log.project.uuid) not in shot_update_list:
                                        shot_update_list[str(log.project.uuid)] = []
                                    shot_update_list[str(log.project.uuid)].append(origin_data['shot_uuid'])

                            except Exception as e:
                                app_logger.log(LoggingType.ERROR, f"Error: {e}")
                                output_details['error'] = str(e)
                                InferenceLog.objects.filter(id=log.id).update(status=InferenceStatus.FAILED.value, output_details=json.dumps(output_details))

                    else:
                        log_status = InferenceStatus.FAILED.value
                        InferenceLog.objects.filter(id=log.id).update(status=log_status, output_details=json.dumps(output_details))
                
                else:
                    InferenceLog.objects.filter(id=log.id).update(status=log_status)
            else:
                app_logger.log(LoggingType.DEBUG, f"Error: {response.content}")
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