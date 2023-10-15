import json
import os
import time
import requests
import setproctitle
from dotenv import load_dotenv
import django
from shared.constants import InferenceParamType, InferenceStatus, ProjectMetaData
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from utils.common_utils import acquire_lock, release_lock
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.replicate.constants import replicate_status_map

from utils.constants import RUNNER_PROCESS_NAME


load_dotenv()
setproctitle.setproctitle(RUNNER_PROCESS_NAME)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
django.setup()
SERVER = os.getenv('SERVER', 'development')

REFRESH_FREQUENCY = 2   # refresh every 2 seconds
MAX_APP_RETRY_CHECK = 3  # if the app is not running after 3 retries then the script will stop

def main():
    if SERVER != 'development':
        return
    
    retries = MAX_APP_RETRY_CHECK
    
    print('runner running')
    while True:
        if not is_app_running():
            if retries <=  0:
                print('runner stopped')
                return
            retries -= 1
        else:
            retries = min(retries + 1, MAX_APP_RETRY_CHECK)
        
        time.sleep(REFRESH_FREQUENCY)
        check_and_update_db()
        # test_data_repo()

def test_data_repo():
    data_repo = DataRepo()
    app_settings = data_repo.get_app_setting_from_uuid()
    print(app_settings.replicate_username)

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
    from backend.models import InferenceLog, AppSetting
    
    app_logger = AppLogger()
    app_setting = AppSetting.objects.filter(is_disabled=False).first()
    replicate_key = app_setting.replicate_key_decrypted
    log_list = InferenceLog.objects.filter(status__in=[InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value],
                                           is_disabled=False).all()
    
    timing_update_list = {}     # {project_id: [timing_uuids]}
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
                result = response.json()
                log_status = replicate_status_map[result['status']] if result['status'] in replicate_status_map else InferenceStatus.IN_PROGRESS.value
                output_details = json.loads(log.output_details)
                
                if log_status == InferenceStatus.COMPLETED.value:
                    output_details['output'] = result['output'] if (output_details['version'] == \
                        "a4a8bafd6089e1716b06057c42b19378250d008b80fe87caa5cd36d40c1eda90" or \
                            isinstance(result['output'], str)) else [result['output'][-1]]
                
                InferenceLog.objects.filter(id=log.id).update(status=log_status, output_details=json.dumps(output_details))
                origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, None)
                if origin_data and log_status == InferenceStatus.COMPLETED.value:
                    from ui_components.methods.common_methods import process_inference_output

                    origin_data['output'] = output_details['output']
                    origin_data['log_uuid'] = log.uuid
                    print("processing inference output")
                    process_inference_output(**origin_data)
                    if  str(log.project.uuid) not in timing_update_list:
                        timing_update_list[str(log.project.uuid)] = []
                    
                    timing_update_list[str(log.project.uuid)].append(origin_data['timing_uuid'])

            else:
                app_logger.log(LoggingType.DEBUG, f"Error: {response.content}")
        else:
            # if not replicate data is present then removing the status
            InferenceLog.objects.filter(id=log.id).update(status="")

    # adding update_data in the project
    from backend.models import Project

    for project_uuid, val in timing_update_list.items():
        key = str(project_uuid)
        if acquire_lock(key):
            val = list(set(val))
            _ = Project.objects.filter(uuid=project_uuid).update(meta_data=json.dumps({ProjectMetaData.DATA_UPDATE.value: val}))
            release_lock(key)

    if not len(log_list):
        # app_logger.log(LoggingType.DEBUG, f"No logs found")
        pass

    return

main()