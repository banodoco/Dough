import json
import os
import time
import requests
import setproctitle
from dotenv import load_dotenv
import django
from shared.constants import InferenceParamType, InferenceStatus
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from utils.ml_processor.replicate.constants import replicate_status_map

from utils.constants import RUNNER_PROCESS_NAME


load_dotenv()
setproctitle.setproctitle(RUNNER_PROCESS_NAME)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
django.setup()
SERVER = os.getenv('SERVER', 'development')

def main():
    if SERVER != 'development':
        return
    
    retries = 3
    
    print('runner running')
    while True:
        if not is_app_running():
            if retries <=  0:
                print('runner stopped')
                return
            retries -= 1
        else:
            retries = min(retries + 1, 3)
        
        time.sleep(1)
        check_and_update_db()

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
    replicate_key = app_setting.replicate_key
    log_list = InferenceLog.objects.filter(status__in=[InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value],
                                           is_disabled=False).all()
    
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
                    output_details['output'] = result['output']    
                
                InferenceLog.objects.filter(id=log.id).update(status=log_status, output_details=json.dumps(output_details))
            else:
                app_logger.log(LoggingType.DEBUG, f"Error: {response.content}")
    return

main()