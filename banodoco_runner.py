import os
import time
import requests
import setproctitle
from dotenv import load_dotenv
import django

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
    from backend.models import AppSetting

    app_settings = AppSetting.objects.first()
    print("user name in db: ", app_settings.user.name)
    return

main()