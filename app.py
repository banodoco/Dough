import time
import streamlit as st
from moviepy.editor import *
import subprocess
import os
import django
from shared.constants import HOSTED_BACKGROUND_RUNNER_MODE, OFFLINE_MODE, SERVER, ServerType
import sentry_sdk
from shared.logging.logging import AppLogger
from utils.app_update_utils import check_for_updates
from utils.common_utils import is_process_active

from utils.constants import AUTH_TOKEN, RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT
from utils.local_storage.url_storage import delete_url_param, get_url_param, set_url_param
from utils.third_party_auth.google.google_auth import get_google_auth_url
from streamlit_server_state import server_state_lock


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
django.setup()
st.session_state['django_init'] = True

from banodoco_settings import project_init
from utils.data_repo.data_repo import DataRepo


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

def start_runner():
    if SERVER != ServerType.DEVELOPMENT.value and HOSTED_BACKGROUND_RUNNER_MODE in [False, 'False']:
        return
    
    with server_state_lock["runner"]:
        app_logger = AppLogger()
        
        if not is_process_active(RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT):
            app_logger.info("Starting runner")
            python_executable = sys.executable
            _ = subprocess.Popen([python_executable, "banodoco_runner.py"])
            max_retries = 6
            while not is_process_active(RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT) and max_retries:
                time.sleep(0.1)
                max_retries -= 1
        else:
            # app_logger.debug("Runner already running")
            pass

def main():
    st.set_page_config(page_title="Dough", layout="wide", page_icon="🎨")

    auth_details = get_url_param(AUTH_TOKEN)
    if (not auth_details or auth_details == 'None')\
        and SERVER != ServerType.DEVELOPMENT.value:
        params = st.experimental_get_query_params()
        
        if params and 'code' in params:
            st.markdown("#### Logging you in, please wait...")
            # st.write(params['code'])
            data = {
                "id_token": params['code'][0]
            }
            data_repo = DataRepo()
            user, token, refresh_token = data_repo.google_user_login(**data)
            if user:
                set_url_param(AUTH_TOKEN, str(token))
                st.rerun()
            else:
                delete_url_param(AUTH_TOKEN)
                st.error("Make sure you are added in the invite list and please login again")
                st.text("Join our discord to request access")
                discord_url = "<a target='_self' href='https://discord.gg/zGgpH9JEw4'> Banodoco Discord </a>"
                st.markdown(discord_url, unsafe_allow_html=True)
        else:
            st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")            
            st.markdown("#### Login with Google to proceed")
    
            auth_url = get_google_auth_url()
            st.markdown(auth_url, unsafe_allow_html=True)
            
    else:
        # if it's the first time, 
        if 'first_load' not in st.session_state:
            if not is_process_active(RUNNER_PROCESS_NAME, RUNNER_PROCESS_PORT):
                check_for_updates() # enabling auto updates only for local version
            st.session_state['first_load'] = True
        start_runner()
        project_init()
        
        from ui_components.setup import setup_app_ui
        from ui_components.components.welcome_page import welcome_page
        
        data_repo = DataRepo()
        app_setting = data_repo.get_app_setting_from_uuid()
        if app_setting.welcome_state == 2:
            setup_app_ui()
        else:            
            welcome_page()

        st.session_state['maintain_state'] = False
                                                    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise e

