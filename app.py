import threading
import time
import streamlit as st
from moviepy.editor import *
import subprocess
import os
import django
from shared.constants import OFFLINE_MODE, SERVER, ServerType
import sentry_sdk
from shared.logging.logging import AppLogger
from utils.common_utils import is_process_active

from utils.constants import AUTH_TOKEN, RUNNER_PROCESS_NAME
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
    # if SERVER != ServerType.DEVELOPMENT.value:
    #     return
    
    with server_state_lock["runner"]:
        app_logger = AppLogger()
        
        if not is_process_active(RUNNER_PROCESS_NAME):
            app_logger.info("Starting runner")
            # _ = subprocess.Popen(["streamlit", "run", "banodoco_runner.py", "--runner.fastReruns", "false", "--server.port", "5502", "--server.headless", "true"])
            _ = subprocess.Popen(["python", "banodoco_runner.py"])
            while not is_process_active(RUNNER_PROCESS_NAME):
                time.sleep(0.1)
        else:
            app_logger.debug("Runner already running")

def main():
    st.set_page_config(page_title="Banodoco", page_icon="🎨", layout="wide")

    auth_details = get_url_param(AUTH_TOKEN)
    if (not auth_details or auth_details == 'None')\
        and SERVER != ServerType.DEVELOPMENT.value:
        st.markdown("# :red[ba]:green[no]:orange[do]:blue[co]")
        st.subheader("Login with Google to proceed")
 
        auth_url = get_google_auth_url()
        st.markdown(auth_url, unsafe_allow_html=True)
        
        params = st.experimental_get_query_params()
        if params and 'code' in params:
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
                st.error("please login again")
    else:
        start_runner()
        project_init()
        
        from ui_components.setup import setup_app_ui
        setup_app_ui()
                                                    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise e

