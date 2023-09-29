import webbrowser
import streamlit as st
from moviepy.editor import *
import time
import os
import django
from shared.constants import OFFLINE_MODE, SERVER, ServerType
import sentry_sdk

from utils.constants import AUTH_TOKEN, LOGGED_USER
from utils.local_storage.url_storage import delete_url_param, get_url_param, set_url_param
from utils.third_party_auth.google.google_auth import get_google_auth_url

# loading the django app
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
# Initialize Django
django.setup()

from banodoco_settings import project_init
from ui_components.models import InternalAppSettingObject
from utils.data_repo.data_repo import DataRepo



if OFFLINE_MODE:
    SENTRY_DSN = os.getenv('SENTRY_DSN', '')
    SENTRY_ENV = os.getenv('SENTRY_ENV', '')
else:
    import boto3
    ssm = boto3.client("ssm", region_name="ap-south-1")

    # SENTRY_ENV = ssm.get_parameter(Name='/banodoco-fe/sentry/environment')['Parameter']['Value']
    # SENTRY_DSN = ssm.get_parameter(Name='/banodoco-fe/sentry/dsn')['Parameter']['Value']

sentry_sdk.init(
    environment=SENTRY_ENV,
    dsn=SENTRY_DSN,
    traces_sample_rate=0
)

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
                st.session_state[LOGGED_USER] = user.to_json() if user else None
                set_url_param(AUTH_TOKEN, str(token))
                # st.experimental_set_query_params(test='testing')
                st.experimental_rerun()
            else:
                delete_url_param(AUTH_TOKEN)
                st.error("please login again")
    else:
        
        project_init()
        
        data_repo = DataRepo()
        app_settings: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()
        app_secret = data_repo.get_app_secrets_from_user_uuid()
        
        from ui_components.setup import setup_app_ui
        setup_app_ui()
                                                    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise e

