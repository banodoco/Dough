import streamlit as st
from moviepy.editor import *
from streamlit_javascript import st_javascript
import time
import os
import django

# loading the django app
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")
# Initialize Django
django.setup()

from banodoco_settings import project_init
from ui_components.models import InternalAppSettingObject
from utils.data_repo.data_repo import DataRepo

def main():
    st.set_page_config(page_title="Banodoco", page_icon="🎨", layout="wide")
    # initializing project constants
    project_init()
    
    data_repo = DataRepo()
    app_settings: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()
    app_secret = data_repo.get_app_secrets_from_user_uuid()

    # hide_img = '''
    # <style>
    # button[title="View fullscreen"]{
    #    display: none;}
    # </style>
    # '''
    # st.markdown(hide_img, unsafe_allow_html=True)
    
    # set online status of user
    if "online" not in st.session_state:
        current_url = st_javascript("await fetch('').then(r => window.parent.location.href)")
        time.sleep(1.5)
        # if current_url contains streamlit.app
        if current_url and "streamlit.app" in current_url:
            st.session_state["online"] = True    
        else:
            st.session_state["online"] = False
                           
        st.session_state["welcome_state"] = app_settings.welcome_state

    if st.session_state["online"] == True:
        st.error("**PLEASE READ:** This is a demo app. While you can click around, *buttons & queries won't work* and some things won't display properly. To use the proper version, follow the instructions [here](https://github.com/peter942/banodoco) to run it locally.")
    else:
        if app_secret["replicate_key"] == "":
            st.error("**To run restyling and other functions, you need to set your Replicate.com API key by going to Settings -> App Settings.**")
    
    from ui_components.setup import setup_app_ui
    setup_app_ui()
                                                    
if __name__ == '__main__':
    main()

