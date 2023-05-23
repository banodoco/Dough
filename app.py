import streamlit as st
from moviepy.editor import *
from streamlit_javascript import st_javascript
import time
from repository.local_repo.csv_repo import get_app_settings

from banodoco_settings import project_init

def main():
    from ui_components.setup import setup_app_ui

    st.set_page_config(page_title="Banodoco", page_icon="🎨", layout="wide")
    # initializing project constants
    project_init()
    
    app_settings = get_app_settings()

    hide_img = '''
    <style>
    button[title="View fullscreen"]{
        display: none;}
    </style>
    '''
    st.markdown(hide_img, unsafe_allow_html=True)
    
    # set online status of user
    if "online" not in st.session_state:
        current_url = st_javascript("await fetch('').then(r => window.parent.location.href)")
        time.sleep(1.5)
        # if current_url contains streamlit.app
        if current_url and "streamlit.app" in current_url:
            st.session_state["online"] = True    
        else:
            st.session_state["online"] = False
                           
        st.session_state["welcome_state"] = app_settings["welcome_state"]

    if st.session_state["online"] == True:
        st.error("**PLEASE READ:** This is a demo app. While you can click around, *buttons & queries won't work* and some things won't display properly. To use the proper version, follow the instructions [here](https://github.com/peter942/banodoco) to run it locally.")
    else:
        if app_settings["replicate_com_api_key"] == "":
            st.error("**To run restyling and other functions, you need to set your Replicate.com API key by going to Settings -> App Settings.**")
    

    setup_app_ui()
                                                    
if __name__ == '__main__':
    main()

