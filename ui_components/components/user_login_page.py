import time
import streamlit as st

from utils.common_utils import set_auth_token
from utils.local_storage.url_storage import set_url_param
from utils.state_refresh import refresh_app
from utils.third_party_auth.google.google_auth import get_auth_provider
from streamlit.web.server.server import Server


def user_login_ui():
    params = st.experimental_get_query_params()
    auth_provider = get_auth_provider()

    # http://localhost:5500/?code=4%2F0AQlEd8xV0xpyTCHnHJH8zopNtZ033s7m419wdSXLT7-fYK5KSk5PqDYR0bdM0F8UXzjJMQ&scope=email+profile+openid+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.profile+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email&authuser=0&prompt=consent
    # params found:  {'code': ['4/0AQlEd8xy2YawBOvBcWiKBpwqtRtEE5i5hbpfx'], 'scope': ['email profile openid], 'authuser': ['0'], 'prompt': ['consent']}
    if params and "code" in params and not st.session_state.get("retry_login"):
        if st.session_state.get("retry_login"):
            st.session_state["retry_login"] = False
            refresh_app()

        st.markdown("#### Logging you in, please wait...")
        # st.write(params['code'])
        data = {"id_token": params["code"][0]}
        auth_token, refresh_token, user = auth_provider.verify_auth_details(data)
        if auth_token and refresh_token:
            st.success("Successfully logged In, settings things up...")
            set_auth_token(auth_token, refresh_token, user)
            refresh_app()
        else:
            st.error("Unable to login..")
            if st.button("Retry Login", key="retry_login_btn"):
                st.session_state["retry_login"] = True
                refresh_app()
    else:
        st.session_state["retry_login"] = False
        st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")
        st.markdown("#### Login with Google to proceed")

        auth_url = auth_provider.get_auth_url(redirect_uri="http://localhost:5500")
        if auth_url:
            st.markdown(auth_url, unsafe_allow_html=True)
        else:
            time.sleep(0.1)
            st.warning("Unable to generate login link, please contact support")
