import time
import streamlit as st
from shared.constants import SERVER, ServerType
from utils.common_utils import get_current_user
from ui_components.components.query_logger_page import query_logger_page

from utils.data_repo.data_repo import DataRepo


def app_settings_page():
    data_repo = DataRepo()

    st.markdown("#### App Settings")
    st.markdown("***")
            
    if SERVER != ServerType.DEVELOPMENT.value:
        with st.expander("Purchase Credits", expanded=True):
            user_credits = get_current_user(invalidate_cache=True).total_credits
            user_credits = round(user_credits, 2) if user_credits else 0
            st.write(f"Total Credits: {user_credits}")
            c1, c2 = st.columns([1,1])
            with c1:
                if 'input_credits' not in st.session_state:
                    st.session_state['input_credits'] = 10

                credits = st.number_input("Credits (1 credit = $1)", value = st.session_state['input_credits'], step = 10)
                if credits != st.session_state['input_credits']:
                    st.session_state['input_credits'] = credits
                    st.rerun()

                if st.button("Generate payment link"):
                    if credits < 10:
                        st.error("Minimum credit value should be atleast 10")
                        time.sleep(0.7)
                        st.rerun()
                    else:
                        payment_link = data_repo.generate_payment_link(credits)
                        payment_link = f"""<a target='_self' href='{payment_link}'> PAYMENT LINK </a>"""
                        st.markdown(payment_link, unsafe_allow_html=True)
    
    # TODO: rn storing 'update_state' in replicate_username inside app_setting to bypass db changes, will change this later
    app_setting = data_repo.get_app_setting_from_uuid()
    update_enabled = True if app_setting.replicate_username and app_setting.replicate_username in ['update', 'bn'] else False
    with st.expander("App Update", expanded=True):
        
        # st.info("We recommend auto-updating the app to get the latest features and bug fixes. However, if you'd like to update manually, you can turn this off and use './scripts/entrypoint.sh --update' when you're starting the app to update.")
        st.toggle("Auto-update app upon restart", key='enable_app_update', value=update_enabled, on_change=update_toggle, help="This will update the app automatically when a new version is available.")

    with st.expander("API Keys", expanded=False):
        api_key_input_component()

    with st.expander("Inference Logs", expanded=False):
        query_logger_page()

def api_key_input_component():
    data_repo = DataRepo()
    app_secrets = data_repo.get_app_secrets_from_user_uuid()
    if 'stability_key' in app_secrets and app_secrets['stability_key']:
        st.session_state['stability_key'] = app_secrets['stability_key']
    else:
        st.session_state['stability_key'] = ""    
    st.write(st.session_state['stability_key'])
    if st.session_state['stability_key'] is None or st.session_state['stability_key'] == "" or 'stability_key' not in st.session_state:
        st.info("""
            Please enter your Stability API key below to use Stable Diffusion 3. To get your API key, you’ll need to:

            1) Sign up for Stability’s platform **[here](https://platform.stability.ai/docs/getting-started)**.
            2) Purchase credits **[here](https://platform.stability.ai/account/credits)**.
            3) Grab your API key **[here](https://platform.stability.ai/account/keys)**.
            4) Enter this key into the field.
             
            
            """)
    
    sai_key = st.text_input("Stability AI API Key", st.session_state['stability_key'])
    
    if st.button("Update"):

        if sai_key and sai_key != st.session_state['stability_key']:
            data_repo.update_app_setting(stability_key=sai_key)
            st.session_state['stability_key'] = sai_key
            st.success("API Key updated successfully.")
            time.sleep(0.7)
            st.rerun()

def update_toggle():
    data_repo = DataRepo()
    data_repo.update_app_setting(replicate_username='update' if st.session_state['enable_app_update'] else 'no_update')