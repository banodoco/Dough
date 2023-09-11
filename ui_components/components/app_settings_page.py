import streamlit as st
import webbrowser
from shared.constants import SERVER, ServerType
from utils.common_utils import get_current_user

from utils.data_repo.data_repo import DataRepo


def app_settings_page():
    # TODO: automatically pick the current user for fetching related details
    data_repo = DataRepo()
    app_settings = data_repo.get_app_setting_from_uuid()
    app_secrets = data_repo.get_app_secrets_from_user_uuid()
            
    if SERVER == ServerType.DEVELOPMENT.value:
        with st.expander("Replicate API Keys:"):
            replicate_username = st.text_input("replicate_username", value = app_secrets["replicate_username"])
            replicate_key = st.text_input("replicate_key", value = app_secrets["replicate_key"])
            if st.button("Save Settings"):
                data_repo.update_app_setting(replicate_username=replicate_username)
                data_repo.update_app_setting(replicate_key=replicate_key)
                st.experimental_rerun()

    if SERVER != ServerType.DEVELOPMENT.value:
        with st.expander("Purchase Credits"):
            user_credits = get_current_user(fresh_fetch=True)['total_credits']
            st.write(f"Total Credits: {user_credits}")
            c1, c2 = st.columns([1,1])
            with c1:
                if 'input_credits' not in st.session_state:
                    st.session_state['input_credits'] = 10

                credits = st.number_input("Credits (1 credit = $1)", value = st.session_state['input_credits'], step = 10)
                if credits != st.session_state['input_credits']:
                    st.session_state['input_credits'] = credits
                    st.experimental_rerun()

                if st.button("Generate payment link"):
                    payment_link = data_repo.generate_payment_link(credits)
                    payment_link = f"""<a target='_self' href='{payment_link}'> PAYMENT LINK </a>"""
                    st.markdown(payment_link, unsafe_allow_html=True)
    

    # locally_or_hosted = st.radio("Do you want to store your files locally or on AWS?", ("Locally", "AWS"),disabled=True, help="Only local storage is available at the moment, let me know if you need AWS storage - it should be pretty easy.")
    
    # if locally_or_hosted == "AWS":
    #     with st.expander("AWS API Keys:"):
    #         aws_access_key_id = st.text_input("aws_access_key_id", value = app_settings["aws_access_key_id"])
    #         aws_secret_access_key = st.text_input("aws_secret_access_key", value = app_settings["aws_secret_access_key"])
                    