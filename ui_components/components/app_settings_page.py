import streamlit as st

from utils.data_repo.data_repo import DataRepo


def app_settings_page():
    # TODO: automatically pick the current user for fetching related details
    data_repo = DataRepo()
    app_settings = data_repo.get_app_setting_from_uuid()
            

    with st.expander("Replicate API Keys:"):
        replicate_user_name = st.text_input("replicate_user_name", value = app_settings["replicate_user_name"])
        replicate_com_api_key = st.text_input("replicate_com_api_key", value = app_settings["replicate_com_api_key"])
        if st.button("Save Settings"):
            data_repo.update_app_setting(replicate_user_name=replicate_user_name)
            data_repo.update_app_setting(replicate_com_api_key=replicate_com_api_key)
            # data_repo.update_app_setting("aws_access_key_id=aws_access_key_id)
            # data_repo.update_app_setting("aws_secret_access_key=aws_secret_access_key)
            st.experimental_rerun()

    with st.expander("Reset Welcome Sequence"):
        st.write("This will reset the welcome sequence so you can see it again.")
        if st.button("Reset Welcome Sequence"):
            st.session_state["welcome_state"] = 0
            data_repo.update_app_setting(welcome_state=0)
            st.experimental_rerun()

    locally_or_hosted = st.radio("Do you want to store your files locally or on AWS?", ("Locally", "AWS"),disabled=True, help="Only local storage is available at the moment, let me know if you need AWS storage - it should be pretty easy.")
    
    if locally_or_hosted == "AWS":
        with st.expander("AWS API Keys:"):
            aws_access_key_id = st.text_input("aws_access_key_id", value = app_settings["aws_access_key_id"])
            aws_secret_access_key = st.text_input("aws_secret_access_key", value = app_settings["aws_secret_access_key"])
                    