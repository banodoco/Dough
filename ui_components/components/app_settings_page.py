import streamlit as st

from repository.local_repo.csv_data import get_app_settings, update_app_settings

def app_settings_page():
    app_settings = get_app_settings()
            

    with st.expander("Replicate API Keys:"):
        replicate_user_name = st.text_input("replicate_user_name", value = app_settings["replicate_user_name"])
        replicate_com_api_key = st.text_input("replicate_com_api_key", value = app_settings["replicate_com_api_key"])
        if st.button("Save Settings"):
            update_app_settings("replicate_user_name", replicate_user_name)
            update_app_settings("replicate_com_api_key", replicate_com_api_key)
            # update_app_settings("aws_access_key_id", aws_access_key_id)
            # update_app_settings("aws_secret_access_key", aws_secret_access_key)
            st.experimental_rerun()

    with st.expander("Reset Welcome Sequence"):
        st.write("This will reset the welcome sequence so you can see it again.")
        if st.button("Reset Welcome Sequence"):
            st.session_state["welcome_state"] = 0
            update_app_settings("welcome_state", 0)
            st.experimental_rerun()

    locally_or_hosted = st.radio("Do you want to store your files locally or on AWS?", ("Locally", "AWS"),disabled=True, help="Only local storage is available at the moment, let me know if you need AWS storage - it should be pretty easy.")
    
    if locally_or_hosted == "AWS":
        with st.expander("AWS API Keys:"):
            aws_access_key_id = st.text_input("aws_access_key_id", value = app_settings["aws_access_key_id"])
            aws_secret_access_key = st.text_input("aws_secret_access_key", value = app_settings["aws_secret_access_key"])
                    