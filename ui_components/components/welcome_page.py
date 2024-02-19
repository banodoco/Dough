from utils.data_repo.data_repo import DataRepo
import streamlit as st

def welcome_page():
    st.success("Welcome!")
    
    if st.button("Next", key="welcome_cta"):
        data_repo = DataRepo()
        data_repo.update_app_setting(welcome_state=1)
        _ = data_repo.get_app_setting_from_uuid()
        st.rerun()