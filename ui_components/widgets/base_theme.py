import time
import streamlit as st
from utils.state_refresh import refresh_app


class BaseTheme:
    @staticmethod
    def success_msg(msg):
        st.success(msg)
        time.sleep(0.5)
        refresh_app()

    @staticmethod
    def error_msg(msg):
        st.error(msg)
        time.sleep(0.5)
        refresh_app()
