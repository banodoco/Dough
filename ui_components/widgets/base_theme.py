import time
import streamlit as st


class BaseTheme:
    @staticmethod
    def success_msg(msg):
        st.success(msg)
        time.sleep(0.5)
        st.rerun()

    @staticmethod
    def error_msg(msg):
        st.error(msg)
        time.sleep(0.5)
        st.rerun()
