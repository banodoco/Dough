import datetime
import streamlit as st
import extra_streamlit_components as stx

from utils.constants import AUTH_TOKEN

# NOTE: code not working properly. check again after patch from the streamlit team
# @st.cache(allow_output_mutation=True)
# def get_manager():
#     return stx.CookieManager()


# def get_cookie(key):
#     cookie_manager = get_manager()
#     return cookie_manager.get(cookie=key)

# def set_cookie(key, value):
#     cookie_manager = get_manager()
#     expiration_time = datetime.datetime.now() + datetime.timedelta(days=1)
#     cookie_manager.set(key, value, expires_at=expiration_time)

# def delete_cookie(key):
#     cookie_manager = get_manager()
#     cookie = get_cookie(key)
#     if cookie:
#         cookie_manager.delete(key)