import streamlit as st
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from utils.data_repo.data_repo import DataRepo

logger = AppLogger()

# TODO: custom elements must be stateless and completely separate from our code logic
def radio(label, options, index=0, key=None, help=None, on_change=None, disabled=False, horizontal=False, label_visibility="visible", default_value=0):    
    
    if key not in st.session_state:        
        st.session_state[key] = default_value                

    selection = st.radio(label=label, options=options, index=st.session_state[key], horizontal=horizontal, label_visibility=label_visibility)

    if options.index(selection) != st.session_state[key]:
        st.session_state[key] = options.index(selection)
        st.experimental_rerun()
        
    return selection


def number_input(label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible", default_value=0, project_settings=None):
    if key not in st.session_state:
        if getattr(project_settings, key, default_value):
            st.session_state[key] = int(getattr(project_settings, key, default_value))
        else:
            st.session_state[key] = default_value

    selection = st.number_input(label, min_value, max_value, st.session_state[key], step, format, key, help, on_change, disabled, label_visibility)

    if selection != st.session_state[key]:
        st.session_state[key] = selection
        if getattr(project_settings, key, default_value):
            data_repo = DataRepo()
            data_repo.update_project_setting(project_settings.project.uuid, key=value)
        st.experimental_rerun()

    return selection


def select_slider(label, options=(), value=None, format_func=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible", default_value=None, project_settings=None):
    if key not in st.session_state:
        if getattr(project_settings, key, default_value):
            st.session_state[key] = getattr(project_settings, key, default_value)
        else:
            st.session_state[key] = default_value

    selection = st.select_slider(label, options, st.session_state[key], format_func, key, help, on_change, disabled, label_visibility)

    if selection != st.session_state[key]:
        st.session_state[key] = selection
        if getattr(project_settings, key, default_value):
            data_repo = DataRepo()
            data_repo.update_project_setting(project_settings.project.uuid, key=value)
        st.experimental_rerun()

    return selection



def toggle(label, key=None, help=None, on_change=None, disabled=False, label_visibility="visible", default_value=False):
    if key not in st.session_state:
        st.session_state[key] = default_value

    selection = st.toggle(label=label, value=st.session_state[key], help=help, on_change=on_change, disabled=disabled, label_visibility=label_visibility)

    if selection != st.session_state[key]:
        st.session_state[key] = selection
        st.experimental_rerun()

    return selection