import streamlit as st
from repository.local_repo.csv_repo import update_project_setting,get_project_settings
from utils.data_repo.data_repo import DataRepo


# TODO: custom elements must be stateless and completely separate from our code logic
def radio(label, options, index=0, key=None, help=None, on_change=None, disabled=False, horizontal=False, label_visibility="visible", default_value=0, project_settings=None):    
    data_repo = DataRepo()

    if key not in st.session_state:
        if project_settings[key] == None:
            st.session_state[key] = default_value
        else:
            st.session_state[key] = options.index(project_settings[key])
                                                              
    selection = st.radio(label=label, options=options, index=st.session_state[key], key=key, help=help, on_change=on_change, disabled=disabled, horizontal=horizontal, label_visibility=label_visibility)

    if options.index(selection) != st.session_state[key]:
        st.session_state[key] = options.index(selection)
        if project_settings[key] != None:
            data_repo.update_project_setting(project_settings.project.uuid, key=selection)                                    
        st.experimental_rerun()
        
    return selection


def number_input(label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible", default_value=0, project_settings=None):
    data_repo = DataRepo()

    if key not in st.session_state:
        if project_settings[key] != None:
            st.session_state[key] = int(project_settings[key])
        else:
            st.session_state[key] = default_value

    selection = st.number_input(label, min_value, max_value, st.session_state[key], step, format, key, help, on_change, disabled, label_visibility)

    if selection != st.session_state[key]:
        st.session_state[key] = selection
        if project_settings[key] != None:
            data_repo.update_project_setting(project_settings.project.uuid, key=value)
        st.experimental_rerun()

    return selection


