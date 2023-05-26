import streamlit as st
from repository.local_repo.csv_repo import update_project_setting,get_project_settings



def radio(label, options, index=0, key=None, help=None, on_change=None, disabled=False, horizontal=False, label_visibility="visible", default_value=0,project_name=None, project_settings=None):    

    if f'{key}_value' not in st.session_state:
        if f'{key}' in project_settings:
            if project_settings[f"{key}"] == "":
                st.session_state[f'{key}_value'] = default_value
            else:
                st.session_state[f'{key}_value'] = options.index(project_settings[f'{key}_value'])
        else:
            st.session_state[f'{key}_value'] = default_value
                                                              
    selection = st.radio(label=label, options=options, index=st.session_state[f'{key}_value'], key=key, help=help, on_change=on_change, disabled=disabled, horizontal=horizontal, label_visibility=label_visibility)

    if options.index(selection) != st.session_state[f'{key}_value']:
        st.session_state[f'{key}_value'] = options.index(selection)
        if f'{key}_value' in project_settings:
            update_project_setting(f'{key}_value', selection, project_name)                                    
        st.experimental_rerun()
        
    return selection


def number_input(label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible", default_value=0,project_name=None, project_settings=None):
    
    if f'{key}_value' not in st.session_state:
        if f'{key}_value' in project_settings:
            if project_settings[f'{key}_value'] != "":
                st.session_state[f'{key}_value'] = int(project_settings[f'{key}_value'])
            else:
                st.session_state[f'{key}_value'] = default_value
        else:
            st.session_state[f'{key}_value'] = default_value
    else:
        st.session_state[f'{key}_value'] = st.session_state[f'{key}_value']

    selection = st.number_input(label, min_value, max_value, st.session_state[f'{key}_value'], step, format, key, help, on_change, disabled, label_visibility)

    if selection != st.session_state[f'{key}_value']:
        st.session_state[f'{key}_value'] = selection
        if f'{key}' in project_settings:
            update_project_setting(f'{key}_value', value, project_name)
        st.experimental_rerun()

    return selection