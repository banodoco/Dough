# no persistent storage present for streamlit at the moment, so storing things in the url. will update this asap
import streamlit as st


def get_url_param(key):
    params = st.experimental_get_query_params()
    val = params.get(key)
    if isinstance(val, list):
        res = val[0]
    else:
        res = val
    
    if not res and ( key in st.session_state and st.session_state[key]):
        set_url_param(key, st.session_state[key])
        return st.session_state[key]
    return res

def set_url_param(key, value):
    st.session_state[key] = value
    st.experimental_set_query_params(**{key: [value]})


def delete_url_param(key):
    print("deleting key: ", key)
    if key in st.session_state:
        del st.session_state[key]

    st.experimental_set_query_params(**{key: None})
