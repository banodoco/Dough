# no persistent storage present for streamlit at the moment, so storing things in the url. will update this asap
import streamlit as st


def get_url_param(key):
    print("fetching url param: ", key)
    params = st.experimental_get_query_params()
    val = params.get(key)
    if isinstance(val, list):
        return val[0]
    return val


def set_url_param(key, value):
    print("setting param: ", key, value, len(value))
    st.experimental_set_query_params(**{key: [value]})


def set_only_url_param(key, value):
    print("setting param: ", key, value)
    params = st.experimental_get_query_params()
    for k, _ in params.items():
        st.experimental_set_query_params(**{k: None})

    st.experimental_set_query_params(**{key: value})


def delete_url_param(key):
    print("deleting key: ", key)
    st.experimental_set_query_params(**{key: None})
