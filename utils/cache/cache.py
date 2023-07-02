import streamlit as st
import threading
from functools import wraps


def synchronized(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        lock = threading.Lock()
        with lock:
            return func(*args, **kwargs)
    return wrapper

def synchronized_class(cls):
    class SynchronizedClass:
        def __init__(self, *args, **kwargs):
            self._lock = threading.Lock()
            self._instance = cls(*args, **kwargs)

        def __getattr__(self, name):
            with self._lock:
                print("____________ fetching: ", name)
                attribute = getattr(self._instance, name)
            return attribute

        def __setattr__(self, name, value):
            if name in ("_lock", "_instance"):
                super().__setattr__(name, value)
            else:
                with self._lock:
                    setattr(self._instance, name, value)

        def __delattr__(self, name):
            with self._lock:
                delattr(self._instance, name)

    return SynchronizedClass

class StCache:
    @staticmethod
    def get(uuid, data_type):
        if data_type in st.session_state:
            for ele in st.session_state[data_type]:
                if ele.uuid == uuid:
                    return ele
        
        return None
    
    @staticmethod
    def update(data, data_type) -> bool:
        object_found = False

        if data_type in st.session_state:
            object_list = st.session_state[data_type]
            for ele in object_list:
                if ele.uuid == data.uuid:
                    ele = data
                    object_found = True
                    break
            
            st.session_state[data_type] = object_list

        return object_found
    
    @staticmethod
    def add(data, data_type) -> bool:
        obj = StCache.get(data.uuid, data_type)
        if obj:
            StCache.update(data, data_type)
        else:
            if data_type in st.session_state:
                object_list = st.session_state[data_type]
            else:
                object_list = []
            
            object_list.append(data)

            st.session_state[data_type] = object_list
            current_thread = threading.current_thread()
            print("Current thread before:", current_thread.ident)
            print("Current thread before name:", current_thread.name)
            current_thread = threading.current_thread()
            print("Current thread after:", current_thread.ident)
            print("session state key length: ", len(st.session_state.keys()))


    @staticmethod
    def delete(uuid, data_type) -> bool:
        object_found = False
        if data_type in st.session_state:
            object_list = st.session_state[data_type]
            for ele in object_list:
                if ele.uuid == uuid:
                    object_list.remove(ele)
                    object_found = True
                    break
            
            st.session_state[data_type] = object_list
        
        return object_found
    
    @staticmethod
    def delete_all(data_type) -> bool:
        if data_type in st.session_state:
            del st.session_state[data_type]
            return True
        
        return False
    
    @staticmethod
    def add_all(data_list, data_type) -> bool:
        object_list = []
        if data_type in st.session_state:
            object_list = st.session_state[data_type]
        
        object_list.extend(data_list)
        st.session_state[data_type] = object_list

        
        return True
        
    @staticmethod
    def get_all(data_type):
        if data_type in st.session_state:
            return st.session_state[data_type]
        
        return []