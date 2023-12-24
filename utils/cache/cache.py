import streamlit as st

from utils.enum import ExtendedEnum

class CacheKey(ExtendedEnum):
    TIMING_DETAILS = "timing_details"
    APP_SETTING = "app_setting"
    APP_SECRET = "app_secret"
    PROJECT_SETTING = "project_setting"
    AI_MODEL = "ai_model"
    LOGGED_USER = "logged_user"
    FILE = "file"
    SHOT = "shot"
    # temp items (only cached for speed boost)
    LOG = 'log'
    LOG_PAGES = 'log_pages'
    PROJECT = 'project'
    USER = 'user'


class StCache:
    @staticmethod
    def get(uuid, data_type):
        uuid = str(uuid)
        if data_type in st.session_state:
            for ele in st.session_state[data_type]:
                ele_uuid = ele['uuid'] if type(ele) is dict else str(ele.uuid)
                if ele_uuid == uuid:
                    return ele
        
        return None
    
    @staticmethod
    def update(data, data_type) -> bool:
        object_found = False
        uuid = data['uuid'] if type(data) is dict else data.uuid
        uuid = str(uuid)

        if data_type in st.session_state:
            object_list = st.session_state[data_type]
            for idx, ele in enumerate(object_list):
                ele_uuid = ele['uuid'] if type(ele) is dict else str(ele.uuid)
                if ele_uuid == uuid:
                    object_list[idx] = data
                    object_found = True
            
            st.session_state[data_type] = object_list

        return object_found
    
    @staticmethod
    def add(data, data_type) -> bool:
        uuid = data['uuid'] if type(data) is dict else data.uuid
        uuid = str(uuid)
        obj = StCache.get(uuid, data_type)
        if obj:
            StCache.update(data, data_type)
        else:
            if data_type in st.session_state:
                object_list = st.session_state[data_type]
            else:
                object_list = []
            
            object_list.append(data)

            st.session_state[data_type] = object_list


    @staticmethod
    def delete(uuid, data_type) -> bool:
        object_found = False
        uuid = str(uuid)
        if data_type in st.session_state:
            object_list = st.session_state[data_type]
            for ele in object_list:
                ele_uuid = ele['uuid'] if type(ele) is dict else str(ele.uuid)
                if ele_uuid == uuid:
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
        for data in data_list:
            StCache.add(data, data_type)
        
        return True
        
    @staticmethod
    def get_all(data_type):
        if data_type in st.session_state:
            return st.session_state[data_type]
        
        return []
    
    # deletes all cached objects of every data type
    @staticmethod
    def clear_entire_cache() -> bool:
        for c in CacheKey.value_list():
            StCache.delete_all(c)
        
        return True