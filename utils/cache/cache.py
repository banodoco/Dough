import streamlit as st

class StCache:
    @staticmethod
    def get(uuid, type):
        if type in st.session_state:
            for ele in st.session_state[type]:
                if ele['uuid'] == uuid:
                    return ele
        
        return None
    
    @staticmethod
    def update(data, type) -> bool:
        object_found = False

        if type in st.session_state:
            object_list = st.session_state[type]
            for ele in object_list:
                if ele['uuid'] == data.uuid:
                    ele = data
                    object_found = True
                    break
            
            st.session_state[type] = object_list

        return object_found
    
    @staticmethod
    def add(data, type) -> bool:
        obj = StCache.get(data.uuid, type)
        if obj:
            StCache.update(data, type)
        else:
            if type in st.session_state:
                object_list = st.session_state[type]
            else:
                object_list = []
            
            object_list.append(data)

            st.session_state[type] = object_list

        StCache.validate(type)

    @staticmethod
    def delete(uuid, type) -> bool:
        object_found = False
        if type in st.session_state:
            object_list = st.session_state[type]
            for ele in object_list:
                if ele['uuid'] == uuid:
                    object_list.remove(ele)
                    object_found = True
                    break
            
            st.session_state[type] = object_list
        
        return object_found
    
    @staticmethod
    def delete_all(type) -> bool:
        if type in st.session_state:
            del st.session_state[type]
            return True
        
        return False
    
    @staticmethod
    def add_all(data_list, type) -> bool:
        object_list = []
        if type in st.session_state:
            object_list = st.session_state[type]
        
        object_list.extend(data_list)
        st.session_state[type] = object_list

        StCache.validate(type)
        return True
        
    @staticmethod
    def get_all(type):
        if type in st.session_state:
            return st.session_state[type]
        
        return []