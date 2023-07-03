
import os

import requests
import streamlit as st
from shared.constants import SERVER, InternalFileType, ServerType

from utils.constants import AUTH_DETAILS, LOGGED_USER


class APIRepo:
    def __init__(self):
        import dotenv
        dotenv.load_dotenv()

        SERVER_URL = os.getenv('SERVER_URL', '')
        self.base_url = SERVER_URL

        self._setup_urls()

    def _setup_url(self):
        # user
        self.USER_OP_URL = '/v1/user/op'
        self.USER_LIST_URL = '/v1/user/list'

        # payment
        self.ORDER_OP_URL = '/v1/payment/order'
        self.ORDER_LIST_URL = '/v1/payment/order/list'
        
        # auth
        self.AUTH_OP_URL = '/v1/authentication/op'
        self.AUTH_REFRESH_URL = '/v1/authentication/refresh'
        self.GOOGLE_LOGIN_URL = '/v1/authentication/google'

        # timing
        self.TIMING_URL = '/v1/data/timing'
        self.PROJECT_TIMING_URL = '/v1/data/timing/project'
        self.TIMING_NUMBER_URL = '/v1/data/timing/number'
        self.SHIFT_TIMING_URL = '/v1/data/timing/shift'
        self.TIMING_LIST_URL = '/v1/data/timing/list'

        # project
        self.PROJECT_URL = '/v1/data/project'
        self.PROJECT_LIST_URL = '/v1/data/project/list'
        
        # project setting
        self.PROJECT_SETTING_URL = '/v1/data/project-setting'
        
        # inference log
        self.LOG_URL = '/v1/data/log'
        self.LOG_LIST_URL = '/v1/data/log/list'
        
        # file
        self.FILE_URL = '/v1/data/file'
        self.FILE_LIST_URL = '/v1/data/file/list'
        self.FILE_UUID_LIST_URL = '/v1/data/file/uuid-list'
        
        # app setting
        self.APP_SETTING_URL = '/v1/data/app-setting'
        self.APP_SECRET_URL = '/v1/data/app-secret'
        
        # ai model
        self.MODEL_URL = '/v1/data/model'
        self.MODEL_LIST_URL = '/v1/data/model/list'

    def logout(self):
        pass

    ################### base http methods
    def _get_headers(self):
        if AUTH_DETAILS not in st.session_state and SERVER != ServerType.DEVELOPMENT.value:
            self.logout()

        headers = {}
        if AUTH_DETAILS in st.session_state and st.session_state[AUTH_DETAILS]:
            token = st.session_state[AUTH_DETAILS]['auth_token']
            headers["Authorization"] = f"Bearer {token}"
            headers["Content-Type"] = "application/json"

        return headers

    def http_get(self, url, params = None):
        res = requests.get(self.base_url + url, params = params, headers=self._get_headers())
        return res.json()

    def http_post(self, url, data = None):
        res = requests.post(self.base_url + url, data=data, headers=self._get_headers())
        return res.json()
    
    def http_put(self, url, data = None):
        res = requests.put(self.base_url + url, data=data, headers=self._get_headers())
        return res.json()
    
    def http_delete(self, url, params=None):
        res = requests.delete(self.base_url + url, params=params, headers=self._get_headers())
        return res.json()

    #########################################

    def create_user(self, **kwargs):
        res = self.http_post(url=self.USER_OP_URL, data=kwargs)
        return res
    
    # making it fetch the current logged in user
    def get_first_active_user(self):
        if LOGGED_USER not in st.session_state:
            return None
        
        logged_user = st.session_state[LOGGED_USER]
        res = self.http_get(self.USER_OP_URL, params={'uuid': logged_user['uuid']})
        return res
    
    def get_user_by_email(self, email):
        res = self.http_get(self.USER_OP_URL, params={'email': email})
        return res
    
    def get_total_user_count(self):
        res = self.http_get(self.USER_LIST_URL)
        payload = {
            'data': res['data']['total_count']
        }
        return payload
    
    def get_all_user_list(self):
        res = self.http_get(self.USER_LIST_URL)
        return res
    
    # TODO: remove this method from everywhere
    def delete_user_by_email(self, email):
        res = self.db_repo.delete_user_by_email(email)
        return res.status

    # internal file object
    # TODO: remove this method from everywhere
    def get_file_from_name(self, name):
        pass

    def get_file_from_uuid(self, uuid):
        res = self.http_get(self.FILE_URL, params={'uuid': uuid})
        return res
    
    def get_all_file_list(self, file_type: InternalFileType, tag = None, project_id = None):
        filter_data = {"type": file_type}
        if tag:
            filter_data['tag'] = tag
        if project_id:
            filter_data['project_id'] = project_id

        res = self.http_get(self.FILE_LIST_URL, params=filter_data)
        return res
    
    def create_or_update_file(self, uuid, type=InternalFileType.IMAGE.value, **kwargs):
        update_data = kwargs
        update_data['uuid'] = uuid
        update_data['type'] = type
        res = self.http_put(url=self.FILE_URL, data=update_data)
        return res
        
    def create_file(self, **kwargs):
        res = self.http_post(url=self.FILE_URL, data=kwargs)
        return res
    
    def delete_file_from_uuid(self, uuid):
        res = self.db_repo.delete_file_from_uuid(uuid)
        return res.status
    
    # TODO: remove file_type from this method
    def get_image_list_from_uuid_list(self, image_uuid_list, file_type=InternalFileType.IMAGE.value):
        res = self.http_get(self.FILE_UUID_LIST_URL, params={'uuid_list': image_uuid_list, 'type': file_type})
        return res
    
    def update_file(self, file_uuid, **kwargs):
        update_data = kwargs
        update_data['uuid'] = file_uuid
        res = self.http_put(url=self.FILE_URL, data=update_data)
        return res
    
    # project
    def get_project_from_uuid(self, uuid):
        res = self.http_get(self.PROJECT_URL, params={'uuid': uuid})
        return res
    
    def get_all_project_list(self, user_id):
        res = self.http_get(self.PROJECT_LIST_URL, params={'user_id': user_id})
        return res
    
    def create_project(self, **kwargs):
        res = self.http_post(url=self.PROJECT_URL, data=kwargs)
        return res
    
    def delete_project_from_uuid(self, uuid):
        res = self.http_delete(self.PROJECT_URL, params={'uuid': uuid})
        return res
    
    # ai model (custom ai model)
    def get_ai_model_from_uuid(self, uuid):
        res = self.http_get(self.MODEL_URL, params={'uuid': uuid})
        return res
    
    # TODO: remove this method from everywhere
    def get_ai_model_from_name(self, name):
        res = self.http_get(self.MODEL_URL, params={'name': name})

    
    # def get_all_ai_model_list(self, model_type=None, user_id=None):
    #     from utils.common_methods import get_current_user_uuid
    #     if not user_id:
    #         user_id = get_current_user_uuid()

    #     model_list = self.db_repo.get_all_ai_model_list(model_type, user_id).data['data']
    #     return [InternalAIModelObject(**model) for model in model_list] if model_list else []
    
    # def create_ai_model(self, **kwargs):
    #     model = self.db_repo.create_ai_model(**kwargs).data['data']
    #     return InternalAIModelObject(**model) if model else None
    
    # def update_ai_model(self, **kwargs):
    #     model = self.db_repo.update_ai_model(**kwargs).data['data']
    #     return  InternalAIModelObject(**model) if model else None
    
    # def delete_ai_model_from_uuid(self, uuid):
    #     res = self.db_repo.delete_ai_model_from_uuid(uuid)
    #     return res.status
    

    # # inference log
    # def get_inference_log_from_uuid(self, uuid):
    #     log = self.db_repo.get_inference_log_from_uuid(uuid).data['data']
    #     return InferenceLogObject(**log) if log else None
    
    # def get_all_inference_log_list(self, project_id=None):
    #     log_list = self.db_repo.get_all_inference_log_list(project_id).data['data']
    #     return [InferenceLogObject(**log) for log in log_list] if log_list else None
    
    # def create_inference_log(self, **kwargs):
    #     log = self.db_repo.create_inference_log(**kwargs).data['data']
    #     return InferenceLogObject(**log) if log else None
    
    # def delete_inference_log_from_uuid(self, uuid):
    #     res = self.db_repo.delete_inference_log_from_uuid(uuid)
    #     return res.status
    

    # # ai model param map
    # # TODO: add DTO in the output
    # def get_ai_model_param_map_from_uuid(self, uuid):
    #     pass
    
    # def get_all_ai_model_param_map_list(self, model_id=None):
    #     pass
    
    # def create_ai_model_param_map(self, **kwargs):
    #     pass
    
    # def delete_ai_model(self, uuid):
    #     pass
    

    # # timing
    # def get_timing_from_uuid(self, uuid):
    #     timing = self.db_repo.get_timing_from_uuid(uuid).data['data']
    #     return InternalFrameTimingObject(**timing) if timing else None
    
    # def get_timing_from_frame_number(self, project_uuid, frame_number):
    #     timing = self.db_repo.get_timing_from_frame_number(project_uuid, frame_number).data['data']
    #     return InternalFrameTimingObject(**timing) if timing else None
    
    
    # # this is based on the aux_frame_index and not the order in the db
    # def get_next_timing(self, uuid):
    #     next_timing = self.db_repo.get_next_timing(uuid).data['data']
    #     return InternalFrameTimingObject(**next_timing) if next_timing else None
    
    # def get_prev_timing(self, uuid):
    #     prev_timing = self.db_repo.get_prev_timing(uuid).data['data']
    #     return InternalFrameTimingObject(**prev_timing) if prev_timing else None
    
    # def get_timing_list_from_project(self, project_uuid=None):
    #     timing_list = self.db_repo.get_timing_list_from_project(project_uuid).data['data']
    #     return [InternalFrameTimingObject(**timing) for timing in timing_list] if timing_list else []
    
    # def create_timing(self, **kwargs):
    #     timing = self.db_repo.create_timing(**kwargs).data['data']
    #     return InternalFrameTimingObject(**timing) if timing else None
    
    # def update_specific_timing(self, uuid, **kwargs):
    #     res = self.db_repo.update_specific_timing(uuid, **kwargs)
    #     return res.status
    
    # def delete_timing_from_uuid(self, uuid):
    #     res = self.db_repo.delete_timing_from_uuid(uuid)
    #     return res.status
    
    # # removes all timing frames from the project
    # def remove_existing_timing(self, project_uuid):
    #     res = self.db_repo.remove_existing_timing(project_uuid)
    #     return res.status
    
    # def remove_primay_frame(self, timing_uuid):
    #     res = self.db_repo.remove_primay_frame(timing_uuid)
    #     return res.status
    
    # def remove_source_image(self, timing_uuid):
    #     res = self.db_repo.remove_source_image(timing_uuid)
    #     return res.status

    # def move_frame_one_step_forward(self, project_uuid, index_of_frame):
    #     res = self.db_repo.move_frame_one_step_forward(project_uuid, index_of_frame)
    #     return res.status
    

    # # app setting
    # def get_app_setting_from_uuid(self, uuid=None):
    #     app_setting = self.db_repo.get_app_setting_from_uuid(uuid).data['data']
    #     return InternalAppSettingObject(**app_setting) if app_setting else None
    
    # def get_app_secrets_from_user_uuid(self, uuid=None):
    #     from utils.common_methods import get_current_user_uuid
    #     # if user is not defined then take the current user
    #     if not uuid:
    #         uuid = get_current_user_uuid()
        
    #     app_secrets = self.db_repo.get_app_secrets_from_user_uuid(uuid).data['data']
    #     return app_secrets
    
    # def get_all_app_setting_list(self):
    #     app_setting_list = self.db_repo.get_all_app_setting_list().data['data']
    #     return [InternalAppSettingObject(**app_setting) for app_setting in app_setting_list] if app_setting_list else None
    
    # def update_app_setting(self, **kwargs):
    #     res = self.db_repo.update_app_setting(**kwargs)
    #     return res.status
    
    # def create_app_setting(self, **kwargs):
    #     app_setting = self.db_repo.create_app_setting(**kwargs).data['data']
    #     return InternalAppSettingObject(**app_setting) if app_setting else None

    # def delete_app_setting(self, user_id):
    #     res = self.db_repo.delete_app_setting(user_id)
    #     return res.status
    

    # # setting
    # def get_project_setting(self, project_id):
    #     project_setting = self.db_repo.get_project_setting(project_id).data['data']
    #     return InternalSettingObject(**project_setting) if project_setting else None
    
    # # TODO: add valid model_id check throughout dp_repo
    # def create_project_setting(self, **kwargs):
    #     project_setting = self.db_repo.create_project_setting(**kwargs).data['data']
    #     return InternalSettingObject(**project_setting) if project_setting else None
    
    # def update_project_setting(self, project_uuid, **kwargs):
    #     kwargs['uuid'] = project_uuid
    #     project_setting = self.db_repo.update_project_setting(**kwargs).data['data']
    #     return InternalSettingObject(**project_setting) if project_setting else None

    # def bulk_update_project_setting(self, **kwargs):
    #     res = self.db_repo.bulk_update_project_setting(**kwargs)
    #     return res.status
    

    # # backup
    # def get_backup_from_uuid(self, uuid):
    #     backup = self.db_repo.get_backup_from_uuid(uuid).data['data']
    #     return InternalBackupObject(**backup) if backup else None
    
    # def create_backup(self, project_uuid, version_name):
    #     backup = self.db_repo.create_backup(project_uuid, version_name).data['data']
    #     return InternalBackupObject(**backup) if backup else None
    
    # def get_backup_list(self, project_id=None):
    #     backup_list = self.db_repo.get_backup_list(project_id).data['data']
    #     return [InternalBackupObject(**backup) for backup in backup_list] if backup_list else []
    
    # def delete_backup(self, uuid):
    #     res = self.db_repo.delete_backup(uuid)
    #     return res.status
    
    # def restore_backup(self, uuid):
    #     res = self.db_repo.restore_backup(uuid)
    #     return res.status