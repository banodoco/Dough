# this repo serves as a middlerware between API backend and the frontend
import threading
from shared.constants import InternalFileType, InternalResponse
from backend.db_repo import DBRepo
from shared.constants import SERVER, ServerType
from ui_components.models import InferenceLogObject, InternalAIModelObject, InternalAppSettingObject, InternalBackupObject, InternalFrameTimingObject, InternalProjectObject, InternalFileObject, InternalSettingObject, InternalUserObject
from utils.common_decorators import count_calls
import streamlit as st
import wrapt

from utils.data_repo.api_repo import APIRepo

# @cache_data
class DataRepo:
    def __init__(self):
        if SERVER == ServerType.DEVELOPMENT.value:
            self.db_repo = DBRepo()
        else:
            self.db_repo = APIRepo()
    
    # TODO: make a dummy user login in the local db repo
    def google_user_login(self, **kwargs):
        data = self.db_repo.google_user_login(**kwargs).data['data']
        user = InternalUserObject(**data['user']) if data and data['user'] else None
        token = data['token'] if data and data['token'] else None
        refresh_token = data['refresh_token'] if data and data['refresh_token'] else None
        return user, token, refresh_token

    def create_user(self, **kwargs):
        user = self.db_repo.create_user(**kwargs).data['data']
        return InternalUserObject(**user) if user else None
    
    def get_first_active_user(self):
        res: InternalResponse = self.db_repo.get_first_active_user()
        user = res.data['data'] if res.status else None
        return InternalUserObject(**user) if user else None
    
    def get_user_by_email(self, email):
        user = self.db_repo.get_user_by_email(email).data['data']
        return InternalUserObject(**user) if user else None
    
    def get_total_user_count(self):
        return self.db_repo.get_total_user_count().data
    
    def get_all_user_list(self):
        user_list = self.db_repo.get_all_user_list().data['data']
        return [InternalUserObject(**user) for user in user_list] if user_list else None
    
    def delete_user_by_email(self, email):
        res = self.db_repo.delete_user_by_email(email)
        return res.status

    # internal file object
    def get_file_from_name(self, name):
        file = self.db_repo.get_file_from_name(name).data['data']
        return InternalFileObject(**file) if file else None

    def get_file_from_uuid(self, uuid):
        file = self.db_repo.get_file_from_uuid(uuid).data['data']
        return InternalFileObject(**file) if file else None
    
    def get_all_file_list(self, file_type: InternalFileType, tag = None, project_id = None):
        filter_data = {"type": file_type}
        if tag:
            filter_data['tag'] = tag
        if project_id:
            filter_data['project_id'] = project_id

        res = self.db_repo.get_all_file_list(**filter_data)
        file_list = res.data['data'] if res.status else None
        
        return [InternalFileObject(**file) for file in file_list] if file_list else []
    
    def create_or_update_file(self, uuid, type=InternalFileType.IMAGE.value, **kwargs):
        file = self.db_repo.create_or_update_file(uuid, type, **kwargs).data['data']
        return InternalFileObject(**file) if file else None
    
    def upload_file(self, file_content, ext):
        res = self.db_repo.upload_file(file_content, ext)
        file_url = res.data['data'] if res.status else None
        return file_url

    def create_file(self, **kwargs):
        if 'hosted_url' not in kwargs and SERVER != ServerType.DEVELOPMENT.value:
            file_content = ('file', open(kwargs['local_path'], 'rb'))
            uploaded_file_url = self.upload_file(file_content)
            kwargs.update({'hosted_url':uploaded_file_url})

        res = self.db_repo.create_file(**kwargs)
        file = res.data['data'] if res.status else None
        return InternalFileObject(**file) if file else None
    
    def delete_file_from_uuid(self, uuid):
        res = self.db_repo.delete_file_from_uuid(uuid)
        return res.status
    
    def get_image_list_from_uuid_list(self, image_uuid_list, file_type=InternalFileType.IMAGE.value):
        if not (image_uuid_list and len(image_uuid_list)):
            return []
        image_list = self.db_repo.get_image_list_from_uuid_list(image_uuid_list, file_type=file_type).data['data']
        return [InternalFileObject(**image) for image in image_list] if image_list else []
    
    def update_file(self, file_uuid, **kwargs):
        # TODO: we are updating hosted_url whenever local_path is updated but we 
        # are not checking if the local_path is a different one - handle this correctly
        if 'local_path' in kwargs and SERVER != ServerType.DEVELOPMENT.value:
            file_content = ('file', open(kwargs['local_path'], 'rb'))
            uploaded_file_url = self.upload_file(file_content)
            kwargs.update({'hosted_url':uploaded_file_url})

        file = self.db_repo.update_file(uuid=file_uuid, **kwargs).data['data']
        return InternalFileObject(**file) if file else None
    
    # project
    def get_project_from_uuid(self, uuid):
        project = self.db_repo.get_project_from_uuid(uuid).data['data']
        return InternalProjectObject(**project) if project else None
    
    def get_all_project_list(self, user_id):
        project_list = self.db_repo.get_all_project_list(user_id).data['data']
        return [InternalProjectObject(**project) for project in project_list] if project_list else None
    
    def create_project(self, **kwargs):
        project = self.db_repo.create_project(**kwargs).data['data']
        return InternalProjectObject(**project) if project else None
    
    def delete_project_from_uuid(self, uuid):
        res = self.db_repo.delete_project_from_uuid(uuid)
        return res.status
    
    def update_project(self, **kwargs):
        project = self.db_repo.update_project(**kwargs).data['data']
        return InternalProjectObject(**project) if project else None
    
    # ai model (custom ai model)
    def get_ai_model_from_uuid(self, uuid):
        res = self.db_repo.get_ai_model_from_uuid(uuid)
        model = res.data['data'] if res.status else None
        return InternalAIModelObject(**model) if model else None
    
    def get_ai_model_from_name(self, name):
        model = self.db_repo.get_ai_model_from_name(name).data['data']
        return InternalAIModelObject(**model) if model else None
    
    def get_all_ai_model_list(self, model_type_list=None, user_id=None, custom_trained=None):
        from utils.common_utils import get_current_user_uuid
        if not user_id:
            user_id = get_current_user_uuid()

        model_list = self.db_repo.get_all_ai_model_list(model_type_list, user_id, custom_trained).data['data']
        return [InternalAIModelObject(**model) for model in model_list] if model_list else []
    
    def create_ai_model(self, **kwargs):
        model = self.db_repo.create_ai_model(**kwargs).data['data']
        return InternalAIModelObject(**model) if model else None
    
    def update_ai_model(self, **kwargs):
        model = self.db_repo.update_ai_model(**kwargs).data['data']
        return  InternalAIModelObject(**model) if model else None
    
    def delete_ai_model_from_uuid(self, uuid):
        res = self.db_repo.delete_ai_model_from_uuid(uuid)
        return res.status
    

    # inference log
    def get_inference_log_from_uuid(self, uuid):
        log = self.db_repo.get_inference_log_from_uuid(uuid).data['data']
        return InferenceLogObject(**log) if log else None
    
    def get_all_inference_log_list(self, project_id=None):
        log_list = self.db_repo.get_all_inference_log_list(project_id).data['data']
        return [InferenceLogObject(**log) for log in log_list] if log_list else None
    
    def create_inference_log(self, **kwargs):
        log = self.db_repo.create_inference_log(**kwargs).data['data']
        return InferenceLogObject(**log) if log else None
    
    def delete_inference_log_from_uuid(self, uuid):
        res = self.db_repo.delete_inference_log_from_uuid(uuid)
        return res.status
    

    # ai model param map
    # TODO: add DTO in the output
    def get_ai_model_param_map_from_uuid(self, uuid):
        pass
    
    def get_all_ai_model_param_map_list(self, model_id=None):
        pass
    
    def create_ai_model_param_map(self, **kwargs):
        pass
    
    def delete_ai_model(self, uuid):
        pass
    

    # timing
    def get_timing_from_uuid(self, uuid):
        timing = self.db_repo.get_timing_from_uuid(uuid).data['data']
        return InternalFrameTimingObject(**timing) if timing else None
    
    def get_timing_from_frame_number(self, project_uuid, frame_number):
        res = self.db_repo.get_timing_from_frame_number(project_uuid, frame_number)
        timing = res.data['data'] if res.status else None
        return InternalFrameTimingObject(**timing) if timing else None
    
    
    # this is based on the aux_frame_index and not the order in the db
    def get_next_timing(self, uuid):
        next_timing = self.db_repo.get_next_timing(uuid).data['data']
        return InternalFrameTimingObject(**next_timing) if next_timing else None
    
    def get_prev_timing(self, uuid):
        prev_timing = self.db_repo.get_prev_timing(uuid).data['data']
        return InternalFrameTimingObject(**prev_timing) if prev_timing else None
    
    def get_timing_list_from_project(self, project_uuid=None):
        res = self.db_repo.get_timing_list_from_project(project_uuid)
        timing_list = res.data['data'] if res.status else None
        return [InternalFrameTimingObject(**timing) for timing in timing_list] if timing_list else []
    
    def create_timing(self, **kwargs):
        res = self.db_repo.create_timing(**kwargs)
        timing = res.data['data'] if res.status else None
        return InternalFrameTimingObject(**timing) if timing else None
    
    def update_specific_timing(self, uuid, **kwargs):
        res = self.db_repo.update_specific_timing(uuid, **kwargs)
        return res.status
    
    def delete_timing_from_uuid(self, uuid):
        res = self.db_repo.delete_timing_from_uuid(uuid)
        return res.status
    
    # removes all timing frames from the project
    def remove_existing_timing(self, project_uuid):
        res = self.db_repo.remove_existing_timing(project_uuid)
        return res.status
    
    def remove_primay_frame(self, timing_uuid):
        res = self.db_repo.remove_primay_frame(timing_uuid)
        return res.status
    
    def remove_source_image(self, timing_uuid):
        res = self.db_repo.remove_source_image(timing_uuid)
        return res.status

    def move_frame_one_step_forward(self, project_uuid, index_of_frame):
        res = self.db_repo.move_frame_one_step_forward(project_uuid, index_of_frame)
        return res.status
    

    # app setting
    def get_app_setting_from_uuid(self, uuid=None):
        res = self.db_repo.get_app_setting_from_uuid(uuid)
        app_setting = res.data['data'] if res.status else None
        return InternalAppSettingObject(**app_setting) if app_setting else None
    
    def get_app_secrets_from_user_uuid(self, uuid=None):
        from utils.common_utils import get_current_user_uuid
        # if user is not defined then take the current user
        if not uuid:
            uuid = get_current_user_uuid()
        
        app_secrets = self.db_repo.get_app_secrets_from_user_uuid(uuid).data['data']
        return app_secrets
    
    def get_all_app_setting_list(self):
        app_setting_list = self.db_repo.get_all_app_setting_list().data['data']
        return [InternalAppSettingObject(**app_setting) for app_setting in app_setting_list] if app_setting_list else None
    
    def update_app_setting(self, **kwargs):
        res = self.db_repo.update_app_setting(**kwargs)
        return res.status
    
    def create_app_setting(self, **kwargs):
        app_setting = self.db_repo.create_app_setting(**kwargs).data['data']
        return InternalAppSettingObject(**app_setting) if app_setting else None

    def delete_app_setting(self, user_id):
        res = self.db_repo.delete_app_setting(user_id)
        return res.status
    

    # setting
    def get_project_setting(self, project_id):
        res = self.db_repo.get_project_setting(project_id)
        project_setting = res.data['data'] if res.status else None
        return InternalSettingObject(**project_setting) if project_setting else None
    
    # TODO: add valid model_id check throughout dp_repo
    def create_project_setting(self, **kwargs):
        res = self.db_repo.create_project_setting(**kwargs)
        project_setting = res.data['data'] if res.status else None
        return InternalSettingObject(**project_setting) if project_setting else None
    
    # TODO: remove db calls for updating guidance_type
    def update_project_setting(self, project_uuid, **kwargs):
        kwargs['project_id'] = project_uuid
        project_setting = self.db_repo.update_project_setting(**kwargs).data['data']
        return InternalSettingObject(**project_setting) if project_setting else None

    def bulk_update_project_setting(self, **kwargs):
        res = self.db_repo.bulk_update_project_setting(**kwargs)
        return res.status
    

    # backup
    def get_backup_from_uuid(self, uuid):
        backup = self.db_repo.get_backup_from_uuid(uuid).data['data']
        return InternalBackupObject(**backup) if backup else None
    
    def create_backup(self, project_uuid, version_name):
        backup = self.db_repo.create_backup(project_uuid, version_name).data['data']
        return InternalBackupObject(**backup) if backup else None
    
    def get_backup_list(self, project_id=None):
        backup_list = self.db_repo.get_backup_list(project_id).data['data']
        return [InternalBackupObject(**backup) for backup in backup_list] if backup_list else []
    
    def delete_backup(self, uuid):
        res = self.db_repo.delete_backup(uuid)
        return res.status
    
    def restore_backup(self, uuid):
        res = self.db_repo.restore_backup(uuid)
        return res.status
    
    # update user credits