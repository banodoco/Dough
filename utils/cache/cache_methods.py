import streamlit as st
from utils.cache.cache import StCache

from utils.enum import ExtendedEnum


class CacheKey(ExtendedEnum):
    USER = 'user'
    FILE = 'file'
    AI_MODEL = 'model'
    TIMING = 'timing'
    PROJECT = 'project'
    INFERENCE_LOG = 'inference_log'
    APP_SETTING = 'app_setting'
    APP_SECRET = 'app_secret'
    PROJECT_SETTING = 'project_setting'
    BACKUP = 'backup'

def cache_data(cls):
    class_methods = [method for method in dir(cls) if callable(getattr(cls, method))]

    # Apply cache decorator to class methods
    for method_name in class_methods:
        # skipping magic methods
        if method_name.startswith('__') and method_name.endswith('__'):
            continue

        cache_method_name = f"_cache_{method_name}"
        cache_method = getattr(cls, cache_method_name, None)

        if cache_method:
            setattr(cls, method_name, cache_method(method_name))


    def _cache_create_user(create_user):
        def _cache_create_user_wrapper(self, *args, **kwargs):
            user = create_user(*args, **kwargs)
            if user:
                StCache.add(user.uuid, user, CacheKey.USER.value)
            
            return user
            
        return _cache_create_user_wrapper
    
    def _cache_get_first_active_user(get_first_active_user):
        def _cache_get_first_active_user_wrapper(self, *args, **kwargs):
            user_list = StCache.get_all(CacheKey.USER.value)
            if user_list and len(user_list):
                return user_list[0]
            
            user = get_first_active_user(*args, **kwargs)
            if user:
                StCache.add(user, CacheKey.USER.value)
            
            return user
            
        return _cache_get_first_active_user_wrapper
    
    def _cache_get_user_by_email(get_user_by_email):
        def _cache_get_user_by_email_wrapper(self, *args, **kwargs):
            user_list = StCache.get_all(CacheKey.USER.value)
            if user_list and len(user_list):
                for user in user_list:
                    if user.email == args[0]:
                        return user
            
            
            user = get_user_by_email(*args, **kwargs)
            if user:
                StCache.add(user.uuid, user, CacheKey.USER.value)
            
            return user
            
        return _cache_get_user_by_email_wrapper

    def _cache_get_total_user_count(get_total_user_count):
        def _cache_get_total_user_count_wrapper(self, *args, **kwargs):
            user_list = StCache.get_all(CacheKey.USER.value)
            if user_list and len(user_list):
                return len(user_list)
            
            return get_total_user_count(*args, **kwargs)
            
        return _cache_get_total_user_count_wrapper
    
    def _cache_get_all_user_list(get_all_user_list):
        def _cache_get_all_user_list_wrapper(self, *args, **kwargs):
            user_list = StCache.get_all(CacheKey.USER.value)
            if user_list and len(user_list):
                return user_list
            
            user_list = get_all_user_list(*args, **kwargs)
            StCache.delete_all(CacheKey.USER.value)
            StCache.add_all(user_list, CacheKey.USER.value)

            return user_list
            
        return _cache_get_all_user_list_wrapper
    

    def _cache_delete_user_by_email(delete_user_by_email):
        def _cache_delete_user_by_email_wrapper(self, *args, **kwargs):
            status = delete_user_by_email(*args, **kwargs)
            
            if status:
                user_list = StCache.get_all(CacheKey.USER.value)
                if user_list and len(user_list):
                    for user in user_list:
                        if user.email == args[0]:
                            StCache.delete(user.uuid, CacheKey.USER.value)
            
            
        return _cache_delete_user_by_email_wrapper
    

    def _cache_get_file_from_name(get_file_from_name):
        def _cache_get_file_from_name_wrapper(self, *args, **kwargs):
            file_list = StCache.get_all(CacheKey.FILE.value)
            if file_list and len(file_list):
                for file in file_list:
                    if file.name == args[0]:
                        return file
            
            file = get_file_from_name(*args, **kwargs)
            if file:
                StCache.add(file.name, file, CacheKey.FILE.value)
            
            return file
            
        return _cache_get_file_from_name_wrapper
    

    def _cache_get_file_from_uuid(get_file_from_uuid):
        def _cache_get_file_from_uuid_wrapper(self, *args, **kwargs):
            file_list = StCache.get_all(CacheKey.FILE.value)
            if file_list and len(file_list):
                for file in file_list:
                    if file.uuid == args[0]:
                        return file
            
            file = get_file_from_uuid(*args, **kwargs)
            if file:
                StCache.add(file, CacheKey.FILE.value)
            
            return file
            
        return _cache_get_file_from_uuid_wrapper
    
    def _cache_get_all_file_list(get_all_file_list):
        def _cache_get_all_file_list_wrapper(self, *args, **kwargs):
            file_list = StCache.get_all(CacheKey.FILE.value)
            if file_list and len(file_list):
                return file_list
            
            file_list = get_all_file_list(*args, **kwargs)
            StCache.delete_all(CacheKey.FILE.value)
            StCache.add_all(file_list, CacheKey.FILE.value)

            return file_list
            
        return _cache_get_all_file_list_wrapper
    
    def _cache_create_or_update_file(create_or_update_file):
        def _cache_create_or_update_file_wrapper(self, *args, **kwargs):
            file = create_or_update_file(*args, **kwargs)
            if file:
                StCache.add(file, CacheKey.FILE.value)
            
            return file
            
        return _cache_create_or_update_file_wrapper
    
    def _cache_create_file(create_file):
        def _cache_create_file_wrapper(self, *args, **kwargs):
            file = create_file(*args, **kwargs)
            if file:
                StCache.add(file, CacheKey.FILE.value)
            
            return file
            
        return _cache_create_file_wrapper
    
    def _cache_delete_file_from_uuid(delete_file_from_uuid):
        def _cache_delete_file_from_uuid_wrapper(self, *args, **kwargs):
            status = delete_file_from_uuid(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.FILE.value)
        
        return _cache_delete_file_from_uuid_wrapper
    
    def _cache_get_image_list_from_uuid_list(get_image_list_from_uuid_list):
        def _cache_get_image_list_from_uuid_list_wrapper(self, *args, **kwargs):
            image_list = StCache.get_all(CacheKey.IMAGE.value)
            if image_list and len(image_list):
                res = []
                for image in image_list:
                    if image.uuid in args[0]:
                        res.append(image)
                return res
            
            image_list = get_image_list_from_uuid_list(*args, **kwargs)
            for img in image_list:
                StCache.add_all(img, CacheKey.IMAGE.value)

            return image_list
            
        return _cache_get_image_list_from_uuid_list
    
    def _cache_update_file(update_file):
        def _cache_update_file_wrapper(self, *args, **kwargs):
            file = update_file(*args, **kwargs)
            if file:
                StCache.add(file, CacheKey.FILE.value)
            
            return file
            
        return _cache_update_file_wrapper
    
    def _cache_get_project_from_uuid(get_project_from_uuid):
        def _cache_get_project_from_uuid_wrapper(self, *args, **kwargs):
            project_list = StCache.get_all(CacheKey.PROJECT.value)
            if project_list and len(project_list):
                for project in project_list:
                    if project.uuid == args[0]:
                        return project
            
            project = get_project_from_uuid(*args, **kwargs)
            if project:
                StCache.add(project, CacheKey.PROJECT.value)
            
            return project
            
        return _cache_get_project_from_uuid_wrapper
    
    def _cache_get_all_project_list(get_all_project_list):
        def _cache_get_all_project_list_wrapper(self, *args, **kwargs):
            project_list = StCache.get_all(CacheKey.PROJECT.value)
            if project_list and len(project_list):
                return project_list
            
            project_list = get_all_project_list(*args, **kwargs)
            StCache.delete_all(CacheKey.PROJECT.value)
            StCache.add_all(project_list, CacheKey.PROJECT.value)

            return project_list
            
        return _cache_get_all_project_list_wrapper
    
    def _cache_create_project(create_project):
        def _cache_create_project_wrapper(self, *args, **kwargs):
            project = create_project(*args, **kwargs)
            if project:
                StCache.add(project, CacheKey.PROJECT.value)
            
            return project
            
        return _cache_create_project_wrapper
    
    def _cache_delete_project_from_uuid(delete_project_from_uuid):
        def _cache_delete_project_from_uuid_wrapper(self, *args, **kwargs):
            status = delete_project_from_uuid(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.PROJECT.value)
        
        return _cache_delete_project_from_uuid_wrapper
    
    def _cache_get_ai_model_from_uuid(get_ai_model_from_uuid):
        def _cache_get_ai_model_from_uuid_wrapper(self, *args, **kwargs):
            ai_model_list = StCache.get_all(CacheKey.AI_MODEL.value)
            if ai_model_list and len(ai_model_list):
                for ai_model in ai_model_list:
                    if ai_model.uuid == args[0]:
                        return ai_model
            
            ai_model = get_ai_model_from_uuid(*args, **kwargs)
            if ai_model:
                StCache.add(ai_model, CacheKey.AI_MODEL.value)
            
            return ai_model
            
        return _cache_get_ai_model_from_uuid_wrapper
    
    def _cache_get_all_ai_model_list(get_all_ai_model_list):
        def _cache_get_all_ai_model_list_wrapper(self, *args, **kwargs):
            ai_model_list = StCache.get_all(CacheKey.AI_MODEL.value)
            if ai_model_list and len(ai_model_list):
                return ai_model_list
            
            ai_model_list = get_all_ai_model_list(*args, **kwargs)
            StCache.delete_all(CacheKey.AI_MODEL.value)
            StCache.add_all(ai_model_list, CacheKey.AI_MODEL.value)

            return ai_model_list
            
        return _cache_get_all_ai_model_list_wrapper
    
    def _cache_get_ai_model_from_name(get_ai_model_from_name):
        def _cache_get_ai_model_from_name_wrapper(self, *args, **kwargs):
            ai_model_list = StCache.get_all(CacheKey.AI_MODEL.value)
            if ai_model_list and len(ai_model_list):
                for ai_model in ai_model_list:
                    if ai_model.name == args[0]:
                        return ai_model
            
            ai_model = get_ai_model_from_name(*args, **kwargs)
            if ai_model:
                StCache.add(ai_model, CacheKey.AI_MODEL.value)
            
            return ai_model
            
        return _cache_get_ai_model_from_name_wrapper
    
    def _cache_create_ai_model(create_ai_model):
        def _cache_create_ai_model_wrapper(self, *args, **kwargs):
            ai_model = create_ai_model(*args, **kwargs)
            if ai_model:
                StCache.add(ai_model, CacheKey.AI_MODEL.value)
            
            return ai_model
            
        return _cache_create_ai_model_wrapper
    
    def _cache_update_ai_model(update_ai_model):
        def _cache_update_ai_model_wrapper(self, *args, **kwargs):
            ai_model = update_ai_model(*args, **kwargs)
            if ai_model:
                StCache.add(ai_model, CacheKey.AI_MODEL.value)
            
            return ai_model
            
        return _cache_update_ai_model_wrapper
    
    def _cache_delete_ai_model_from_uuid(delete_ai_model_from_uuid):
        def _cache_delete_ai_model_from_uuid_wrapper(self, *args, **kwargs):
            status = delete_ai_model_from_uuid(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.AI_MODEL.value)
        
        return _cache_delete_ai_model_from_uuid_wrapper
    
    def _cache_get_inference_log_from_uuid(get_inference_log_from_uuid):
        def _cache_get_inference_log_from_uuid_wrapper(self, *args, **kwargs):
            inference_log_list = StCache.get_all(CacheKey.INFERENCE_LOG.value)
            if inference_log_list and len(inference_log_list):
                for inference_log in inference_log_list:
                    if inference_log.uuid == args[0]:
                        return inference_log
            
            inference_log = get_inference_log_from_uuid(*args, **kwargs)
            if inference_log:
                StCache.add(inference_log, CacheKey.INFERENCE_LOG.value)
            
            return inference_log
            
        return _cache_get_inference_log_from_uuid_wrapper

    def  _cache_get_all_inference_log_list(get_all_inference_log_list):
        def _cache_get_all_inference_log_list_wrapper(self, *args, **kwargs):
            inference_log_list = StCache.get_all(CacheKey.INFERENCE_LOG.value)
            if inference_log_list and len(inference_log_list):
                return inference_log_list
            
            inference_log_list = get_all_inference_log_list(*args, **kwargs)
            StCache.delete_all(CacheKey.INFERENCE_LOG.value)
            StCache.add_all(inference_log_list, CacheKey.INFERENCE_LOG.value)

            return inference_log_list
            
        return _cache_get_all_inference_log_list_wrapper
    
    def _cache_create_inference_log(create_inference_log):
        def _cache_create_inference_log_wrapper(self, *args, **kwargs):
            inference_log = create_inference_log(*args, **kwargs)
            if inference_log:
                StCache.add(inference_log, CacheKey.INFERENCE_LOG.value)
            
            return inference_log
            
        return _cache_create_inference_log_wrapper
    
    def _cache_delete_inference_log_from_uuid(delete_inference_log_from_uuid):
        def _cache_delete_inference_log_from_uuid_wrapper(self, *args, **kwargs):
            status = delete_inference_log_from_uuid(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.INFERENCE_LOG.value)
        
        return _cache_delete_inference_log_from_uuid_wrapper
    
    def _cache_get_timing_from_uuid(get_timing_from_uuid):
        def _cache_get_timing_from_uuid_wrapper(self, *args, **kwargs):
            timing_list = StCache.get_all(CacheKey.TIMING.value)
            if timing_list and len(timing_list):
                for timing in timing_list:
                    if timing.uuid == args[0]:
                        return timing
            
            timing = get_timing_from_uuid(*args, **kwargs)
            if timing:
                StCache.add(timing, CacheKey.TIMING.value)
            
            return timing
            
        return _cache_get_timing_from_uuid_wrapper
    
    def _cache_get_timing_from_frame_number(get_timing_from_frame_number):
        def _cache_get_timing_from_frame_number_wrapper(self, *args, **kwargs):
            timing_list = StCache.get_all(CacheKey.TIMING.value)
            if timing_list and len(timing_list):
                for timing in timing_list:
                    if timing.aux_frame_number == args[0]:
                        return timing
            
            timing = get_timing_from_frame_number(*args, **kwargs)
            if timing:
                StCache.add(timing, CacheKey.TIMING.value)
            
            return timing
            
        return _cache_get_timing_from_frame_number_wrapper
    
    def _cache_get_next_timing(get_next_timing):
        def _cache_get_next_timing_wrapper(self, *args, **kwargs):
            timing_list = StCache.get_all(CacheKey.TIMING.value)
            if timing_list and len(timing_list):
                cur_timing = None
                for timing in timing_list:
                    if timing.aux_frame_number == args[0]:
                        cur_timing = timing
                
                if cur_timing:
                    for timing in timing_list:
                        if timing.aux_frame_number == cur_timing.aux_frame_number + 1:
                            return timing
            
            timing = get_next_timing(*args, **kwargs)
            if timing:
                StCache.add(timing, CacheKey.TIMING.value)
            
            return timing
            
        return _cache_get_next_timing_wrapper
    
    def _cache_get_prev_timing(get_prev_timing):
        def _cache_get_prev_timing_wrapper(self, *args, **kwargs):
            timing_list = StCache.get_all(CacheKey.TIMING.value)
            if timing_list and len(timing_list):
                cur_timing = None
                for timing in timing_list:
                    if timing.aux_frame_number == args[0]:
                        cur_timing = timing
                
                if cur_timing:
                    for timing in timing_list:
                        if timing.aux_frame_number == cur_timing.aux_frame_number - 1:
                            return timing
            
            timing = get_prev_timing(*args, **kwargs)
            if timing:
                StCache.add(timing, CacheKey.TIMING.value)
            
            return timing
            
        return _cache_get_prev_timing_wrapper
    
    def _cache_get_timing_list_from_project(get_timing_list_from_project):
        def _cache_get_timing_list_from_project_wrapper(self, *args, **kwargs):
            timing_list = StCache.get_all(CacheKey.TIMING.value)
            if timing_list and len(timing_list):
                return timing_list
            
            timing_list = get_timing_list_from_project(*args, **kwargs)
            StCache.delete_all(CacheKey.TIMING.value)
            StCache.add_all(timing_list, CacheKey.TIMING.value)

            return timing_list
            
        return _cache_get_timing_list_from_project_wrapper
    
    def _cache_create_timing(create_timing):
        def _cache_create_timing_wrapper(self, *args, **kwargs):
            timing = create_timing(*args, **kwargs)
            if timing:
                StCache.add(timing, CacheKey.TIMING.value)
            
            return timing
            
        return _cache_create_timing_wrapper
    
    def _cache_update_specific_timing(update_specific_timing):
        def _cache_update_specific_timing_wrapper(self, *args, **kwargs):
            status = update_specific_timing(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.TIMING.value)
        
        return _cache_update_specific_timing_wrapper
    
    def _cache_delete_timing_from_uuid(delete_timing_from_uuid):
        def _cache_delete_timing_from_uuid_wrapper(self, *args, **kwargs):
            status = delete_timing_from_uuid(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.TIMING.value)
        
        return _cache_delete_timing_from_uuid_wrapper
    
    def _cache_remove_existing_timing(remove_existing_timing):
        def _cache_remove_existing_timing_wrapper(self, *args, **kwargs):
            status = remove_existing_timing(*args, **kwargs)
            
            if status:
                StCache.delete_all(CacheKey.TIMING.value)
        
        return _cache_remove_existing_timing_wrapper
    
    def _cache_remove_primary_frame(remove_primary_frame):
        def _cache_remove_primary_frame_wrapper(self, *args, **kwargs):
            status = remove_primary_frame(*args, **kwargs)
            
            if status:
                timing = StCache.get(args[0], CacheKey.TIMING.value)
                timing.primary_image = None
                StCache.add(timing, CacheKey.TIMING.value)
        
        return _cache_remove_primary_frame_wrapper
    
    def _cache_remove_source_image(remove_source_image):
        def _cache_remove_source_image_wrapper(self, *args, **kwargs):
            status = remove_source_image(*args, **kwargs)
            
            if status:
                timing = StCache.get(args[0], CacheKey.TIMING.value)
                timing.source_image = None
                StCache.add(timing, CacheKey.TIMING.value)
        
        return _cache_remove_source_image_wrapper
    
    def _cache_move_frame_one_step_forward(move_frame_one_step_forward):
        def _cache_move_frame_one_step_forward_wrapper(self, *args, **kwargs):
            status = move_frame_one_step_forward(*args, **kwargs)
            
            if status:
                timing_list = StCache.get_all(CacheKey.TIMING.value)
                if timing_list and len(timing_list):
                    for timing in timing_list:
                        if timing.aux_frame_number >= args[1]:
                            timing.aux_frame_number += 1
                            StCache.add(timing, CacheKey.TIMING.value)
        
        return _cache_move_frame_one_step_forward_wrapper
    
    def _cache_get_app_setting_from_uuid(get_app_setting_from_uuid):
        def _cache_get_app_setting_from_uuid_wrapper(self, *args, **kwargs):
            app_setting_list = StCache.get_all(CacheKey.APP_SETTING.value)
            if app_setting_list and len(app_setting_list):
                for app_setting in app_setting_list:
                    if app_setting.uuid == args[0]:
                        return app_setting
            
            app_setting = get_app_setting_from_uuid(*args, **kwargs)
            if app_setting:
                StCache.add(app_setting, CacheKey.APP_SETTING.value)
            
            return app_setting
            
        return _cache_get_app_setting_from_uuid_wrapper
    
    def _cache_get_app_secrets_from_user_uuid(get_app_secrets_from_user_uuid):
        def _cache_get_app_secrets_from_user_uuid_wrapper(self, *args, **kwargs):
            app_secrets_list = StCache.get_all(CacheKey.APP_SECRET.value)
            if app_secrets_list and len(app_secrets_list):
                for app_secret in app_secrets_list:
                    if app_secret.user_uuid == args[0]:
                        return app_secret
            
            app_secrets = get_app_secrets_from_user_uuid(*args, **kwargs)
            if app_secrets:
                StCache.add_all(app_secrets, CacheKey.APP_SECRET.value)
            
            return app_secrets
            
        return _cache_get_app_secrets_from_user_uuid_wrapper
    
    def _cache_get_all_app_setting_list(get_all_app_setting_list):
        def _cache_get_all_app_setting_list_wrapper(self, *args, **kwargs):
            app_setting_list = StCache.get_all(CacheKey.APP_SETTING.value)
            if app_setting_list and len(app_setting_list):
                return app_setting_list
            
            app_setting_list = get_all_app_setting_list(*args, **kwargs)
            StCache.delete_all(CacheKey.APP_SETTING.value)
            StCache.add_all(app_setting_list, CacheKey.APP_SETTING.value)

            return app_setting_list
            
        return _cache_get_all_app_setting_list_wrapper
    
    def _cache_update_app_setting(update_app_setting):
        def _cache_update_app_setting_wrapper(self, *args, **kwargs):
            status = update_app_setting(*args, **kwargs)
            
            if status:
                StCache.add(kwargs['uuid'], CacheKey.APP_SETTING.value)
        
        return _cache_update_app_setting_wrapper
    
    def _cache_create_app_setting(create_app_setting):
        def _cache_create_app_setting_wrapper(self, *args, **kwargs):
            app_setting = create_app_setting(*args, **kwargs)
            if app_setting:
                StCache.add(app_setting, CacheKey.APP_SETTING.value)
            
            return app_setting
            
        return _cache_create_app_setting_wrapper
    
    def _cache_delete_app_setting(delete_app_setting):
        def _cache_delete_app_setting_wrapper(self, *args, **kwargs):
            status = delete_app_setting(*args, **kwargs)
            
            if status:
                # invalidate app setting object
                StCache.delete(args[0], CacheKey.APP_SETTING.value)
        
        return _cache_delete_app_setting_wrapper
    
    def _cache_get_project_setting(get_project_setting):
        def _cache_get_project_setting_wrapper(self, *args, **kwargs):
            project_setting = StCache.get(args[0], CacheKey.PROJECT_SETTING.value)
            if project_setting:
                return project_setting
            
            project_setting = get_project_setting(*args, **kwargs)
            if project_setting:
                StCache.add(project_setting, CacheKey.PROJECT_SETTING.value)
            
            return project_setting
            
        return _cache_get_project_setting_wrapper
    
    def _cache_create_project_setting(create_project_setting):
        def _cache_create_project_setting_wrapper(self, *args, **kwargs):
            project_setting = create_project_setting(*args, **kwargs)
            if project_setting:
                StCache.add(project_setting, CacheKey.PROJECT_SETTING.value)
            
            return project_setting
            
        return _cache_create_project_setting_wrapper
    
    def _cache_update_project_setting(update_project_setting):
        def _cache_update_project_setting_wrapper(self, *args, **kwargs):
            status = update_project_setting(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.PROJECT_SETTING.value)
        
        return _cache_update_project_setting_wrapper
    
    def _cache_bulk_update_project_setting(bulk_update_project_setting):
        def _cache_bulk_update_project_setting_wrapper(self, *args, **kwargs):
            status = bulk_update_project_setting(*args, **kwargs)
            
            if status:
                # invalidate project setting object
                StCache.add_all(kwargs['uuid'], CacheKey.PROJECT_SETTING.value)
        
        return _cache_bulk_update_project_setting_wrapper
    
    def _cache_get_backup_from_uuid(get_backup_from_uuid):
        def _cache_get_backup_from_uuid_wrapper(self, *args, **kwargs):
            backup_list = StCache.get_all(CacheKey.BACKUP.value)
            if backup_list and len(backup_list):
                for backup in backup_list:
                    if backup.uuid == args[0]:
                        return backup
            
            backup = get_backup_from_uuid(*args, **kwargs)
            if backup:
                StCache.add(backup.uuid, backup, CacheKey.BACKUP.value)
            
            return backup
            
        return _cache_get_backup_from_uuid_wrapper
    
    def _cache_create_backup(create_backup):
        def _cache_create_backup_wrapper(self, *args, **kwargs):
            backup = create_backup(*args, **kwargs)
            if backup:
                StCache.add(backup, CacheKey.BACKUP.value)
            
            return backup
            
        return _cache_create_backup_wrapper
    
    def _cache_get_backup_list(get_backup_list):
        def _cache_get_backup_list_wrapper(self, *args, **kwargs):
            backup_list = StCache.get_all(CacheKey.BACKUP.value)
            if backup_list and len(backup_list):
                return backup_list
            
            backup_list = get_backup_list(*args, **kwargs)
            StCache.delete_all(CacheKey.BACKUP.value)
            StCache.add_all(backup_list, CacheKey.BACKUP.value)

            return backup_list
            
        return _cache_get_backup_list_wrapper
    
    def _cache_delete_backup(delete_backup):
        def _cache_delete_backup_wrapper(self, *args, **kwargs):
            status = delete_backup(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.BACKUP.value)
        
        return _cache_delete_backup_wrapper
    
    def _cache_restore_backup(restore_backup):
        def _cache_restore_backup_wrapper(self, *args, **kwargs):
            status = restore_backup(*args, **kwargs)
            
            if status:
                StCache.delete(args[0], CacheKey.BACKUP.value)
        
        return _cache_restore_backup_wrapper

    return cls