from shared.logging.logging import AppLogger
from utils.cache.cache import StCache

from utils.enum import ExtendedEnum

logger = AppLogger()

class CacheKey(ExtendedEnum):
    TIMING_DETAILS = "timing_details"
    APP_SETTING = "app_setting"
    APP_SECRET = "app_secret"
    PROJECT_SETTING = "project_setting"

# NOTE: caching only timing_details, project settings and app settings. invalidating cache everytime a related data is updated
def cache_data(cls):
    def _cache_create_or_update_file(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_or_update_file')
        file = original_func(self, *args, **kwargs)
        StCache.delete_all(CacheKey.TIMING_DETAILS.value)
        
        return file
    
    setattr(cls, '_original_create_or_update_file', cls.create_or_update_file)
    setattr(cls, "create_or_update_file", _cache_create_or_update_file)
    

    def _cache_create_file(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_file')
        file = original_func(self, *args, **kwargs)
        StCache.delete_all(CacheKey.TIMING_DETAILS.value)
        
        return file
    
    setattr(cls, '_original_create_file', cls.create_file)
    setattr(cls, "create_file", _cache_create_file)
    
    def _cache_delete_file_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_file_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_delete_file_from_uuid', cls.delete_file_from_uuid)
    setattr(cls, "delete_file_from_uuid", _cache_delete_file_from_uuid)
    
    
    def _cache_update_file(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_file')
        file = original_func(self, *args, **kwargs)
        if file:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
        
        return file
    
    setattr(cls, '_original_update_file', cls.update_file)
    setattr(cls, "update_file", _cache_update_file)
    
    
    def _cache_create_project(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_project')
        project = original_func(self, *args, **kwargs)
        if project:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
        
        return project
    
    setattr(cls, '_original_create_project', cls.create_project)
    setattr(cls, "create_project", _cache_create_project)
    
    
    def _cache_delete_project_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_project_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_delete_project_from_uuid', cls.delete_project_from_uuid)
    setattr(cls, "delete_project_from_uuid", _cache_delete_project_from_uuid)
  
    
    def _cache_create_ai_model(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_ai_model')
        ai_model = original_func(self, *args, **kwargs)
        if ai_model:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
        
        return ai_model
    
    setattr(cls, '_original_create_ai_model', cls.create_ai_model)
    setattr(cls, "create_ai_model", _cache_create_ai_model)
    
    def _cache_update_ai_model(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_ai_model')
        ai_model = original_func(self, *args, **kwargs)
        if ai_model:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
        
        return ai_model
    
    setattr(cls, '_original_update_ai_model', cls.update_ai_model)
    setattr(cls, "update_ai_model", _cache_update_ai_model)
    
    def _cache_delete_ai_model_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_ai_model_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
    
    setattr(cls, '_original_delete_ai_model_from_uuid', cls.delete_ai_model_from_uuid)
    setattr(cls, "delete_ai_model_from_uuid", _cache_delete_ai_model_from_uuid)

    
    def _cache_get_timing_list_from_project(self, *args, **kwargs):
        timing_list = StCache.get_all(CacheKey.TIMING_DETAILS.value)
        if timing_list and len(timing_list) and len(args) > 0:
            project_specific_list = []
            for timing in timing_list:
                if timing.project.uuid == args[0]:
                    project_specific_list.append(timing)

            return project_specific_list
        
        original_func = getattr(cls, '_original_get_timing_list_from_project')
        timing_list = original_func(self, *args, **kwargs)
        if timing_list and len(timing_list):
            StCache.add_all(timing_list, CacheKey.TIMING_DETAILS.value)

        return timing_list
    
    setattr(cls, '_original_get_timing_list_from_project', cls.get_timing_list_from_project)
    setattr(cls, "get_timing_list_from_project", _cache_get_timing_list_from_project)
    
    def _cache_create_timing(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_timing')
        timing = original_func(self, *args, **kwargs)
        if timing:
            StCache.add(timing, CacheKey.TIMING_DETAILS.value)
        
        return timing
    
    setattr(cls, '_original_create_timing', cls.create_timing)
    setattr(cls, "create_timing", _cache_create_timing)

    def _cache_update_specific_timing(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_specific_timing')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_update_specific_timing', cls.update_specific_timing)
    setattr(cls, "update_specific_timing", _cache_update_specific_timing)

    def _cache_delete_timing_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_timing_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete(args[0],CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_delete_timing_from_uuid', cls.delete_timing_from_uuid)
    setattr(cls, "delete_timing_from_uuid", _cache_delete_timing_from_uuid)

    def _cache_remove_existing_timing(self, *args, **kwargs):
        original_func = getattr(cls, '_original_remove_existing_timing')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_remove_existing_timing', cls.remove_existing_timing)
    setattr(cls, "remove_existing_timing", _cache_remove_existing_timing)

    def _cache_remove_primary_frame(self, *args, **kwargs):
        original_func = getattr(cls, '_original_remove_primary_frame')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_remove_primary_frame', cls.remove_primary_frame)
    setattr(cls, "remove_primary_frame", _cache_remove_primary_frame)

    def _cache_remove_source_image(self, *args, **kwargs):
        original_func = getattr(cls, '_original_remove_source_image')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_remove_source_image', cls.remove_source_image)
    setattr(cls, "remove_source_image", _cache_remove_source_image)

    def _cache_move_frame_one_step_forward(self, *args, **kwargs):
        original_func = getattr(cls, '_original_move_frame_one_step_forward')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_move_frame_one_step_forward', cls.move_frame_one_step_forward)
    setattr(cls, "move_frame_one_step_forward", _cache_move_frame_one_step_forward)

    def _cache_get_app_setting_from_uuid(self, *args, **kwargs):
        app_setting_list = StCache.get_all(CacheKey.APP_SETTING.value)
        if app_setting_list and len(app_setting_list) and len(kwargs.keys()):
            for app_setting in app_setting_list:
                if app_setting.uuid == kwargs['uuid']:
                    return app_setting
        
        original_func = getattr(cls, '_original_get_app_setting_from_uuid')
        app_setting = original_func(self, *args, **kwargs)
        if app_setting:
            StCache.add(app_setting, CacheKey.APP_SETTING.value)
        
        return app_setting
    
    setattr(cls, '_original_get_app_setting_from_uuid', cls.get_app_setting_from_uuid)
    setattr(cls, "get_app_setting_from_uuid", _cache_get_app_setting_from_uuid)


    def _cache_get_all_app_setting_list(self, *args, **kwargs):
        app_setting_list = StCache.get_all(CacheKey.APP_SETTING.value)
        if app_setting_list and len(app_setting_list):
            return app_setting_list
        
        original_func = getattr(cls, '_original_get_all_app_setting_list')
        app_setting_list = original_func(self, *args, **kwargs)
        StCache.delete_all(CacheKey.APP_SETTING.value)
        StCache.add_all(app_setting_list, CacheKey.APP_SETTING.value)

        return app_setting_list
    
    setattr(cls, '_original_get_all_app_setting_list', cls.get_all_app_setting_list)
    setattr(cls, "get_all_app_setting_list", _cache_get_all_app_setting_list)

    def _cache_update_app_setting(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_app_setting')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.APP_SETTING.value)
            StCache.delete_all(CacheKey.APP_SECRET.value)
        
        return status
    
    setattr(cls, '_original_update_app_setting', cls.update_app_setting)
    setattr(cls, "update_app_setting", _cache_update_app_setting)

    def _cache_create_app_setting(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_app_setting')
        app_setting = original_func(self, *args, **kwargs)
        if app_setting:
            StCache.delete_all(CacheKey.APP_SETTING.value)
            StCache.delete_all(CacheKey.APP_SECRET.value)

            StCache.add(app_setting, CacheKey.APP_SETTING.value)

        
        return app_setting
    
    setattr(cls, '_original_create_app_setting', cls.create_app_setting)
    setattr(cls, "create_app_setting", _cache_create_app_setting)

    def _cache_delete_app_setting(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_app_setting')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.APP_SETTING.value)
            StCache.delete_all(CacheKey.APP_SECRET.value)
        
        return status
    
    setattr(cls, '_original_delete_app_setting', cls.delete_app_setting)
    setattr(cls, "delete_app_setting", _cache_delete_app_setting)

    def _cache_get_project_setting(self, *args, **kwargs):
        project_setting = StCache.get(args[0], CacheKey.PROJECT_SETTING.value)
        if project_setting:
            return project_setting
        
        original_func = getattr(cls, '_original_get_project_setting')
        project_setting = original_func(self, *args, **kwargs)
        if project_setting:
            StCache.add(project_setting, CacheKey.PROJECT_SETTING.value)
        
        return project_setting
    
    setattr(cls, '_original_get_project_setting', cls.get_project_setting)
    setattr(cls, "get_project_setting", _cache_get_project_setting)

    def _cache_create_project_setting(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_project_setting')
        project_setting = original_func(self, *args, **kwargs)
        if project_setting:
            StCache.add(project_setting, CacheKey.PROJECT_SETTING.value)
        
        return project_setting
    
    setattr(cls, '_original_create_project_setting', cls.create_project_setting)
    setattr(cls, "create_project_setting", _cache_create_project_setting)

    def _cache_update_project_setting(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_project_setting')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
        
        return status
    
    setattr(cls, '_original_update_project_setting', cls.update_project_setting)
    setattr(cls, "update_project_setting", _cache_update_project_setting)

    def _cache_bulk_update_project_setting(self, *args, **kwargs):
        original_func = getattr(cls, '_original_bulk_update_project_setting')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.PROJECT_SETTING.value)
        
        return status
    
    setattr(cls, '_original_bulk_update_project_setting', cls.bulk_update_project_setting)
    setattr(cls, "bulk_update_project_setting", _cache_bulk_update_project_setting)

    return cls