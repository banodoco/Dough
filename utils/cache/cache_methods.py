from shared.logging.logging import AppLogger
from utils.cache.cache import CacheKey, StCache

logger = AppLogger()


# NOTE: caching only timing_details, project settings, models and app settings. invalidating cache everytime a related data is updated
def cache_data(cls):
    # ---------------- FILE METHODS ----------------------
    def _cache_create_or_update_file(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_or_update_file')
        file = original_func(self, *args, **kwargs)
        
        if file:
            StCache.delete(file.uuid, CacheKey.FILE.value)
            StCache.add(file, CacheKey.FILE.value)
        
        return file
    
    setattr(cls, '_original_create_or_update_file', cls.create_or_update_file)
    setattr(cls, "create_or_update_file", _cache_create_or_update_file)
    
    def _cache_create_file(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_file')
        file = original_func(self, *args, **kwargs)
        
        if file:
            StCache.delete(file.uuid, CacheKey.FILE.value)
            StCache.add(file, CacheKey.FILE.value)
        
        return file
    
    setattr(cls, '_original_create_file', cls.create_file)
    setattr(cls, "create_file", _cache_create_file)
    
    def _cache_delete_file_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_file_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete(args[0], CacheKey.TIMING_DETAILS.value)
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_delete_file_from_uuid', cls.delete_file_from_uuid)
    setattr(cls, "delete_file_from_uuid", _cache_delete_file_from_uuid)
    
    def _cache_update_file(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_file')
        file = original_func(self, *args, **kwargs)
        
        if file:
            StCache.delete(file.uuid, CacheKey.FILE.value)
            StCache.add(file, CacheKey.FILE.value)
        
        return file
    
    setattr(cls, '_original_update_file', cls.update_file)
    setattr(cls, "update_file", _cache_update_file)

    def _cache_get_file_from_name(self, *args, **kwargs):
        file_list = StCache.get_all(CacheKey.FILE.value)
        if file_list and len(file_list) and len(args) > 0:
            for file in file_list:
                if file.name == args[0]:
                    return file
        
        original_func = getattr(cls, '_original_get_file_from_name')
        file = original_func(self, *args, **kwargs)
        if file:
            StCache.add(file, CacheKey.FILE.value)

        return file
    
    setattr(cls, '_original_get_file_from_name', cls.get_file_from_name)
    setattr(cls, "get_file_from_name", _cache_get_file_from_name)

    def _cache_get_file_from_uuid(self, *args, **kwargs):
        file_list = StCache.get_all(CacheKey.FILE.value)
        if file_list and len(file_list) and len(args) > 0:
            for file in file_list:
                if file.uuid == args[0]:
                    return file
        
        original_func = getattr(cls, '_original_get_file_from_uuid')
        file = original_func(self, *args, **kwargs)
        if file:
            StCache.add(file, CacheKey.FILE.value)

        return file
    
    setattr(cls, '_original_get_file_from_uuid', cls.get_file_from_uuid)
    setattr(cls, "get_file_from_uuid", _cache_get_file_from_uuid)

    def _cache_get_image_list_from_uuid_list(self, *args, **kwargs):
        not_found_list, found_list = [], {}
        # finding the images in the cache
        file_list = StCache.get_all(CacheKey.FILE.value)
        if file_list and len(file_list) and len(args) > 0:
            for file in file_list:
                if file.uuid in args[0]:
                    found_list[file.uuid] = file

        for file_uuid in args[0]:
            if file_uuid not in found_list:
                not_found_list.append(file_uuid)

        # images which are not present in the cache are fetched through the db
        if len(not_found_list):
            original_func = getattr(cls, '_original_get_image_list_from_uuid_list')
            res = original_func(self, not_found_list, **kwargs)
            for file in res:
                found_list[file.uuid] = file

        # ordering the result
        res = []
        if found_list:
            for file_uuid in args[0]:
                if file_uuid in found_list:
                    res.append(found_list[file_uuid])

        for file in res:
            StCache.delete(file, CacheKey.FILE.value)
            StCache.add(file, CacheKey.FILE.value)
        
        return res
    
    setattr(cls, '_original_get_image_list_from_uuid_list', cls.get_image_list_from_uuid_list)
    setattr(cls, "get_image_list_from_uuid_list", _cache_get_image_list_from_uuid_list)

    def _cache_get_file_list_from_log_uuid_list(self, *args, **kwargs):
        not_found_list, found_list = [], {}
        # finding files in the cache
        file_list = StCache.get_all(CacheKey.FILE.value)
        if file_list and len(file_list) and len(args) > 0:
            for file in file_list:
                if file.inference_log and file.inference_log.uuid in args[0]:
                    found_list[file.inference_log.uuid] = file
        
        for log_uuid in args[0]:
            if log_uuid not in found_list:
                not_found_list.append(log_uuid)

        if len(not_found_list):
            original_func = getattr(cls, '_original_get_file_list_from_log_uuid_list')
            res = original_func(self, not_found_list, **kwargs)
            for file in res:
                found_list[file.inference_log.uuid] = file

        res = []
        if found_list:
            for log_uuid in args[0]:
                if log_uuid in found_list:
                    res.append(found_list[log_uuid])

        for file in res:
            StCache.delete(file, CacheKey.FILE.value)
            StCache.add(file, CacheKey.FILE.value)
        
        return res
    
    setattr(cls, '_original_get_file_list_from_log_uuid_list', cls.get_file_list_from_log_uuid_list)
    setattr(cls, "get_file_list_from_log_uuid_list", _cache_get_file_list_from_log_uuid_list)
    
    
    # ------------------ PROJECT METHODS -----------------------
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
    
    
    # -------------------- AI MODEL METHODS ----------------------
    def _cache_get_ai_model_from_uuid(self, *args, **kwargs):
        model_list = StCache.get_all(CacheKey.AI_MODEL.value)
        if model_list and len(model_list) and len(args) > 0:
            for model in model_list:
                if model.uuid == args[0]:
                    return model
        
        original_func = getattr(cls, '_original_get_ai_model_from_uuid')
        model = original_func(self, *args, **kwargs)
        if model:
            StCache.add(model, CacheKey.AI_MODEL.value)

        return model

    setattr(cls, '_original_get_ai_model_from_uuid', cls.get_ai_model_from_uuid)
    setattr(cls, "get_ai_model_from_uuid", _cache_get_ai_model_from_uuid)

    def _cache_get_ai_model_from_name(self, *args, **kwargs):
        model_list = StCache.get_all(CacheKey.AI_MODEL.value)
        if model_list and len(model_list) and len(args) > 0:
            for model in model_list:
                if model.name == args[0]:
                    return model
        
        original_func = getattr(cls, '_original_get_ai_model_from_name')
        model = original_func(self, *args, **kwargs)
        if model:
            StCache.add(model, CacheKey.AI_MODEL.value)

        return model

    setattr(cls, '_original_get_ai_model_from_name', cls.get_ai_model_from_name)
    setattr(cls, "get_ai_model_from_name", _cache_get_ai_model_from_name)
    
    def _cache_create_ai_model(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_ai_model')
        ai_model = original_func(self, *args, **kwargs)
        if ai_model:
            StCache.delete_all(CacheKey.AI_MODEL.value)
        
        return ai_model
    
    setattr(cls, '_original_create_ai_model', cls.create_ai_model)
    setattr(cls, "create_ai_model", _cache_create_ai_model)
    
    def _cache_update_ai_model(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_ai_model')
        ai_model = original_func(self, *args, **kwargs)
        if ai_model:
            StCache.delete_all(CacheKey.AI_MODEL.value)
        
        return ai_model
    
    setattr(cls, '_original_update_ai_model', cls.update_ai_model)
    setattr(cls, "update_ai_model", _cache_update_ai_model)
    
    def _cache_delete_ai_model_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_ai_model_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.AI_MODEL.value)
    
    setattr(cls, '_original_delete_ai_model_from_uuid', cls.delete_ai_model_from_uuid)
    setattr(cls, "delete_ai_model_from_uuid", _cache_delete_ai_model_from_uuid)

    
    # ------------------- TIMING METHODS ---------------------
    def _cache_get_timing_list_from_project(self, *args, **kwargs):
        # checking if it's already present in the cache
        timing_list = StCache.get_all(CacheKey.TIMING_DETAILS.value)
        if timing_list and len(timing_list) and len(args) > 0:
            project_specific_list = []
            for timing in timing_list:
                if timing.shot.project.uuid == args[0]:
                    project_specific_list.append(timing)

            # if there are any timings for the project, return them
            if len(project_specific_list):
                sorted_objects = sorted(project_specific_list, key=lambda x: x.aux_frame_index)
                return sorted_objects
        
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
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
            StCache.delete_all(CacheKey.SHOT.value)
        
        return timing
    
    setattr(cls, '_original_create_timing', cls.create_timing)
    setattr(cls, "create_timing", _cache_create_timing)

    def _cache_update_specific_timing(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_specific_timing')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
            # deleting shots as well. for e.g. timing update can be moving it from
            # one shot to another
            StCache.delete_all(CacheKey.SHOT.value)

        # updating the timing list
        timing_func = getattr(cls, '_original_get_timing_from_uuid')
        timing = timing_func(self, args[0])
        if timing and timing.shot.project:
            original_func = getattr(cls, '_original_get_timing_list_from_project')
            timing_list = original_func(self, timing.shot.project.uuid)
            if timing_list and len(timing_list):
                StCache.add_all(timing_list, CacheKey.TIMING_DETAILS.value)
    
    setattr(cls, '_original_update_specific_timing', cls.update_specific_timing)
    setattr(cls, "update_specific_timing", _cache_update_specific_timing)

    def _cache_get_timing_from_uuid(self, *args, **kwargs):
        if not kwargs.get('invalidate_cache', False):
            timing_list = StCache.get_all(CacheKey.TIMING_DETAILS.value)
            if timing_list and len(timing_list) and len(args) > 0:
                for timing in timing_list:
                    if timing.uuid == args[0]:
                        return timing
        
        original_func = getattr(cls, '_original_get_timing_from_uuid')
        timing = original_func(self, *args, **kwargs)

        if timing:
            StCache.add(timing, CacheKey.TIMING_DETAILS.value)

        return timing
    
    setattr(cls, '_original_get_timing_from_uuid', cls.get_timing_from_uuid)
    setattr(cls, "get_timing_from_uuid", _cache_get_timing_from_uuid)

    def _cache_delete_timing_from_uuid(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_timing_from_uuid')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete(args[0],CacheKey.TIMING_DETAILS.value)
            StCache.delete_all(CacheKey.SHOT.value)
    
    setattr(cls, '_original_delete_timing_from_uuid', cls.delete_timing_from_uuid)
    setattr(cls, "delete_timing_from_uuid", _cache_delete_timing_from_uuid)

    def _cache_remove_existing_timing(self, *args, **kwargs):
        original_func = getattr(cls, '_original_remove_existing_timing')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
            StCache.delete_all(CacheKey.SHOT.value)
    
    setattr(cls, '_original_remove_existing_timing', cls.remove_existing_timing)
    setattr(cls, "remove_existing_timing", _cache_remove_existing_timing)

    def _cache_remove_primary_frame(self, *args, **kwargs):
        original_func = getattr(cls, '_original_remove_primary_frame')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
            StCache.delete_all(CacheKey.SHOT.value)
    
    setattr(cls, '_original_remove_primary_frame', cls.remove_primary_frame)
    setattr(cls, "remove_primary_frame", _cache_remove_primary_frame)

    def _cache_remove_source_image(self, *args, **kwargs):
        original_func = getattr(cls, '_original_remove_source_image')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.TIMING_DETAILS.value)
            StCache.delete_all(CacheKey.SHOT.value)
    
    setattr(cls, '_original_remove_source_image', cls.remove_source_image)
    setattr(cls, "remove_source_image", _cache_remove_source_image)

    
    # ------------------ APP SETTING METHODS ---------------------
    def _cache_get_app_setting_from_uuid(self, *args, **kwargs):
        app_setting_list = StCache.get_all(CacheKey.APP_SETTING.value)
        if not len(kwargs) and len(app_setting_list):
            return app_setting_list[0]
        
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


    # ------------------ PROJECT SETTING METHODS ---------------------
    def _cache_get_project_setting(self, *args, **kwargs):
        project_setting_list = StCache.get_all(CacheKey.PROJECT_SETTING.value)
        for ele in project_setting_list:
            if str(ele.project.uuid) == str(args[0]):
                return ele
        
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


    # ---------------------- USER METHODS ---------------------
    def _cache_update_user(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_user')
        user = original_func(self, *args, **kwargs)
        StCache.delete_all(CacheKey.LOGGED_USER.value)

        return user
    
    setattr(cls, '_original_update_user', cls.update_user)
    setattr(cls, "update_user", _cache_update_user)

    def _cache_get_first_active_user(self, *args, **kwargs):
        logged_user_list = StCache.get_all(CacheKey.LOGGED_USER.value)
        if logged_user_list and len(logged_user_list):
            return logged_user_list[0]

        original_func = getattr(cls, '_original_get_first_active_user')
        user = original_func(self, *args, **kwargs)
        StCache.delete_all(CacheKey.LOGGED_USER.value)
        StCache.add(user, CacheKey.LOGGED_USER.value)

        return user
    
    setattr(cls, '_original_get_first_active_user', cls.get_first_active_user)
    setattr(cls, "get_first_active_user", _cache_get_first_active_user)

    def _cache_create_user(self, **kwargs):
        original_func = getattr(cls, '_original_create_user')
        user = original_func(self, **kwargs)
        StCache.update(user, CacheKey.LOGGED_USER.value)

        return user
    
    setattr(cls, '_original_create_user', cls.create_user)
    setattr(cls, "create_user", _cache_create_user)

    def _cache_google_user_login(self, **kwargs):
        original_func = getattr(cls, '_original_google_user_login')
        user, token, refresh_token = original_func(self, **kwargs)
        StCache.delete_all(CacheKey.LOGGED_USER.value)
        if user:
            StCache.add(user, CacheKey.LOGGED_USER.value)

        return user, token, refresh_token
    
    setattr(cls, '_original_google_user_login', cls.google_user_login)
    setattr(cls, "google_user_login", _cache_google_user_login)

    # ---------------------- SHOT METHODS ---------------------
    def _cache_get_shot_from_uuid(self, *args, **kwargs):
        shot_list = StCache.get_all(CacheKey.SHOT.value)
        if shot_list and len(shot_list):
            for shot in shot_list:
                if shot.uuid == args[0]:
                    return shot
        
        original_func = getattr(cls, '_original_get_shot_from_uuid')
        shot = original_func(self, *args, **kwargs)
        
        return shot

    setattr(cls, '_original_get_shot_from_uuid', cls.get_shot_from_uuid)
    setattr(cls, "get_shot_from_uuid", _cache_get_shot_from_uuid)

    def _cache_get_shot_from_number(self, *args, **kwargs):
        shot_list = StCache.get_all(CacheKey.SHOT.value)
        if shot_list and len(shot_list):
            for shot in shot_list:
                if shot.project.uuid == args[0] and shot.shot_idx == kwargs['shot_number']:
                    return shot
        
        original_func = getattr(cls, '_original_get_shot_from_number')
        shot = original_func(self, *args, **kwargs)
        
        return shot
    
    setattr(cls, '_original_get_shot_from_number', cls.get_shot_from_number)
    setattr(cls, "get_shot_from_number", _cache_get_shot_from_number)

    def _cache_get_shot_list(self, *args, **kwargs):
        shot_list = StCache.get_all(CacheKey.SHOT.value)
        if shot_list and len(shot_list):
            res = []
            for shot in shot_list:
                if shot.project.uuid == args[0]:
                    res.append(shot)
            if len(res):
                return res
        
        original_func = getattr(cls, '_original_get_shot_list')
        shot_list = original_func(self, *args, **kwargs)
        if shot_list:
            StCache.add_all(shot_list, CacheKey.SHOT.value)
        
        return shot_list
    
    setattr(cls, '_original_get_shot_list', cls.get_shot_list)
    setattr(cls, "get_shot_list", _cache_get_shot_list)

    def _cache_create_shot(self, *args, **kwargs):
        original_func = getattr(cls, '_original_create_shot')
        shot = original_func(self, *args, **kwargs)
        
        if shot:
            # deleting all the shots as this could have affected other shots as well
            # for e.g. shot_idx shift
            StCache.delete_all(CacheKey.SHOT.value)
        
        return shot
    
    setattr(cls, '_original_create_shot', cls.create_shot)
    setattr(cls, "create_shot", _cache_create_shot)

    def _cache_update_shot(self, *args, **kwargs):
        original_func = getattr(cls, '_original_update_shot')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete_all(CacheKey.SHOT.value)
        
        return status
    
    setattr(cls, '_original_update_shot', cls.update_shot)
    setattr(cls, "update_shot", _cache_update_shot)

    def _cache_delete_shot(self, *args, **kwargs):
        original_func = getattr(cls, '_original_delete_shot')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete(args[0], CacheKey.SHOT.value)
        
        return status
    
    setattr(cls, '_original_delete_shot', cls.delete_shot)
    setattr(cls, "delete_shot", _cache_delete_shot)

    def _cache_add_interpolated_clip(self, *args, **kwargs):
        original_func = getattr(cls, '_original_add_interpolated_clip')
        status = original_func(self, *args, **kwargs)
        
        if status:
            StCache.delete(args[0], CacheKey.SHOT.value)
        
        return status
    
    setattr(cls, '_original_add_interpolated_clip', cls.add_interpolated_clip)
    setattr(cls, "add_interpolated_clip", _cache_add_interpolated_clip)

    def _cache_get_timing_list_from_shot(self, *args, **kwargs):
        shot_list = StCache.get_all(CacheKey.SHOT.value)
        if shot_list and len(shot_list):
            for shot in shot_list:
                if str(shot.uuid) == str(args[0]):
                    return shot.timing_list
        
        original_func = getattr(cls, '_original_get_timing_list_from_shot')
        timing_list = original_func(self, *args, **kwargs)
        
        return timing_list
    
    setattr(cls, '_original_get_timing_list_from_shot', cls.get_timing_list_from_shot)
    setattr(cls, "get_timing_list_from_shot", _cache_get_timing_list_from_shot)

    def _cache_duplicate_shot(self, *args, **kwargs):
        original_func = getattr(cls, '_original_duplicate_shot')
        shot = original_func(self, *args, **kwargs)
        
        if shot:
            StCache.delete_all(CacheKey.SHOT.value)
        
        return shot
    
    setattr(cls, '_original_duplicate_shot', cls.duplicate_shot)
    setattr(cls, "duplicate_shot", _cache_duplicate_shot)

    return cls