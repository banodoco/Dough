import os
import sqlite3
from database.constants import InternalFileType

from django_settings import DB_LOCATION

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from database.models import AIModel, AIModelParamMap, AppSetting, InferenceLog, InternalFileObject, Project, Setting, Timing, User

from database.serializers.dao import CreateAIModelDao, CreateAIModelParamMapDao, CreateAppSettingDao, CreateFileDao, CreateInferenceLogDao, CreateProjectDao, CreateSettingDao, CreateTimingDao, CreateUserDao, UpdateSettingDao
from utils.internal_response import InternalResponse


# TODO: if local and hosted DB inteferences are very different then separate them into different classes

class DBRepo:
    def __init__(self):
        database_file = DB_LOCATION
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")

        from django.core.wsgi import get_wsgi_application
        application = get_wsgi_application()

        # creating db if not already present
        if not os.path.exists(database_file):
            from django.core.management import execute_from_command_line
            conn = sqlite3.connect(database_file)
            conn.close()
            
            execute_from_command_line(['manage.py', 'migrate'])

    # user operations
    def create_user(self, **kwargs):
        data = CreateUserDao(data=kwargs)
        if not data.is_valid():
            return InternalResponse({}, data.errors, False)
        
        user = User.objects.filter(email=data.validated_data['email'], is_disabled=False).first()
        if user:
            return InternalResponse({}, 'user already present', False)
        
        user = User.objects.create(**data.validated_data)
        
        return InternalResponse(user, 'user created successfully', True)
    
    def get_user_by_email(self, email):
        user = User.objects.filter(email=email, is_disabled=False).first()
        if user:
            return InternalResponse(user, 'user found', True)
        
        return InternalResponse({}, 'user not found', False)
    
    def get_all_user_list(self):
        user_list = User.objects.all()
        return InternalResponse(user_list, 'user list', True)
    
    def delete_user_by_email(self, email):
        user = User.objects.filter(email=email, is_disabled=False).first()
        if user:
            user.is_disabled = True
            user.save()
            return InternalResponse({}, 'user deleted successfully', True)
        
        return InternalResponse({}, 'user not found', False)

    # internal file object
    def get_file_from_uuid(self, uuid):
        file = InternalFileObject.objects.filter(uuid=uuid, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'file not found', False)
        
        return InternalResponse(file, 'file found', True)
    
    def get_all_file_list(self, file_type: InternalFileType):
        file_list = InternalFileObject.objects.filter(file_type=file_type.value, is_disabled=False).all()
        return InternalResponse(file_list, 'file list', True)
    
    def create_file(self, **kwargs):
        data = CreateFileDao(data=kwargs)
        if not data.is_valid():
            return InternalResponse({}, data.errors, False)
        
        file = InternalFileObject.objects.create(**data.validated_data)
        
        return InternalResponse(file, 'file created successfully', True)
    
    def delete_file_from_uuid(self, uuid):
        file = InternalFileObject.objects.filter(uuid=uuid, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'invalid file uuid', False)
        
        return InternalResponse({}, 'file deleted successfully', True)
    
    # project
    def get_project_from_uuid(self, uuid):
        project = Project.objects.filter(uuid=uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project uuid', False)
        
        return InternalResponse(project, 'project fetched', True)
    
    def get_all_project_list(self, user_id):
        project_list = Project.objects.filter(user_id=user_id, is_disabled=False).all()
        return InternalResponse(project_list, 'project list fetched', True)
    
    def create_project(self, **kwargs):
        data = CreateProjectDao(data=kwargs)
        if not data.is_valid():
            return InternalResponse({}, data.errors, False)
        
        user = User.objects.filter(uuid=data.validated_data['user_id'], is_disabled=False).first()
        if not user:
            return InternalResponse({}, 'invalid user', False)
        
        print(data.data)
        data._data['user_id'] = user.id
        
        project = Project.objects.create(**data.data)
        return InternalResponse(project, 'project created successfully', True)
    
    def delete_project_from_uuid(self, uuid):
        project = Project.objects.filter(uuid=uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project uuid', False)
        
        project.is_disabled = True
        project.save()
        return InternalResponse({}, 'project deleted successfully', True)

    
    # ai model (custom ai model)
    def get_ai_model_from_uuid(self, uuid):
        ai_model = AIModel.objects.filter(uuid=uuid, is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model uuid', False)
        
        return InternalResponse(ai_model, 'ai model fetched', True)
    
    def get_all_ai_model_list(self, user_id=None):
        if user_id:
            ai_model_list = AIModel.objects.filter(user_id=user_id, is_disabled=False).all()
        else:
            ai_model_list = AIModel.objects.filter(is_disabled=False).all()

        return InternalResponse(ai_model_list, 'ai model list fetched', True)
    
    def create_ai_model(self, **kwargs):
        attributes = CreateAIModelDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'user_id' in attributes.data and attributes.data['user_id']:
            user = User.objects.filter(uuid=attributes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            print(attributes.data['user_id'])
            attributes.data['user_id'] = user.id
        
        ai_model = InternalFileObject.objects.create(**attributes.data)
        
        return InternalResponse(ai_model, 'ai model created successfully', True)
    
    def delete_ai_model_from_uuid(self, uuid):
        ai_model = AIModel.objects.filter(uuid=uuid, is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model uuid', False)
        
        ai_model.is_disabled = True
        ai_model.save()
        return InternalResponse({}, 'ai model deleted successfully', True)
    

    # inference log
    def get_inference_log_from_uuid(self, uuid):
        log = InferenceLog.objects.filter(uuid=uuid, is_disabled=False).first()
        if not log:
            return InternalResponse({}, 'invalid inference log uuid', False)
        
        return InternalResponse(log, 'inference log fetched', True)
    
    def get_all_inference_log_list(self, project_id=None):
        if project_id:
            log_list = InferenceLog.objects.filter(project_id=project_id, is_disabled=False).all()
        else:
            log_list = InferenceLog.objects.filter(is_disabled=False).all()
        
        return InternalResponse(log_list, 'inference log list fetched', True)
    
    def create_inference_log(self, **kwargs):
        attributes = CreateInferenceLogDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            attributes.data['project_id'] = project.id
        
        if 'model_id' in attributes.data and attributes.data['model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes.data['model_id'] = model.id

        log = InferenceLog.objects.create(**attributes.data)

        
        return InternalResponse(log, 'inference log created successfully', True)
    
    def delete_inference_log_from_uuid(self, uuid):
        log = InferenceLog.objects.filter(uuid=uuid, is_disabled=False).first()
        if not log:
            return InternalResponse({}, 'invalid inference log uuid', False)
        
        log.is_disabled = True
        log.save()
        return InternalResponse({}, 'inference log deleted successfully', True)
    

    # ai model param map
    def get_ai_model_param_map_from_uuid(self, uuid):
        map = AIModelParamMap.objects.filter(uuid=uuid, is_disabled=False).first()
        if not map:
            return InternalResponse({}, 'invalid ai model param map uuid', False)
        
        return InternalResponse(map, 'ai model param map fetched', True)
    
    def get_all_ai_model_param_map_list(self, model_id=None):
        if model_id:
            map_list = AIModelParamMap.objects.filter(model_id=model_id, is_disabled=False).all()
        else:
            map_list = AIModelParamMap.objects.filter(is_disabled=False).all()
        
        return InternalResponse(map_list, 'ai model param map list fetched', True)
    
    def create_ai_model_param_map(self, **kwargs):
        attributes = CreateAIModelParamMapDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'model_id' in attributes.data and attributes.data['model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes.data['model_id'] = model.id
        
        map = AIModelParamMap.objects.create(**attributes.data)
        
        return InternalResponse(map, 'ai model param map created successfully', True)
    
    def delete_ai_model(self, uuid):
        map = AIModelParamMap.objects.filter(uuid=uuid, is_disabled=False).first()
        if not map:
            return InternalResponse({}, 'invalid ai model param map uuid', False)
        
        map.is_disabled = True
        map.save()
        return InternalResponse({}, 'ai model param map deleted successfully', True)
    

    # timing
    def get_timing_from_uuid(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        return InternalResponse(timing, 'timing fetched', True)
    
    def get_timing_list_from_project(self, project_id=None):
        if project_id:
            timing_list = Timing.objects.filter(project_id=project_id, is_disabled=False).all()
        else:
            timing_list = Timing.objects.filter(is_disabled=False).all()
        
        return InternalResponse(timing_list, 'timing list fetched', True)
    
    def create_timing(self, **kwargs):
        attributes = CreateTimingDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            attributes.data['project_id'] = project.id
        
        timing = Timing.objects.create(**attributes.data)
        
        return InternalResponse(timing, 'timing created successfully', True)
    
    def delete_timing_from_uuid(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.is_disabled = True
        timing.save()
        return InternalResponse({}, 'timing deleted successfully', True)
    

    # app setting
    def get_app_setting_from_uuid(self, uuid):
        if uuid:
            app_setting = AppSetting.objects.filter(uuid=uuid, is_disabled=False).first()
        else:
            app_setting = AppSetting.objects.filter(is_disabled=False).first()

        return InternalResponse(app_setting, 'app setting fetched', True)
    
    def get_all_app_setting_list(self):
        app_setting_list = AppSetting.objects.filter(is_disabled=False).all()
        return InternalResponse(app_setting_list, 'app setting list fetched', True)
    
    def create_app_setting(self, **kwargs):
        attributes = CreateAppSettingDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'user_id' in attributes.data and attributes.data['user_id']:
            user = User.objects.filter(uuid=attributes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            attributes.data['user_id'] = user.id
        
        app_setting = AppSetting.objects.create(**attributes.data)
        
        return InternalResponse(app_setting, 'app setting created successfully', True)
    

    def delete_app_setting(self, user_id):
        if AppSetting.objects.filter(is_disabled=False).count() <= 1:
            return InternalResponse({}, 'cannot delete the last app setting', False)
        
        app_setting = AppSetting.objects.filter(user_id=user_id, is_disabled=False).first()
        if not app_setting:
            return InternalResponse({}, 'invalid app setting', False)
        
        app_setting.is_disabled = True
        app_setting.save()
        return InternalResponse({}, 'app setting deleted successfully', True)
    

    # setting
    def get_project_setting(self, project_id):
        setting = Setting.objects.filter(project_id=project_id, is_disabled=False).first()
        if not setting:
            return InternalResponse({}, 'invalid project_id', False)
        
        return InternalResponse(setting, 'setting fetched', True)
    
    # TODO: add valid model_id check throughout dp_repo
    def create_project_setting(self, **kwargs):
        attributes = CreateSettingDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            attributes.data['project_id'] = project.id
        
        setting = Setting.objects.create(**attributes.data)
        
        return InternalResponse(setting, 'setting created successfully', True)
    
    def update_project_setting(self, **kwargs):
        attributes = UpdateSettingDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        setting = Setting.objects.filter(uuid=attributes.data['uuid'], is_disabled=False).first()
        if not setting:
            return InternalResponse({}, 'invalid project', False)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            print(attributes.data)
            attributes.data['project_id'] = project.id
        
        if 'default_model_id' in attributes.data and attributes.data['default_model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['default_model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes.data['default_model_id'] = model.id

        if 'audio_id' in attributes.data and attributes.data['audio_id']:
            audio = InternalFileObject.objects.filter(uuid=attributes.data['audio_id'], is_disabled=False).first()
            if not audio:
                return InternalResponse({}, 'invalid audio', False)
            
            attributes.data['audio_id'] = audio.id

        if 'input_video_id' in attributes.data and attributes.data['input_video_id']:
            video = InternalFileObject.objects.filter(uuid=attributes.data['input_video_id'], is_disabled=False).first()
            if not video:
                return InternalResponse({}, 'invalid video', False)
            
            attributes.data['input_video_id'] = video.id

        
        setting.update(**attributes.data)
        
        return InternalResponse(setting, 'setting updated successfully', True)