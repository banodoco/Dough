import os
import sqlite3
import uuid
from backend.constants import InternalFileType
from backend.serializers.dto import AIModelDto, AppSettingDto, InferenceLogDto, InternalFileDto, ProjectDto, SettingDto, TimingDto, UserDto

from django_settings import DB_LOCATION
from shared.constants import AUTOMATIC_FILE_HOSTING, LOCAL_DATABASE_NAME
from shared.file_upload.s3 import upload_file

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from backend.models import AIModel, AIModelParamMap, AppSetting, InferenceLog, InternalFileObject, Project, Setting, Timing, User

from backend.serializers.dao import CreateAIModelDao, CreateAIModelParamMapDao, CreateAppSettingDao, CreateFileDao, CreateInferenceLogDao, CreateProjectDao, CreateSettingDao, CreateTimingDao, CreateUserDao, UpdateAIModelDao, UpdateSettingDao
from utils.internal_response import InternalResponse


# TODO: if local and hosted DB inteferences are very different then separate them into different classes

class DBRepo:
    def __init__(self):
        database_file = '../' + LOCAL_DATABASE_NAME
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

        payload = {
            'data': UserDto(user).data
        }
        
        return InternalResponse(payload, 'user created successfully', True)
    
    def get_user_by_email(self, email):
        user = User.objects.filter(email=email, is_disabled=False).first()
        if user:
            return InternalResponse(user, 'user found', True)
        
        payload = {
            'data': UserDto(user).data
        }
        
        return InternalResponse(payload, 'user not found', False)
    
    def get_all_user_list(self):
        user_list = User.objects.all()

        payload = {
            'data': UserDto(user_list, many=True).data
        }
        return InternalResponse(payload, 'user list', True)
    
    def delete_user_by_email(self, email):
        user = User.objects.filter(email=email, is_disabled=False).first()
        if user:
            user.is_disabled = True
            user.save()
            return InternalResponse({}, 'user deleted successfully', True)
        
        payload = {
            'data': UserDto(user).data
        }
        
        return InternalResponse(payload, 'user not found', False)

    # internal file object
    def get_file_from_name(self, name):
        file = InternalFileObject.objects.filter(name=name, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'file not found', False)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)

    def get_file_from_uuid(self, uuid):
        file = InternalFileObject.objects.filter(uuid=uuid, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'file not found', False)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)
    
    def get_all_file_list(self, file_type: InternalFileType):
        file_list = InternalFileObject.objects.filter(file_type=file_type.value, is_disabled=False).all()
        
        payload = {
            'data': InternalFileDto(file_list, many=True).data
        }

        return InternalResponse(payload, 'file found', True)
    
    def create_or_update_file(self, filename, type=InternalFileType.IMAGE.value, **kwargs):
        file = InternalFileType.objects.filter(name=filename, type=type, is_disabled=False).first()
        if not file:
            file = InternalFileObject.objects.create(name=filename, file_type=type, **kwargs)
        else:
            file.update(**kwargs)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)
    
    def create_file(self, **kwargs):
        data = CreateFileDao(data=kwargs)

        # hosting the file if only local path is provided and it's a production environment
        if 'hosted_url' not in kwargs and AUTOMATIC_FILE_HOSTING:
            # this is the user who is uploading the file
            user = User.objects.filter(uuid=data.kwargs['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            app_setting: AppSetting = AppSetting.objects.filter(is_disabled=False).first()
            if not app_setting:
                return InternalResponse({}, 'app setting not found', False)
            
            filename = str(uuid.uuid4())
            hosted_url = upload_file(filename, app_setting.aws_access_key_decrypted, \
                                     app_setting.aws_secret_access_key_decrypted)
            
            print(data.data)
            data._data['hosted_url'] = hosted_url
        

        if not data.is_valid():
            return InternalResponse({}, data.errors, False)
        
        file = InternalFileObject.objects.create(**data.data)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)
    
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
        
        payload = {
            'data': ProjectDto(project).data
        }
        
        return InternalResponse(payload, 'project fetched', True)
    
    def get_all_project_list(self, user_id):
        project_list = Project.objects.filter(user_id=user_id, is_disabled=False).all()
        
        payload = {
            'data': ProjectDto(project_list, many=True).data
        }
        
        return InternalResponse(payload, 'project fetched', True)
    
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
        
        payload = {
            'data': ProjectDto(project).data
        }
        
        return InternalResponse(payload, 'project fetched', True)
    
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
        
        payload = {
            'data': AIModelDto(ai_model).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
    def get_all_ai_model_list(self, user_id=None):
        if user_id:
            ai_model_list = AIModel.objects.filter(user_id=user_id, is_disabled=False).all()
        else:
            ai_model_list = AIModel.objects.filter(is_disabled=False).all()

        payload = {
            'data': AIModelDto(ai_model_list, many=True).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
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
        
        payload = {
            'data': AIModelDto(ai_model).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
    def update_ai_model(self, **kwargs):
        attirbutes = UpdateAIModelDao(attirbutes=kwargs)
        if not attirbutes.is_valid():
            return InternalResponse({}, attirbutes.errors, False)
        
        ai_model = AIModel.objects.filter(uuid=attirbutes.data['uuid'], is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model uuid', False)
        
        if 'user_id' in attirbutes.data and attirbutes.data['user_id']:
            user = User.objects.filter(uuid=attirbutes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            print(attirbutes.data['user_id'])
            attirbutes.data['user_id'] = user.id
        
        ai_model.update(**attirbutes.data)
        
        payload = {
            'data': AIModelDto(ai_model).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
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
        
        payload = {
            'data': InferenceLogDto(log).data
        }
        
        return InternalResponse(payload, 'inference log fetched', True)
    
    def get_all_inference_log_list(self, project_id=None):
        if project_id:
            log_list = InferenceLog.objects.filter(project_id=project_id, is_disabled=False).all()
        else:
            log_list = InferenceLog.objects.filter(is_disabled=False).all()
        
        payload = {
            'data': InferenceLogDto(log_list, many=True).data
        }
        
        return InternalResponse(payload, 'inference log list fetched', True)
    
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

        
        payload = {
            'data': InferenceLogDto(log).data
        }
        
        return InternalResponse(payload, 'inference log created successfully', True)
    
    def delete_inference_log_from_uuid(self, uuid):
        log = InferenceLog.objects.filter(uuid=uuid, is_disabled=False).first()
        if not log:
            return InternalResponse({}, 'invalid inference log uuid', False)
        
        log.is_disabled = True
        log.save()

        return InternalResponse({}, 'inference log deleted successfully', True)
    

    # ai model param map
    # TODO: add DTO in the output
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
        
        payload = {
            'data': TimingDto(timing).data
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    def get_timing_from_frame_number(self, project_uuid, frame_number):
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if project:
            timing = Timing.objects.filter(frame_number=frame_number, project_id=project.id, is_disabled=False).first()
            if timing:
                payload = {
                    'data': TimingDto(timing).data
                }
                
                return InternalResponse(payload, 'timing fetched', True)
            
        return InternalResponse({}, 'invalid timing frame number', False)
    
    def get_primary_variant_location(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)

        payload = {
            'data': timing.primary_variant_location
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    # this is based on the aux_frame_index and not the order in the db
    def get_next_timing(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        next_timing = Timing.objects.filter(aux_frame_index__gt=timing.aux_frame_index, is_disabled=False).order_by('aux_frame_index').first()
        
        payload = {
            'data': TimingDto(next_timing).data
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    def get_prev_timing(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        next_timing = Timing.objects.filter(aux_frame_index__lt=timing.aux_frame_index, is_disabled=False).order_by('aux_frame_index').first()
        
        payload = {
            'data': TimingDto(next_timing).data
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    def get_alternative_image_list(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse([], 'invalid timing uuid', False)
        
        return timing.alternative_image_list
    
    def get_timing_list_from_project(self, project_id=None):
        if project_id:
            timing_list = Timing.objects.filter(project_id=project_id, is_disabled=False).all()
        else:
            timing_list = Timing.objects.filter(is_disabled=False).all()
        
        payload = {
            'data': TimingDto(timing_list, many=True).data
        }
        
        return InternalResponse(payload, 'timing list fetched', True)
    
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
        
        payload = {
            'data': TimingDto(timing).data
        }
        
        return InternalResponse(payload, 'timing created successfully', True)
    
    def update_specific_timing(self, uuid, **kwargs):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        # TODO: handle foreign key update
        timing.update(**kwargs)
        return InternalResponse({}, 'timing updated successfully', True)
    
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

        payload = {
            'data': AppSettingDto(app_setting).data
        }
        
        return InternalResponse(payload, 'app_setting fetched successfully', True)
    
    def get_app_secrets_from_user_uuid(self, user_uuid):
        if user_uuid:
            user: User = User.objects.filter(uuid=user_uuid, is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            app_setting = AppSetting.objects.filter(user_id=user.id, is_disabled=False).first()
        else:
            app_setting = AppSetting.objects.filter(is_disabled=False).first()
        
        payload = {
            'data': {
                'aws_access_key': app_setting.aws_access_key_decrypted,
                'aws_secret_key': app_setting.aws_secret_access_key_decrypted,
                'replicate_key': app_setting.replicate_key_decrypted
            }
        }

        return InternalFileObject(payload, 'app_setting fetched successfully', True)
    
    def get_all_app_setting_list(self):
        app_setting_list = AppSetting.objects.filter(is_disabled=False).all()

        payload = {
            'data': AppSettingDto(app_setting_list, many=True).data
        }
        
        return InternalResponse(payload, 'app_setting list fetched successfully', True)
    
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
        
        payload = {
            'data': AppSettingDto(app_setting).data
        }
        
        return InternalResponse(payload, 'app_setting created successfully', True)
    

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
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)
    
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
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)
    
    def update_project_setting(self, project_id, **kwargs):
        setting = Setting.objects.filter(project_id=project_id, is_disabled=False).first()
        if not setting:
            return InternalResponse({}, 'invalid project', False)
        
        for attr, value in kwargs.items():
            setattr(setting, attr, value)
        setting.save()
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)

    def bulk_update_project_setting(self, **kwargs):
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
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)