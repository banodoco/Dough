import json
import os

import sys
sys.path.append('../')

import sqlite3
import subprocess
from typing import List
import uuid
from shared.constants import Colors, InternalFileType
from backend.serializers.dto import  AIModelDto, AppSettingDto, BackupDto, BackupListDto, InferenceLogDto, InternalFileDto, ProjectDto, SettingDto, TimingDto, UserDto

from shared.constants import AUTOMATIC_FILE_HOSTING, LOCAL_DATABASE_NAME, SERVER, ServerType
from shared.file_upload.s3 import upload_file

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from backend.models import AIModel, AIModelParamMap, AppSetting, BackupTiming, InferenceLog, InternalFileObject, Project, Setting, Timing, User

from backend.serializers.dao import CreateAIModelDao, CreateAIModelParamMapDao, CreateAppSettingDao, CreateFileDao, CreateInferenceLogDao, CreateProjectDao, CreateSettingDao, CreateTimingDao, CreateUserDao, UpdateAIModelDao, UpdateAppSettingDao, UpdateSettingDao
from shared.constants import InternalResponse



class DBRepo:
    def __init__(self):
        print("initializing database")
        database_file = LOCAL_DATABASE_NAME
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")

        from django.core.wsgi import get_wsgi_application
        application = get_wsgi_application()

        # creating db if not already present
        if not os.path.exists(database_file):
            from django.core.management import execute_from_command_line
            print(Colors.RED + "Database not present, creating" + Colors.RESET)
            conn = sqlite3.connect(database_file)
            conn.close()

            completed_process = subprocess.run(['python', 'manage.py', 'migrate'], capture_output=True, text=True)
            if completed_process.returncode == 0:
                print(Colors.BLUE + "Migrations completed successfully." + Colors.RESET)
            else:
                print(Colors.RED + "Migrations failed with an error." + Colors.RESET)
        else:
            print(Colors.BLUE + "Database already present" + Colors.RESET)

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
    
    def get_first_active_user(self):
        user = User.objects.filter(is_disabled=False).first()
        if not user:
            return InternalResponse(None, 'no user found', True)
        
        payload = {
            'data': UserDto(user).data
        }
        
        return InternalResponse(payload, 'user not found', False)
    
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
    
    def get_total_user_count(self):
        if SERVER != ServerType.PRODUCTION.value:
            count = User.objects.filter(is_disabled=False).count()
        else:
            count = 0
        
        return InternalResponse(count, 'user count fetched', True)
    
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
    
    def get_all_file_list(self, file_type: InternalFileType, tag=None):
        if tag:
            file_list = InternalFileObject.objects.filter(file_type=file_type.value, tag=tag, is_disabled=False).all()
        else:
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
    
    def remove_existing_timing(self, project_uuid):
        if project_uuid:
            project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        else:
            project: Project = Project.objects.filter(is_disabled=False).first()
        
        if project:
            Timing.objects.filter(project_id=project.id, is_disabled=False).update(is_disabled=True)
        
        return InternalResponse({}, 'timing removed successfully', True)
    
    def update_specific_timing(self, uuid, **kwargs):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        if 'primary_image' in kwargs:
            if kwargs['primary_image'] < len(timing.alternative_images_list):
                kwargs['primary_image_id'] = timing.alternative_images_list[kwargs['primary_image']].uuid
                del kwargs['primary_image']
        
        if 'primay_image_uuid' in kwargs:
            primay_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['primay_image_uuid'], is_disabled=False).first()
            if not primay_image:
                return InternalResponse({}, 'invalid primary image uuid', False)
            
            kwargs['primay_image_id'] = primay_image.id

        if 'model_uuid' in kwargs:
            model: AIModel = AIModel.objects.filter(uuid=kwargs['model_uuid'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model uuid', False)
        

        if 'source_image_uuid' in kwargs:
            source_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['source_image_uuid'], is_disabled=False).first()
            if not source_image:
                return InternalResponse({}, 'invalid source image uuid', False)
            
            kwargs['source_image_id'] = source_image.id
        

        if 'interpolated_clip_uuid' in kwargs:
            interpolated_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['interpolated_clip_uuid'], is_disabled=False).first()
            if not interpolated_clip:
                return InternalResponse({}, 'invalid interpolated clip uuid', False)
            
            kwargs['interpolated_clip_id'] = interpolated_clip.id
        

        if 'timed_clip_uuid' in kwargs:
            timed_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['timed_clip_uuid'], is_disabled=False).first()
            if not timed_clip:
                return InternalResponse({}, 'invalid timed clip uuid', False)
            
            kwargs['timed_clip_id'] = timed_clip.id
        

        if 'mask_uuid' in kwargs:
            mask: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['mask_uuid'], is_disabled=False).first()
            if not mask:
                return InternalResponse({}, 'invalid mask uuid', False)
            
            kwargs['mask_id'] = mask.id
        

        if 'canny_image_uuid' in kwargs:
            canny_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['canny_image_uuid'], is_disabled=False).first()
            if not canny_image:
                return InternalResponse({}, 'invalid canny image uuid', False)
            
            kwargs['canny_image_id'] = canny_image.id
        

        if 'preview_video_uuid' in kwargs:
            preview_video: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['preview_video_uuid'], is_disabled=False).first()
            if not preview_video:
                return InternalResponse({}, 'invalid preview video uuid', False)
            
            kwargs['preview_video_id'] = preview_video.id
        

        if 'primay_image_uuid' in kwargs:
            primay_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['primay_image_uuid'], is_disabled=False).first()
            if not primay_image:
                return InternalResponse({}, 'invalid primary image uuid', False)
            
            kwargs['primay_image_id'] = primay_image.id
        
        timing.update(**kwargs)
        return InternalResponse({}, 'timing updated successfully', True)
    
    def delete_timing_from_uuid(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.is_disabled = True
        timing.save()
        return InternalResponse({}, 'timing deleted successfully', True)

    def remove_primay_frame(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.primay_image_id = None
        timing.save()
        return InternalResponse({}, 'primay frame removed successfully', True)
    
    def remove_source_image(self, uuid):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.source_image_id = None
        timing.save()
        return InternalResponse({}, 'source image removed successfully', True)
    

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
    
    def update_app_setting(self, **kwargs):
        attributes = UpdateAppSettingDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        if 'uuid' in attributes.data and attributes.data['uuid']:
            app_setting = AppSetting.objects.filter(uuid=attributes.data['uuid'], is_disabled=False).first()
        else:
            app_setting = AppSetting.objects.filter(is_disabled=False).first()
        
        if 'user_id' in attributes.data and attributes.data['user_id']:
            user = User.objects.filter(uuid=attributes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            print(attributes.data)
            attributes._data['user_id'] = user.id

        app_setting.update(**attributes.data)

        return InternalResponse({}, 'app_setting updated successfully', True)

    
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
                'replicate_key': app_setting.replicate_key_decrypted,
                'replicate_username': app_setting.replicate_user_name
            }
        }

        return InternalResponse(payload, 'app_setting fetched successfully', True)
    
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
    def get_project_setting(self, project_uuid):
        project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project_id', False)
    
        setting = Setting.objects.filter(project_id=project.id, is_disabled=False).first()
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
    
    def update_project_setting(self, project_uuid, **kwargs):
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project', False)
        
        setting = Setting.objects.filter(project_id=project.id, is_disabled=False).first()
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
    
    
    # backup data
    def create_backup(self, project_uuid, backup_name):
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project', False)
        
        timing_list: List[Timing] = self.get_timing_list_from_project(project_uuid)
        
        # bulk fetching files and models from the database
        model_uuid_list = set()
        file_uuid_list = set()
        for timing in timing_list:
            model_uuid_list.add(timing.model.uuid)
            file_uuid_list.add(timing.source_image.uuid)
            file_uuid_list.add(timing.interpolated_clip.uuid)
            file_uuid_list.add(timing.timed_clip.uuid)
            file_uuid_list.add(timing.mask.uuid)
            file_uuid_list.add(timing.canny_image.uuid)
            file_uuid_list.add(timing.preview_video.uuid)
            file_uuid_list.add(timing.primary_image.uuid)
        
        model_uuid_list = list(model_uuid_list)
        file_uuid_list = list(file_uuid_list)
        
        # fetch the models and files from the database
        model_list = AIModel.objects.filter(uuid__in=model_uuid_list, is_disabled=False).all()
        file_list = InternalFileObject.objects.filter(uuid__in=file_uuid_list, is_disabled=False).all()
        id_model_dict, id_file_dict = {}, {}

        for model in model_list:
            id_model_dict[model.uuid] = model
        
        for file in file_list:
            id_file_dict[file.uuid] = file

        # replacing ids (foreign keys) with uuids
        final_list = list(timing_list.values())
        for timing in final_list:
            timing['model_uuid'] = id_model_dict[timing['model_id']]['uuid']
            del timing['model_id']

            timing['source_image_uuid'] = id_file_dict[timing['source_image_id']]['uuid']
            del timing['source_image_id']

            timing['interpolated_clip_uuid'] = id_file_dict[timing['interpolated_clip_id']]['uuid']
            del timing['interpolated_clip_id']

            timing['timed_clip_uuid'] = id_file_dict[timing['timed_clip_id']]['uuid']
            del timing['timed_clip_id']

            timing['mask_uuid'] = id_file_dict[timing['mask_id']]['uuid']
            del timing['mask_id']

            timing['canny_image_uuid'] = id_file_dict[timing['canny_image_id']]['uuid']
            del timing['canny_image_id']

            timing['preview_video_uuid'] = id_file_dict[timing['preview_video_id']]['uuid']
            del timing['preview_video_id']

            timing['primary_image_uuid'] = id_file_dict[timing['primary_image_id']]['uuid']
            del timing['primary_image_id']


        serialized_data = json.dumps(list(final_list.values()))
        backup_data = {
            "name" : backup_name,
            "project_id" : project.id,
            "note" : "",
            "data_dump" : serialized_data
        }
        backup = BackupTiming.objects.create(**backup_data)
        
        payload = {
            'data': BackupDto(backup).data
        }

        return InternalResponse(payload, 'backup created', True)
    
    def get_backup_from_uuid(self, backup_uuid):
        backup: BackupTiming = BackupTiming.objects.filter(uuid=backup_uuid, is_disabled=False).first()
        if not backup:
            return InternalResponse({}, 'invalid backup', False)
        
        payload = {
            'data': BackupDto(backup).data
        }

        return InternalResponse(payload, 'backup fetched', True)
    
    def get_backup_list(self, project_uuid):
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project', False)
        
        backup_list = BackupTiming.objects.filter(project_id=project.id, is_disabled=False).all()
        
        payload = {
            'data': BackupListDto(backup_list, many=True).data
        }

        return InternalResponse(payload, 'backup list fetched', True)
    
    def delete_backup(self, backup_uuid):
        backup: BackupTiming = BackupTiming.objects.filter(uuid=backup_uuid, is_disabled=False).first()
        if not backup:
            return InternalResponse({}, 'invalid backup', False)
        
        backup.is_disabled = True
        backup.save()
        
        return InternalResponse({}, 'backup deleted', True)
    
    def restore_backup(self, backup_uuid: str):
        backup: BackupTiming = self.get_backup_from_uuid(backup_uuid)

        current_timing_list: List[Timing] = self.get_timing_list_from_project(backup.project.uuid)
        backup_data = backup.data_dump_dict     # contains a list of dict of backed up timings

        if not backup_data:
            return InternalResponse({}, 'no backup data', False)
        
        for timing in current_timing_list:
            matching_timing_list = [item for item in backup_data if item['uuid'] == str(timing.uuid)]

            if len(matching_timing_list):
                backup_timing = matching_timing_list[0]

                self.update_specific_timing(
                    timing.uuid,
                    model_uuid=backup_timing['model_uuid'],
                    source_image_uuid=backup_timing['source_image_uuid'],
                    interpolated_clip=backup_timing['interpolated_clip_uuid'],
                    timed_clip=backup_timing['timed_clip_uuid'],
                    mask=backup_timing['mask_uuid'],
                    canny_image=backup_timing['canny_image_uuid'],
                    preview_video=backup_timing['preview_video_uuid'],
                    primary_image=backup_timing['primary_image_uuid'],
                    custom_model_id_list=backup_timing['custom_model_id_list'],
                    frame_time=backup_timing['frame_time'],
                    frame_number=backup_timing['frame_number'],
                    alternative_images=backup_timing['alternative_images'],
                    custom_pipeline=backup_timing['custom_pipeline'],
                    prompt=backup_timing['prompt'],
                    negative_prompt=backup_timing['negative_prompt'],
                    guidance_scale=backup_timing['guidance_scale'],
                    seed=backup_timing['seed'],
                    num_inteference_steps=backup_timing['num_inteference_steps'],
                    strength=backup_timing['strength'],
                    notes=backup_timing['notes'],
                    adapter_type=backup_timing['adapter_type'],
                    clip_duration=backup_timing['clip_duration'],
                    animation_style=backup_timing['animation_style'],
                    interpolation_steps=backup_timing['interpolation_steps'],
                    low_threshold=backup_timing['low_threshold'],
                    high_threshold=backup_timing['high_threshold'],
                    aux_frame_index=backup_timing['aux_frame_index']
                )

