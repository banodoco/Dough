import inspect
import json
import os

import sys
from shared.logging.constants import LoggingType

from shared.logging.logging import AppLogger
from utils.common_decorators import count_calls, measure_execution_time
sys.path.append('../')

import sqlite3
import subprocess
from typing import List
import uuid
from shared.constants import Colors, InternalFileType
from backend.serializers.dto import  AIModelDto, AppSettingDto, BackupDto, BackupListDto, InferenceLogDto, InternalFileDto, ProjectDto, SettingDto, TimingDto, UserDto

from shared.constants import AUTOMATIC_FILE_HOSTING, LOCAL_DATABASE_NAME, SERVER, ServerType
from shared.file_upload.s3 import upload_file, upload_file_from_obj

from backend.models import AIModel, AIModelParamMap, AppSetting, BackupTiming, InferenceLog, InternalFileObject, Project, Setting, Timing, User

from backend.serializers.dao import CreateAIModelDao, CreateAIModelParamMapDao, CreateAppSettingDao, CreateFileDao, CreateInferenceLogDao, CreateProjectDao, CreateSettingDao, CreateTimingDao, CreateUserDao, UpdateAIModelDao, UpdateAppSettingDao, UpdateSettingDao
from shared.constants import InternalResponse
from django.db.models import F

logger = AppLogger()

class DBRepo:
    _instance = None
    _count = 0
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            database_file = LOCAL_DATABASE_NAME
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_settings")

            # creating db if not already present
            if not os.path.exists(database_file):
                from django.core.management import execute_from_command_line
                logger.log(LoggingType.INFO,  "Database not found. Creating new one.")
                conn = sqlite3.connect(database_file)
                conn.close()

                completed_process = subprocess.run(['python', 'manage.py', 'migrate'], capture_output=True, text=True)
                if completed_process.returncode == 0:
                    logger.log(LoggingType.INFO, "Migrations completed successfully")
                else:
                    logger.log(LoggingType.ERROR, "Migrations failed")
            else:
                # logger.log(LoggingType.INFO, "Database already present")
                pass

            self._initialized = True

    # user operations
    def create_user(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
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
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        user = User.objects.filter(is_disabled=False).first()
        if not user:
            return InternalResponse(None, 'no user found', True)
        
        payload = {
            'data': UserDto(user).data
        }
        
        return InternalResponse(payload, 'user found', True)
    
    def get_user_by_email(self, email):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        user = User.objects.filter(email=email, is_disabled=False).first()
        if user:
            return InternalResponse(user, 'user found', True)
        
        payload = {
            'data': UserDto(user).data
        }
        
        return InternalResponse(payload, 'user not found', False)
    
    def update_user(self, user_id, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if user_id:
            user = User.objects.filter(uuid=user_id, is_disabled=False).first()
        else:
            user = User.objects.filter(is_disabled=False).first()
            
        if not user:
            return InternalResponse({}, 'invalid user id', False)
        
        if 'credits_to_add' in kwargs:
            # credits won't change in the local environment
            kwargs['total_credits'] = 1000     # max(user.total_credits + kwargs['credits_to_add'], 0)

        for k,v in kwargs.items():
            setattr(user, k, v)

        user.save()
        payload = {
            'data': UserDto(user).data
        }

        return InternalResponse(payload, 'user updated successfully', True)

    def get_all_user_list(self):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        user_list = User.objects.all()

        payload = {
            'data': UserDto(user_list, many=True).data
        }
        return InternalResponse(payload, 'user list', True)
    
    def get_total_user_count(self):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if SERVER != ServerType.PRODUCTION.value:
            count = User.objects.filter(is_disabled=False).count()
        else:
            count = 0
        
        return InternalResponse(count, 'user count fetched', True)
    
    def delete_user_by_email(self, email):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
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
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        file = InternalFileObject.objects.filter(name=name, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'file not found', False)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)

    def get_file_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        file = InternalFileObject.objects.filter(uuid=uuid, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'file not found', False)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)
    
    # TODO: create a dao for this
    def get_all_file_list(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        kwargs['is_disabled'] = False

        if 'project_id' in kwargs and kwargs['project_id']:
            project = Project.objects.filter(uuid=kwargs['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'project not found', False)

            kwargs['project_id'] = project.id

        file_list = InternalFileObject.objects.filter(**kwargs).all()
        
        payload = {
            'data': InternalFileDto(file_list, many=True).data
        }

        return InternalResponse(payload, 'file found', True)
    
    def create_or_update_file(self, file_uuid, type=InternalFileType.IMAGE.value, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        file = InternalFileType.objects.filter(uuid=file_uuid, type=type, is_disabled=False).first()
        if not file:
            file = InternalFileObject.objects.create(uuid=file_uuid, name=str(uuid.uuid4()), file_type=type, **kwargs)
        else:
            kwargs['file_type'] = type

            for attr, value in kwargs.items():
                setattr(file, attr, value)
            file.save()
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)
    
    def upload_file(self, file, ext):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        url = upload_file_from_obj(file, ext)
        payload = {
            'data': url
        }

        return InternalResponse(payload, 'file uploaded', True)
    
    def create_file(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        data = CreateFileDao(data=kwargs)
        if not data.is_valid():
            return InternalResponse({}, data.errors, False)

        print(data.data)
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
            
            data._data['hosted_url'] = hosted_url

        if 'project_id' in kwargs and kwargs['project_id']:
            project = Project.objects.filter(uuid=kwargs['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            data._data['project_id'] = project.id

        if 'inference_log_id' in kwargs and kwargs['inference_log_id']:
            inference_log = InferenceLog.objects.filter(uuid=kwargs['inference_log_id'], is_disabled=False).first()
            if not inference_log:
                return InternalResponse({}, 'invalid log id', False)
            
            data._data['inference_log_id'] = inference_log.id
        

        if not data.is_valid():
            return InternalResponse({}, data.errors, False)
        
        file = InternalFileObject.objects.create(**data.data)
        
        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file found', True)
    
    def delete_file_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        file = InternalFileObject.objects.filter(uuid=uuid, is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'invalid file uuid', False)
        
        return InternalResponse({}, 'file deleted successfully', True)
    
    def get_image_list_from_uuid_list(self, uuid_list, file_type=InternalFileType.IMAGE.value):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        file_list = InternalFileObject.objects.filter(uuid__in=uuid_list, \
                                                      is_disabled=False, type=file_type).all()
        
        if file_list and len(file_list):
            uuid_dict = {str(obj.uuid): obj for obj in file_list}
            file_list = [uuid_dict[uuid] for uuid in uuid_list if uuid in uuid_dict]

        payload = {
            'data': InternalFileDto(file_list, many=True).data
        }
        
        return InternalResponse(payload, 'file list fetched', True)
    
    def update_file(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if 'uuid' not in kwargs:
            return InternalResponse({}, 'uuid is required', False)
        
        file = InternalFileObject.objects.filter(uuid=kwargs['uuid'], is_disabled=False).first()
        if not file:
            return InternalResponse({}, 'invalid file uuid', False)
        
        if 'project_id' in kwargs:
            project = Project.objects.filter(uuid=kwargs['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            kwargs['project_id'] = project.id

        if 'inference_log_id' in kwargs and kwargs['inference_log_id']:
            inference_log = InferenceLog.objects.filter(uuid=kwargs['inference_log_id'], is_disabled=False).first()
            if not inference_log:
                return InternalResponse({}, 'invalid log id', False)
            
            kwargs['inference_log_id'] = inference_log.id
        
        for k,v in kwargs.items():
            setattr(file, k, v)
        
        file.save()

        payload = {
            'data': InternalFileDto(file).data
        }

        return InternalResponse(payload, 'file updated successfully', True)
    
    # project
    def get_project_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project = Project.objects.filter(uuid=uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project uuid', False)
        
        payload = {
            'data': ProjectDto(project).data
        }
        
        return InternalResponse(payload, 'project fetched', True)
    
    def update_project(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project = Project.objects.filter(uuid=kwargs['uuid'], is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project uuid', False)
        
        for k,v in kwargs.items():
            setattr(project, k, v)

        project.save()

        payload = {
            'data': ProjectDto(project).data
        }

        return InternalResponse(payload, 'successfully updated project', True)
    
    def get_all_project_list(self, user_uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        user: User = User.objects.filter(uuid=user_uuid, is_disabled=False).first()
        if not user:
            return InternalResponse({}, 'invalid user', False)
        
        project_list = Project.objects.filter(user_id=user.id, is_disabled=False).all()
        
        payload = {
            'data': ProjectDto(project_list, many=True).data
        }
        
        return InternalResponse(payload, 'project fetched', True)
    
    def create_project(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
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
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project = Project.objects.filter(uuid=uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project uuid', False)
        
        project.is_disabled = True
        project.save()
        return InternalResponse({}, 'project deleted successfully', True)

    
    # ai model (custom ai model)
    def get_ai_model_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        ai_model = AIModel.objects.filter(uuid=uuid, is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model uuid', False)
        
        payload = {
            'data': AIModelDto(ai_model).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
    def get_ai_model_from_name(self, name):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        ai_model = AIModel.objects.filter(replicate_url=name, is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model name', False)

        payload = {
            'data': AIModelDto(ai_model).data
        }

        return InternalResponse(payload, 'ai_model fetched', True)
    
    def get_all_ai_model_list(self, model_category_list=None, user_id=None, custom_trained=False, model_type_list=None):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        query = {'custom_trained': "all" if custom_trained == None else ("user" if custom_trained else "predefined"), 'is_disabled': False}
        if user_id:
            user = User.objects.filter(uuid=user_id, is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            query['user_id'] = user.id

        query['custom_trained'] = custom_trained
            
        ai_model_list = AIModel.objects.filter(**query).all()

        filtered_list = []
        for model in ai_model_list:
            category_check = True if (not model_category_list or (model_category_list and model.category in model_category_list)) else False
            type_check = True if (not model_type_list or (model_type_list and any(item in model_type_list for item in json.loads(model.model_type)))) else False

            if category_check and type_check:
                filtered_list.append(model)
        
        payload = {
            'data': AIModelDto(filtered_list, many=True).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
    def create_ai_model(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = CreateAIModelDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
        if 'user_id' in attributes.data and attributes.data['user_id']:
            user = User.objects.filter(uuid=attributes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            print(attributes.data['user_id'])
            attributes._data['user_id'] = user.id
        
        ai_model = AIModel.objects.create(**attributes.data)
        
        payload = {
            'data': AIModelDto(ai_model).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
    def update_ai_model(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = UpdateAIModelDao(attributes=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        ai_model = AIModel.objects.filter(uuid=attributes.data['uuid'], is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model uuid', False)
        
        if 'user_id' in attributes.data and attributes.data['user_id']:
            user = User.objects.filter(uuid=attributes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            print(attributes.data['user_id'])
            attributes.data['user_id'] = user.id
        
        for attr, value in attributes.data.items():
            setattr(ai_model, attr, value)
        ai_model.save()
        
        payload = {
            'data': AIModelDto(ai_model).data
        }
        
        return InternalResponse(payload, 'ai_model fetched', True)
    
    def delete_ai_model_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        ai_model = AIModel.objects.filter(uuid=uuid, is_disabled=False).first()
        if not ai_model:
            return InternalResponse({}, 'invalid ai model uuid', False)
        
        ai_model.is_disabled = True
        ai_model.save()
        return InternalResponse({}, 'ai model deleted successfully', True)
    

    # inference log
    def get_inference_log_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        log = InferenceLog.objects.filter(uuid=uuid, is_disabled=False).first()
        if not log:
            return InternalResponse({}, 'invalid inference log uuid', False)
        
        payload = {
            'data': InferenceLogDto(log).data
        }
        
        return InternalResponse(payload, 'inference log fetched', True)
    
    def get_all_inference_log_list(self, project_id=None):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if project_id:
            project = Project.objects.filter(uuid=project_id, is_disabled=False).first()
            log_list = InferenceLog.objects.filter(project_id=project.id, is_disabled=False).all()
        else:
            log_list = InferenceLog.objects.filter(is_disabled=False).all()
        
        payload = {
            'data': InferenceLogDto(log_list, many=True).data
        }
        
        return InternalResponse(payload, 'inference log list fetched', True)
    
    def create_inference_log(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = CreateInferenceLogDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            attributes._data['project_id'] = project.id
        
        if 'model_id' in attributes.data and attributes.data['model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes._data['model_id'] = model.id

        log = InferenceLog.objects.create(**attributes.data)

        
        payload = {
            'data': InferenceLogDto(log).data
        }
        
        return InternalResponse(payload, 'inference log created successfully', True)
    
    def delete_inference_log_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        log = InferenceLog.objects.filter(uuid=uuid, is_disabled=False).first()
        if not log:
            return InternalResponse({}, 'invalid inference log uuid', False)
        
        log.is_disabled = True
        log.save()

        return InternalResponse({}, 'inference log deleted successfully', True)
    
    def update_inference_log(self, uuid, **kwargs):
        log = InferenceLog.objects.filter(uuid=uuid, is_disabled=False).first()
        if not log:
            return InternalResponse({}, 'invalid inference log uuid', False)
        
        for attr, value in kwargs.items():
            setattr(log, attr, value)
        log.save()

        payload = {
            'data': InferenceLogDto(log).data
        }

        return InternalResponse(payload, 'inference log updated successfully', True)

    # ai model param map
    # TODO: add DTO in the output
    def get_ai_model_param_map_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        map = AIModelParamMap.objects.filter(uuid=uuid, is_disabled=False).first()
        if not map:
            return InternalResponse({}, 'invalid ai model param map uuid', False)
        
        return InternalResponse(map, 'ai model param map fetched', True)
    
    def get_all_ai_model_param_map_list(self, model_id=None):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if model_id:
            map_list = AIModelParamMap.objects.filter(model_id=model_id, is_disabled=False).all()
        else:
            map_list = AIModelParamMap.objects.filter(is_disabled=False).all()
        
        return InternalResponse(map_list, 'ai model param map list fetched', True)
    
    def create_ai_model_param_map(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = CreateAIModelParamMapDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
        if 'model_id' in attributes.data and attributes.data['model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes._data['model_id'] = model.id
        
        map = AIModelParamMap.objects.create(**attributes.data)
        
        return InternalResponse(map, 'ai model param map created successfully', True)
    
    def delete_ai_model(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        map = AIModelParamMap.objects.filter(uuid=uuid, is_disabled=False).first()
        if not map:
            return InternalResponse({}, 'invalid ai model param map uuid', False)
        
        map.is_disabled = True
        map.save()
        return InternalResponse({}, 'ai model param map deleted successfully', True)
    

    # timing
    def get_timing_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({'data': None}, 'invalid timing uuid', False)
        
        payload = {
            'data': TimingDto(timing).data
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    def get_timing_from_frame_number(self, project_uuid, frame_number):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if project:
            timing = Timing.objects.filter(aux_frame_index=frame_number, project_id=project.id, is_disabled=False).first()
            if timing:
                payload = {
                    'data': TimingDto(timing).data
                }
                
                return InternalResponse(payload, 'timing fetched', True)
            
        return InternalResponse({'data': None}, 'invalid timing frame number', False)
    
    def get_primary_variant_location(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)

        payload = {
            'data': timing.primary_variant_location
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    # this is based on the aux_frame_index and not the order in the db
    def get_next_timing(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        next_timing = Timing.objects.filter(aux_frame_index=timing.aux_frame_index + 1, project_id=timing.project_id, is_disabled=False).order_by('aux_frame_index').first()
        
        payload = {
            'data': TimingDto(next_timing).data if next_timing else None
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    def get_prev_timing(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        prev_timing = Timing.objects.filter(aux_frame_index=timing.aux_frame_index - 1, project_id=timing.project_id, is_disabled=False).order_by('aux_frame_index').first()
        
        payload = {
            'data': TimingDto(prev_timing).data if prev_timing else None
        }
        
        return InternalResponse(payload, 'timing fetched', True)
    
    def get_alternative_image_list(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse([], 'invalid timing uuid', False)
        
        return timing.alternative_image_list
    
    def get_timing_list_from_project(self, project_uuid=None):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if project_uuid:
            project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            timing_list = Timing.objects.filter(project_id=project.id, is_disabled=False).order_by('aux_frame_index').all()
        else:
            timing_list = Timing.objects.filter(is_disabled=False).order_by('aux_frame_index').all()
        
        payload = {
            'data': TimingDto(timing_list, many=True).data
        }
        
        return InternalResponse(payload, 'timing list fetched', True)
    
    def create_timing(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = CreateTimingDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            print(attributes.data)
            attributes._data['project_id'] = project.id
        
        if 'aux_frame_index' not in attributes.data or attributes.data['aux_frame_index'] == None: 
            attributes._data['aux_frame_index'] = Timing.objects.filter(project_id=attributes.data['project_id'], is_disabled=False).count()
        
        if 'model_id' in attributes.data:
            if attributes.data['model_id'] != None:
                model: AIModel = AIModel.objects.filter(uuid=attributes.data['model_id'], is_disabled=False).first()
                if not model:
                    return InternalResponse({}, 'invalid model uuid', False)
                
                attributes._data['model_id'] = model.id
        

        if 'source_image_id' in attributes.data:
            if attributes.data['source_image_id'] != None:
                source_image: InternalFileObject = InternalFileObject.objects.filter(uuid=attributes.data['source_image_id'], is_disabled=False).first()
                if not source_image:
                    return InternalResponse({}, 'invalid source image uuid', False)
                
                attributes._data['source_image_id'] = source_image.id
        

        if 'interpolated_clip_list' in attributes.data and attributes.data['interpolated_clip_list'] != None:
            for clip_uuid in attributes.data['interpolated_clip_list']:
                interpolated_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=clip_uuid, is_disabled=False).first()
                if not interpolated_clip:
                    return InternalResponse({}, 'invalid interpolated clip uuid', False)
                
                attributes._data['interpolated_clip_list'] = list(set(attributes._data['interpolated_clip_list']))
        

        if 'timed_clip_id' in attributes.data:
            if attributes.data['timed_clip_id'] != None:
                timed_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=attributes.data['timed_clip_id'], is_disabled=False).first()
                if not timed_clip:
                    return InternalResponse({}, 'invalid timed clip uuid', False)
                
                attributes._data['timed_clip_id'] = timed_clip.id
        

        if 'mask_id' in attributes.data:
            if attributes.data['mask_id'] != None:
                mask: InternalFileObject = InternalFileObject.objects.filter(uuid=attributes.data['mask_id'], is_disabled=False).first()
                if not mask:
                    return InternalResponse({}, 'invalid mask uuid', False)
                
                attributes._data['mask_id'] = mask.id
        

        if 'canny_image_id' in attributes.data:
            if attributes.data['canny_image_id'] != None:
                canny_image: InternalFileObject = InternalFileObject.objects.filter(uuid=attributes.data['canny_image_id'], is_disabled=False).first()
                if not canny_image:
                    return InternalResponse({}, 'invalid canny image uuid', False)
                
                attributes._data['canny_image_id'] = canny_image.id
        

        if 'preview_video_id' in attributes.data:
            if attributes.data['preview_video_id'] != None:
                preview_video: InternalFileObject = InternalFileObject.objects.filter(uuid=attributes.data['preview_video_id'], is_disabled=False).first()
                if not preview_video:
                    return InternalResponse({}, 'invalid preview video uuid', False)
                
                attributes._data['preview_video_id'] = preview_video.id
        

        if 'primay_image_id' in attributes.data:
            if attributes.data['primay_image_id'] != None:
                primay_image: InternalFileObject = InternalFileObject.objects.filter(uuid=attributes.data['primay_image_id'], is_disabled=False).first()
                if not primay_image:
                    return InternalResponse({}, 'invalid primary image uuid', False)
                
                attributes._data['primay_image_id'] = primay_image.id
        
        
        timing = Timing.objects.create(**attributes.data)
        
        payload = {
            'data': TimingDto(timing).data
        }
        
        return InternalResponse(payload, 'timing created successfully', True)
    
    def remove_existing_timing(self, project_uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if project_uuid:
            project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        else:
            project: Project = Project.objects.filter(is_disabled=False).first()
        
        if project:
            Timing.objects.filter(project_id=project.id, is_disabled=False).update(is_disabled=True)
        
        return InternalResponse({}, 'timing removed successfully', True)
    
    def add_interpolated_clip(self, uuid, **kwargs):
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        if 'interpolated_clip_id' in kwargs and kwargs['interpolated_clip_id'] != None:
            interpolated_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['interpolated_clip_id'], is_disabled=False).first()
            if not interpolated_clip:
                return InternalResponse({}, 'invalid interpolated clip uuid', False)
                
            timing.add_interpolated_clip_list([interpolated_clip.uuid.hex])
            timing.save()

        return InternalResponse({}, 'success', True)
    
    # TODO: add dao in this method
    def update_specific_timing(self, uuid, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        if 'primary_image_id' in kwargs:
            if kwargs['primary_image_id'] != None:
                primary_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['primary_image_id'], is_disabled=False).first()
                if not primary_image:
                    return InternalResponse({}, 'invalid primary image uuid', False)
                
                kwargs['primary_image_id'] = primary_image.id

        if 'model_id' in kwargs:
            if kwargs['model_id'] != None:
                model: AIModel = AIModel.objects.filter(uuid=kwargs['model_id'], is_disabled=False).first()
                if not model:
                    return InternalResponse({}, 'invalid model uuid', False)
                
                kwargs['model_id'] = model.id
        

        if 'source_image_id' in kwargs:
            if kwargs['source_image_id'] != None:
                source_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['source_image_id'], is_disabled=False).first()
                if not source_image:
                    return InternalResponse({}, 'invalid source image uuid', False)
                
                kwargs['source_image_id'] = source_image.id
        

        if 'interpolated_clip_list' in kwargs and kwargs['interpolated_clip_list'] != None:
            cur_list = []
            for clip_uuid in kwargs['interpolated_clip_list']:
                interpolated_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=clip_uuid, is_disabled=False).first()
                if not interpolated_clip:
                    return InternalResponse({}, 'invalid interpolated clip uuid', False)
                
                cur_list.append(interpolated_clip.uuid)
            kwargs['interpolated_clip_list'] = list(set(kwargs['interpolated_clip_list']))
        

        if 'timed_clip_id' in kwargs:
            if kwargs['timed_clip_id'] != None:
                timed_clip: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['timed_clip_id'], is_disabled=False).first()
                if not timed_clip:
                    return InternalResponse({}, 'invalid timed clip uuid', False)
                
                kwargs['timed_clip_id'] = timed_clip.id
        

        if 'mask_id' in kwargs:
            if kwargs['mask_id'] != None:
                mask: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['mask_id'], is_disabled=False).first()
                if not mask:
                    return InternalResponse({}, 'invalid mask uuid', False)
                
                kwargs['mask_id'] = mask.id
        

        if 'canny_image_id' in kwargs:
            if kwargs['canny_image_id'] != None:
                canny_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['canny_image_id'], is_disabled=False).first()
                if not canny_image:
                    return InternalResponse({}, 'invalid canny image uuid', False)
                
                kwargs['canny_image_id'] = canny_image.id
        

        if 'preview_video_id' in kwargs:
            if kwargs['preview_video_id'] != None:
                preview_video: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['preview_video_id'], is_disabled=False).first()
                if not preview_video:
                    return InternalResponse({}, 'invalid preview video uuid', False)
                
                kwargs['preview_video_id'] = preview_video.id
        

        if 'primay_image_id' in kwargs:
            if kwargs['primay_image_id'] != None:
                primay_image: InternalFileObject = InternalFileObject.objects.filter(uuid=kwargs['primay_image_id'], is_disabled=False).first()
                if not primay_image:
                    return InternalResponse({}, 'invalid primary image uuid', False)
                
                kwargs['primay_image_id'] = primay_image.id
        
        for attr, value in kwargs.items():
            setattr(timing, attr, value)
        timing.save()

        payload = {}

        return InternalResponse(payload, 'timing updated successfully', True)
    
    def delete_timing_from_uuid(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.is_disabled = True
        timing.save()
        return InternalResponse({}, 'timing deleted successfully', True)

    def remove_primary_frame(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.primay_image_id = None
        timing.save()
        return InternalResponse({}, 'primay frame removed successfully', True)
    
    def remove_source_image(self, uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        timing = Timing.objects.filter(uuid=uuid, is_disabled=False).first()
        if not timing:
            return InternalResponse({}, 'invalid timing uuid', False)
        
        timing.source_image_id = None
        timing.save()
        return InternalResponse({}, 'source image removed successfully', True)
    
    def move_frame_one_step_forward(self, project_uuid, index_of_frame):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project uuid', False)
        
        timing_list = Timing.objects.filter(project_id=project.id, \
                                            aux_frame_index__gte=index_of_frame, is_disabled=False).order_by('frame_number')
        
        timing_list.update(aux_frame_index=F('aux_frame_index') + 1)

        return InternalResponse({}, 'frames moved successfully', True)
    

    # app setting
    def get_app_setting_from_uuid(self, uuid=None):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        if uuid:
            app_setting = AppSetting.objects.filter(uuid=uuid, is_disabled=False).first()
        else:
            app_setting = AppSetting.objects.filter(is_disabled=False).first()

        payload = {
            'data': AppSettingDto(app_setting).data
        }
        
        return InternalResponse(payload, 'app_setting fetched successfully', True)
    
    def update_app_setting(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = UpdateAppSettingDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
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

        for attr, value in attributes.data.items():
            setattr(app_setting, attr, value)
        app_setting.save()

        return InternalResponse({}, 'app_setting updated successfully', True)

    
    def get_app_secrets_from_user_uuid(self, user_uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
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
                'replicate_username': app_setting.replicate_username
            }
        }

        return InternalResponse(payload, 'app_setting fetched successfully', True)
    
    def get_all_app_setting_list(self):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        app_setting_list = AppSetting.objects.filter(is_disabled=False).all()

        payload = {
            'data': AppSettingDto(app_setting_list, many=True).data
        }
        
        return InternalResponse(payload, 'app_setting list fetched successfully', True)
    
    def create_app_setting(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = CreateAppSettingDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
        if 'user_id' in attributes.data and attributes.data['user_id']:
            user = User.objects.filter(uuid=attributes.data['user_id'], is_disabled=False).first()
            if not user:
                return InternalResponse({}, 'invalid user', False)
            
            attributes._data['user_id'] = user.id
        
        app_setting = AppSetting.objects.create(**attributes.data)
        
        payload = {
            'data': AppSettingDto(app_setting).data
        }
        
        return InternalResponse(payload, 'app_setting created successfully', True)
    

    def delete_app_setting(self, user_id):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
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
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
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
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = CreateSettingDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        print(attributes.data)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            attributes._data['project_id'] = project.id

        if 'default_model_id' in attributes.data and attributes.data['default_model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['default_model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes._data['default_model_id'] = model.id

        if "audio_id" in attributes.data and attributes.data["audio_id"]:
            audio = InternalFileObject.objects.filter(uuid=attributes.data["audio_id"], is_disabled=False).first()
            if not audio:
                return InternalResponse({}, 'invalid audio', False)
            
            attributes._data["audio_id"] = audio.id
    
        if "input_video_id" in attributes.data and attributes.data["input_video_id"]:
            video = InternalFileObject.objects.filter(uuid=attributes.data["input_video_id"], is_disabled=False).first()
            if not video:
                return InternalResponse({}, 'invalid video', False)
            
            attributes._data["input_video_id"] = video.id
        
        setting = Setting.objects.create(**attributes.data)
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)
    
    def update_project_setting(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = UpdateSettingDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        project: Project = Project.objects.filter(uuid=kwargs['project_id'], is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project', False)
        
        print(attributes.data)
        attributes._data['project_id'] = project.id
        
        setting = Setting.objects.filter(project_id=project.id, is_disabled=False).first()
        if not setting:
            return InternalResponse({}, 'invalid project', False)

        if 'default_model_id' in attributes.data and attributes.data['default_model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['default_model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes._data['default_model_id'] = model.id

        if "audio_id" in attributes.data and attributes.data["audio_id"]:
            audio = InternalFileObject.objects.filter(uuid=attributes.data["audio_id"], is_disabled=False).first()
            if not audio:
                return InternalResponse({}, 'invalid audio', False)
            
            attributes._data["audio_id"] = audio.id
    
        if "input_video_id" in attributes.data and attributes.data["input_video_id"]:
            video = InternalFileObject.objects.filter(uuid=attributes.data["input_video_id"], is_disabled=False).first()
            if not video:
                return InternalResponse({}, 'invalid video', False)
            
            attributes._data["input_video_id"] = video.id

        if 'model_id' in attributes.data and attributes.data['model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes._data['model_id'] = model.id
        
        for attr, value in attributes.data.items():
            setattr(setting, attr, value)
        setting.save()
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)

    def bulk_update_project_setting(self, **kwargs):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        attributes = UpdateSettingDao(data=kwargs)
        if not attributes.is_valid():
            return InternalResponse({}, attributes.errors, False)
        
        setting = Setting.objects.filter(uuid=attributes.data['uuid'], is_disabled=False).first()
        if not setting:
            return InternalResponse({}, 'invalid project', False)
        
        print(attributes.data)
        
        if 'project_id' in attributes.data and attributes.data['project_id']:
            project = Project.objects.filter(uuid=attributes.data['project_id'], is_disabled=False).first()
            if not project:
                return InternalResponse({}, 'invalid project', False)
            
            print(attributes.data)
            attributes._data['project_id'] = project.id
        
        if 'default_model_id' in attributes.data and attributes.data['default_model_id']:
            model = AIModel.objects.filter(uuid=attributes.data['default_model_id'], is_disabled=False).first()
            if not model:
                return InternalResponse({}, 'invalid model', False)
            
            attributes._data['default_model_id'] = model.id

        if 'audio_id' in attributes.data and attributes.data['audio_id']:
            audio = InternalFileObject.objects.filter(uuid=attributes.data['audio_id'], is_disabled=False).first()
            if not audio:
                return InternalResponse({}, 'invalid audio', False)
            
            attributes._data['audio_id'] = audio.id

        if 'input_video_id' in attributes.data and attributes.data['input_video_id']:
            video = InternalFileObject.objects.filter(uuid=attributes.data['input_video_id'], is_disabled=False).first()
            if not video:
                return InternalResponse({}, 'invalid video', False)
            
            attributes._data['input_video_id'] = video.id

        for attr, value in attributes.data.items():
            setattr(setting, attr, value)
        setting.save()
        
        payload = {
            'data': SettingDto(setting).data
        }

        return InternalResponse(payload, 'setting fetched', True)
    
    
    # backup data
    def create_backup(self, project_uuid, backup_name):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project', False)
        
        timing_list = Timing.objects.filter(project_id=project.id, is_disabled=False).order_by('aux_frame_index').all()
        
        # bulk fetching files and models from the database
        model_uuid_list = set()
        file_uuid_list = set()
        for timing in timing_list:
            if timing.model:
                model_uuid_list.add(timing.model.uuid)
            
            if timing.source_image:
                file_uuid_list.add(timing.source_image.uuid)

            if timing.interpolated_clip_list:
                file_uuid_list.extend(json.loads(timing.interpolated_clip_list))
            
            if timing.timed_clip:
                file_uuid_list.add(timing.timed_clip.uuid)
            
            if timing.mask:
                file_uuid_list.add(timing.mask.uuid)
            
            if timing.canny_image:
                file_uuid_list.add(timing.canny_image.uuid)
            
            if timing.preview_video:
                file_uuid_list.add(timing.preview_video.uuid)
            
            if timing.primary_image:
                file_uuid_list.add(timing.primary_image.uuid)
        
        model_uuid_list = list(model_uuid_list)
        file_uuid_list = list(file_uuid_list)
        
        # fetch the models and files from the database
        model_list = AIModel.objects.filter(uuid__in=model_uuid_list, is_disabled=False).all()
        file_list = InternalFileObject.objects.filter(uuid__in=file_uuid_list, is_disabled=False).all()
        id_model_dict, id_file_dict = {}, {}

        for model in model_list:
            id_model_dict[model.id] = model
        
        for file in file_list:
            id_file_dict[file.id] = file

        # replacing ids (foreign keys) with uuids
        final_list = list(timing_list.values())
        for timing in final_list:
            timing['uuid'] = str(timing['uuid'])
            timing['model_uuid'] = str(id_model_dict[timing['model_id']].uuid) if timing['model_id'] else None
            del timing['model_id']

            timing['source_image_uuid'] = str(id_file_dict[timing['source_image_id']].uuid) if timing['source_image_id'] else None
            del timing['source_image_id']

            # TODO: fix this code using interpolated_clip_list
            timing['interpolated_clip_uuid'] = str(id_file_dict[timing['interpolated_clip_id']].uuid) if timing['interpolated_clip_id'] else None
            del timing['interpolated_clip_id']

            timing['timed_clip_uuid'] = str(id_file_dict[timing['timed_clip_id']].uuid) if timing['timed_clip_id'] else None
            del timing['timed_clip_id']

            timing['mask_uuid'] = str(id_file_dict[timing['mask_id']].uuid) if timing['mask_id'] else None
            del timing['mask_id']

            timing['canny_image_uuid'] = str(id_file_dict[timing['canny_image_id']].uuid) if timing['canny_image_id'] else None
            del timing['canny_image_id']

            timing['preview_video_uuid'] = str(id_file_dict[timing['preview_video_id']].uuid) if timing['preview_video_id'] else None
            del timing['preview_video_id']

            timing['primary_image_uuid'] = str(id_file_dict[timing['primary_image_id']].uuid) if timing['primary_image_id'] else None
            del timing['primary_image_id']

            # converting datetime to isoformat
            timing['created_on'] = timing['created_on'].isoformat()
            timing['updated_on'] = timing['updated_on'].isoformat()


        serialized_data = json.dumps(list(final_list))
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
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        backup: BackupTiming = BackupTiming.objects.filter(uuid=backup_uuid, is_disabled=False).first()
        if not backup:
            return InternalResponse({}, 'invalid backup', False)
        
        payload = {
            'data': BackupDto(backup).data
        }

        return InternalResponse(payload, 'backup fetched', True)
    
    def get_backup_list(self, project_uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        project: Project = Project.objects.filter(uuid=project_uuid, is_disabled=False).first()
        if not project:
            return InternalResponse({}, 'invalid project', False)
        
        backup_list = BackupTiming.objects.filter(project_id=project.id, is_disabled=False).all()
        
        payload = {
            'data': BackupListDto(backup_list, many=True).data
        }

        return InternalResponse(payload, 'backup list fetched', True)
    
    def delete_backup(self, backup_uuid):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        backup: BackupTiming = BackupTiming.objects.filter(uuid=backup_uuid, is_disabled=False).first()
        if not backup:
            return InternalResponse({}, 'invalid backup', False)
        
        backup.is_disabled = True
        backup.save()
        
        return InternalResponse({}, 'backup deleted', True)
    
    def restore_backup(self, backup_uuid: str):
        # DBRepo._count += 1
        # cls_name = inspect.currentframe().f_code.co_name
        # print("db call: ", DBRepo._count, " class name: ", cls_name)
        backup: BackupTiming = self.get_backup_from_uuid(backup_uuid)

        current_timing_list: List[Timing] = self.get_timing_list_from_project(backup.project.uuid)
        backup_data = backup.data_dump_dict     # contains a list of dict of backed up timings

        if not backup_data:
            return InternalResponse({}, 'no backup data', False)
        
        for timing in current_timing_list:
            matching_timing_list = [item for item in backup_data if item['uuid'] == str(timing.uuid)]

            if len(matching_timing_list):
                backup_timing = matching_timing_list[0]

                # TODO: fix this code using interpolated_clip_list
                self.update_specific_timing(
                    timing.uuid,
                    model_uuid=backup_timing['model_uuid'],
                    source_image_uuid=backup_timing['source_image_uuid'],
                    interpolated_clip_list=backup_timing['interpolated_clip_list'],
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

    # payment
    def generate_payment_link(self, amount):
        return InternalResponse({'data': 'https://buy.stripe.com/test_8wMbJib8g3HK7vi5ko'}, 'success', True)     # temp link