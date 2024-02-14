import datetime
import json
import os
from urllib.parse import urlparse
from shared.constants import InferenceParamType, ProjectMetaData

from ui_components.constants import DefaultProjectSettingParams, DefaultTimingStyleParams
from utils.common_decorators import session_state_attributes
from utils.constants import MLQueryObject


class InternalFileObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if key_present('uuid', kwargs) else None
        self.name = kwargs['name'] if key_present('name', kwargs) else None
        self.type = kwargs['type'] if key_present('type', kwargs) else None
        self.local_path = kwargs['local_path'] if key_present('local_path', kwargs) else None
        self.hosted_url = kwargs['hosted_url'] if key_present('hosted_url', kwargs) else None
        self.tag = kwargs['tag'] if key_present('tag', kwargs) else None
        self.created_on = kwargs['created_on'] if key_present('created_on', kwargs) else None
        self.inference_log = InferenceLogObject(**kwargs['inference_log']) if key_present('inference_log', kwargs) else None
        self.project = InternalProjectObject(**kwargs['project']) if key_present('project', kwargs) else None

    @property
    def location(self):
        if self.local_path:
            return self.local_path
        return self.hosted_url
    
    @property
    def inference_params(self):
        log = self.inference_log
        if not log:
            from utils.data_repo.data_repo import DataRepo

            data_repo = DataRepo()
            fresh_obj = data_repo.get_file_from_uuid(self.uuid)
            log = fresh_obj.inference_log
        
        if log and log.input_params:
            return json.loads(log.input_params)
        
        return None
    
    @property
    def filename(self):
        input_path = self.location
        if urlparse(input_path).scheme:
            parts = urlparse(input_path)
            filename = os.path.basename(parts.path)
        else:
            filename = os.path.basename(input_path)

        return filename


class InternalProjectObject:
    def __init__(self, uuid, name, user_uuid, created_on, temp_file_list, meta_data=None):
        self.uuid = uuid
        self.name = name
        self.user_uuid = user_uuid
        self.created_on = created_on
        self.temp_file_list = temp_file_list
        self.meta_data = meta_data

    @property
    def project_temp_file_list(self):
        return json.loads(self.temp_file_list) if self.temp_file_list else {}
    
    def get_temp_mask_file(self, key):
        temp_files_list = self.project_temp_file_list
        for k, v in temp_files_list.items():
            if k == key:
                from utils.data_repo.data_repo import DataRepo
                data_repo = DataRepo()
                file = data_repo.get_file_from_uuid(v)
                return file
            
        return None
    
    def get_background_image_list(self):
        image_list = json.loads(self.meta_data).get(ProjectMetaData.BACKGROUND_IMG_LIST.value, [])
        if image_list and len(image_list):
            from utils.data_repo.data_repo import DataRepo
            data_repo = DataRepo()
            image_list = data_repo.get_image_list_from_uuid_list(image_list)
            return image_list
        
        return []


class InternalAIModelObject:
    def __init__(self, uuid, name, user_uuid, version, replicate_model_id, replicate_url,
                 diffusers_url, category, custom_trained, training_image_list, keyword, created_on):
        self.uuid = uuid
        self.name = name
        self.user_uuid = user_uuid
        self.version = version
        self.replicate_model_id = replicate_model_id
        self.replicate_url = replicate_url
        self.diffusers_url = diffusers_url
        self.category = category
        self.training_image_list = self._get_training_image_list(
            training_image_list)
        self.keyword = keyword
        self.created_on = created_on
        self.custom_trained = custom_trained

    # training_image_list contains uuid list of images
    def _get_training_image_list(self, training_image_list):
        if not (training_image_list and len(training_image_list)):
            return []
        
        from utils.data_repo.data_repo import DataRepo
        data_repo = DataRepo()
        training_image_list = json.loads(training_image_list)
        file_list = data_repo.get_image_list_from_uuid_list(
            training_image_list)
        return file_list

@session_state_attributes(DefaultTimingStyleParams)
class InternalFrameTimingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.source_image = InternalFileObject(**kwargs["source_image"]) if key_present('source_image', kwargs) else None
        self.shot = InternalShotObject(**kwargs['shot']) if key_present('shot', kwargs) else None
        self.mask = InternalFileObject(**kwargs["mask"]) if key_present('mask', kwargs) else None
        self.canny_image = InternalFileObject( **kwargs["canny_image"]) if key_present('canny_image', kwargs) else None
        self.primary_image = InternalFileObject(**kwargs["primary_image"]) if key_present('primary_image', kwargs) else None
        self.alternative_images = kwargs['alternative_images'] if key_present('alternative_images', kwargs) else []
        self.notes = kwargs['notes'] if 'notes' in kwargs and kwargs["notes"] else ""
        self.clip_duration = kwargs['clip_duration'] if key_present('clip_duration', kwargs) else 0
        self.aux_frame_index = kwargs['aux_frame_index'] if 'aux_frame_index' in kwargs else 0

    @property
    def alternative_images_list(self):
        if not (self.alternative_images and len(self.alternative_images)):
            return []
        
        from utils.data_repo.data_repo import DataRepo
        
        data_repo = DataRepo()
        image_id_list = json.loads(
            self.alternative_images) if self.alternative_images else []
        image_list = data_repo.get_image_list_from_uuid_list(image_id_list)
        return image_list

    @property
    def primary_image_location(self):
        if not len(self.alternative_images_list):
            return ""
        elif self.primary_image:
            return self.primary_image.location
        else:
            return ""

    @property
    def primary_variant_index(self):
        if not self.primary_image:
            return -1

        alternative_images_list = self.alternative_images_list
        idx = 0
        for img in alternative_images_list:
            if img.uuid == self.primary_image.uuid:
                return idx
            idx += 1

        return -1


class InternalShotObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if key_present('uuid', kwargs) else None
        self.name = kwargs['name'] if key_present('name', kwargs) else ""
        self.project = InternalProjectObject(**kwargs['project']) if key_present('project', kwargs) else None
        self.desc = kwargs['desc'] if key_present('desc', kwargs) else ""
        self.shot_idx = kwargs['shot_idx'] if key_present('shot_idx', kwargs) else 0
        self.duration = kwargs['duration'] if key_present('duration', kwargs) else 0
        self.meta_data = kwargs['meta_data'] if key_present('meta_data', kwargs) else {}
        self.timing_list = [InternalFrameTimingObject(**timing) for timing in kwargs["timing_list"]] \
            if key_present('timing_list', kwargs) and kwargs["timing_list"] else []
        self.interpolated_clip_list = [InternalFileObject(**vid) for vid in kwargs['interpolated_clip_list']] if key_present('interpolated_clip_list', kwargs) \
                    else []
        self.main_clip = InternalFileObject(**kwargs['main_clip']) if key_present('main_clip', kwargs) else \
                    None

    @property
    def meta_data_dict(self):
        return json.loads(self.meta_data) if self.meta_data else {}
    
    @property
    def primary_interpolated_video_index(self):
        video_list = self.interpolated_clip_list
        if not len(video_list):
            return -1
        
        if self.main_clip:
            for idx, vid in enumerate(video_list):
                if vid.uuid == self.main_clip.uuid:
                    return idx
        
        return -1


class InternalAppSettingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.user = InternalUserObject(
            **kwargs["user"]) if 'user' in kwargs else None
        self.previous_project = InternalProjectObject(
            **kwargs["project"]) if key_present('project', kwargs) else None
        self.replicate_username = kwargs['replicate_username'] if 'replicate_username' in kwargs and kwargs['replicate_username'] else ""
        self.welcome_state = kwargs['welcome_state'] if 'welcome_state' in kwargs else None
        self.aws_secret_access_key = kwargs['aws_secret_access_key'] if 'aws_secret_access_key' in kwargs else None
        self.aws_access_key = kwargs['aws_access_key'] if 'aws_access_key' in kwargs else None
        self.replicate_key = kwargs['replicate_key'] if 'replicate_key' in kwargs and kwargs['replicate_key'] else ""


@session_state_attributes(DefaultProjectSettingParams)
class InternalSettingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(
            **kwargs["project"]) if key_present('project', kwargs) else None
        self.default_model = InternalAIModelObject(
            **kwargs["default_model"]) if key_present('default_model', kwargs) else None
        self.audio = InternalFileObject(
            **kwargs["audio"]) if key_present('audio', kwargs) else None
        self.input_type = kwargs['input_type'] if key_present(
            'input_type', kwargs) else None
        self.width = kwargs['width'] if key_present('width', kwargs) else None
        self.height = kwargs['height'] if key_present(
            'height', kwargs) else None


class InternalBackupObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(
            **kwargs["project"]) if 'project' in kwargs else None
        self.name = kwargs['name'] if 'name' in kwargs else None
        self.data_dump = kwargs['data_dump'] if 'data_dump' in kwargs else None
        self.note = kwargs['note'] if 'note' in kwargs else None
        self.created_on = datetime.datetime.fromisoformat(kwargs['created_on']) if 'created_on' in kwargs else None

    @property
    def data_dump_dict(self):
        return json.loads(self.data_dump) if self.data_dump else None


class InternalUserObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.name = kwargs['name'] if 'name' in kwargs else None
        self.email = kwargs['email'] if 'email' in kwargs else None
        self.type = kwargs['type'] if 'type' in kwargs else None
        self.total_credits = kwargs['total_credits'] if 'total_credits' in kwargs else 0

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

# input_params = {**input_params, "query_dict": {}, "origin_data": {}, "replicate_inference": {}}
class InferenceLogObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if key_present('uuid', kwargs) else None
        self.project = InternalProjectObject(
            **kwargs["project"]) if key_present('project', kwargs) else None
        self.model = InternalAIModelObject(
            **kwargs["model"]) if key_present('model', kwargs) else None
        self.input_params = kwargs['input_params'] if key_present('input_params', kwargs) else None
        self.output_details = kwargs['output_details'] if key_present('output_details', kwargs) else None
        self.total_inference_time = kwargs['total_inference_time'] if key_present('total_inference_time', kwargs) else None
        self.status = kwargs['status'] if key_present('status', kwargs) else None
        self.updated_on = datetime.datetime.fromisoformat(kwargs['updated_on'][:26]) if key_present('updated_on', kwargs) else None


def key_present(key, dict):
    if key in dict and dict[key] is not None:
        return True

    return False
