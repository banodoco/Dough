import datetime
import json


class InternalFileObject:
    def __init__(self, uuid, name, type, local_path, hosted_url, created_on, tag=""):
        self.uuid = uuid
        self.name = name
        self.type = type
        self.local_path = local_path
        self.hosted_url = hosted_url
        self.tag = tag
        self.created_on = created_on

    @property
    def location(self):
        if self.local_path:
            return self.local_path
        return self.hosted_url


class InternalProjectObject:
    def __init__(self, uuid, name, user_uuid, created_on):
        self.uuid = uuid
        self.name = name
        self.user_uuid = user_uuid
        self.created_on = created_on


class InternalAIModelObject:
    def __init__(self, uuid, name, user_uuid, version, replicate_model_id, replicate_url,
                 diffusers_url, category, training_image_list, keyword, created_on):
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

    # training_image_list contains uuid list of images
    def _get_training_image_list(self, training_image_list):
        if not (training_image_list and len(training_image_list)):
            return []
        
        from utils.data_repo.data_repo import DataRepo
        data_repo = DataRepo()
        file_list = data_repo.get_image_list_from_uuid_list(
            training_image_list)
        return file_list


class InternalFrameTimingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(
            **kwargs["project"]) if 'project' in kwargs and kwargs["project"] else None
        self.model = InternalAIModelObject(
            **kwargs["model"]) if 'model' in kwargs and kwargs["model"] else None
        self.source_image = InternalFileObject(
            **kwargs["source_image"]) if 'source_image' in kwargs and kwargs["source_image"] else None
        self.interpolated_clip = InternalFileObject(
            **kwargs["interpolated_clip"]) if 'interpolated_clip' in kwargs and kwargs["interpolated_clip"] else None
        self.timed_clip = InternalFileObject(
            **kwargs["timed_clip"]) if 'timed_clip' in kwargs and kwargs["timed_clip"] else None
        self.mask = InternalFileObject(
            **kwargs["mask"]) if 'mask' in kwargs and kwargs["mask"] else None
        self.canny_image = InternalFileObject(
            **kwargs["canny_image"]) if 'canny_image' in kwargs and kwargs["canny_image"] else None
        self.preview_video = InternalFileObject(
            **kwargs["preview_video"]) if 'preview_video' in kwargs and kwargs["preview_video"] else None
        self.primary_image = InternalFileObject(
            **kwargs["primary_image"]) if 'primary_image' in kwargs and kwargs["primary_image"] else None
        self.custom_model_id_list = kwargs['custom_model_id_list'] if 'custom_model_id_list' in kwargs and kwargs["custom_model_id_list"] else [
        ]
        self.frame_time = kwargs['frame_time'] if 'frame_time' in kwargs else None
        self.frame_number = kwargs['frame_number'] if 'frame_number' in kwargs else None
        self.alternative_images = kwargs['alternative_images'] if 'alternative_images' in kwargs and kwargs["alternative_images"] else [
        ]
        self.custom_pipeline = kwargs['custom_pipeline'] if 'custom_pipeline' in kwargs and kwargs["custom_pipeline"] else None
        self.prompt = kwargs['prompt'] if 'prompt' in kwargs and kwargs["prompt"] else None
        self.negative_prompt = kwargs['negative_prompt'] if 'negative_prompt' in kwargs and kwargs["negative_prompt"] else None
        self.guidance_scale = kwargs['guidance_scale'] if 'guidance_scale' in kwargs else None
        self.seed = kwargs['seed'] if 'seed' in kwargs else None
        self.num_inteference_steps = kwargs['num_inteference_steps'] if 'num_inteference_steps' in kwargs and kwargs["num_inteference_steps"] else None
        self.strength = kwargs['strength'] if 'strength' in kwargs else None
        self.notes = kwargs['notes'] if 'notes' in kwargs and kwargs["notes"] else None
        self.adapter_type = kwargs['adapter_type'] if 'adapter_type' in kwargs and kwargs["adapter_type"] else None
        self.clip_duration = kwargs['clip_duration'] if 'clip_duration' in kwargs and kwargs["clip_duration"] else None
        self.animation_style = kwargs['animation_style'] if 'animation_style' in kwargs and kwargs["animation_style"] else None
        self.interpolation_steps = kwargs['interpolation_steps'] if 'interpolation_steps' in kwargs and kwargs["interpolation_steps"] else None
        self.low_threshold = kwargs['low_threshold'] if 'low_threshold' in kwargs and kwargs["low_threshold"] else None
        self.high_threshold = kwargs['high_threshold'] if 'high_threshold' in kwargs and kwargs["high_threshold"] else None
        self.aux_frame_index = kwargs['aux_frame_index'] if 'aux_frame_index' in kwargs else None

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
        else:
            return self.primary_image.location

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


class InternalAppSettingObject:
    def __init__(self, **kwargs):
        self.user = InternalUserObject(
            **kwargs["user"]) if 'user' in kwargs else None
        self.previous_project = kwargs['previous_project'] if 'previous_project' in kwargs else None
        self.replicate_username = kwargs['replicate_username'] if 'replicate_username' in kwargs and kwargs['replicate_username'] else ""
        self.welcome_state = kwargs['welcome_state'] if 'welcome_state' in kwargs else None
        self.aws_secret_access_key = kwargs['aws_secret_access_key'] if 'aws_secret_access_key' in kwargs else None
        self.aws_access_key = kwargs['aws_access_key'] if 'aws_access_key' in kwargs else None
        self.replicate_key = kwargs['replicate_key'] if 'replicate_key' in kwargs and kwargs['replicate_key'] else ""


class InternalSettingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(
            **kwargs["project"]) if key_present('project', kwargs) else None
        self.default_model = InternalAIModelObject(
            **kwargs["default_model"]) if key_present('default_model', kwargs) else None
        self.audio = InternalFileObject(
            **kwargs["audio"]) if key_present('audio', kwargs) else None
        self.input_video = InternalFileObject(
            **kwargs["input_video"]) if key_present('input_video', kwargs) else None
        self.default_prompt = kwargs['default_prompt'] if key_present(
            'default_prompt', kwargs) else None
        self.default_strength = kwargs['default_strength'] if key_present(
            'default_strength', kwargs) else None
        self.default_custom_pipeline = kwargs['default_custom_pipeline'] if key_present(
            'default_custom_pipeline', kwargs) else None
        self.input_type = kwargs['input_type'] if key_present(
            'input_type', kwargs) else None
        self.extraction_type = kwargs['extraction_type'] if key_present(
            'extraction_type', kwargs) else None
        self.width = kwargs['width'] if key_present('width', kwargs) else None
        self.height = kwargs['height'] if key_present(
            'height', kwargs) else None
        self.default_negative_prompt = kwargs['default_negative_prompt'] if key_present(
            'default_negative_prompt', kwargs) else None
        self.default_guidance_scale = kwargs['default_guidance_scale'] if key_present(
            'default_guidance_scale', kwargs) else None
        self.default_seed = kwargs['default_seed'] if key_present(
            'default_seed', kwargs) else None
        self.default_num_inference_steps = kwargs['default_num_inference_steps'] if key_present(
            'default_num_inference_steps', kwargs) else None
        self.default_stage = kwargs['default_stage'] if key_present(
            'default_stage', kwargs) else None
        self.default_custom_model_uuid_list = kwargs['default_custom_model_uuid_list'] if key_present(
            'default_custom_model_uuid_list', kwargs) else []
        self.default_adapter_type = kwargs['default_adapter_type'] if key_present(
            'default_adapter_type', kwargs) else None
        self.guidance_type = kwargs['guidance_type'] if key_present(
            'guidance_type', kwargs) else None
        self.default_animation_style = kwargs['default_animation_style'] if key_present(
            'default_animation_style', kwargs) else None
        self.default_low_threshold = kwargs['default_low_threshold'] if key_present(
            'default_low_threshold', kwargs) else None
        self.default_high_threshold = kwargs['default_high_threshold'] if key_present(
            'default_high_threshold', kwargs) else None
        self.zoom_level = kwargs['zoom_level'] if key_present(
            'zoom_level', kwargs) else None


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

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class InferenceLogObject:
    def __init__(self, **kwargs):
        self.project = InternalProjectObject(
            **kwargs["project"]) if 'project' in kwargs else None
        self.model = InternalAIModelObject(
            **kwargs["model"]) if 'model' in kwargs else None
        self.input_params = kwargs['input_params'] if 'input_params' in kwargs else None
        self.output_details = kwargs['output_details'] if 'output_details' in kwargs else None
        self.total_inference_time = kwargs['total_inference_time'] if 'total_inference_time' in kwargs else None


def key_present(key, dict):
    if key in dict and dict[key] is not None:
        return True

    return False
