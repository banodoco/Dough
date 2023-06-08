import json


class InternalFileObject:
    def __init__(self, uuid, name, type, local_path, hosted_url, created_on, tag=""):
        self.uuid = uuid
        self.name = name
        self.type = type
        self.local_path = local_path
        self.hosted_url =  hosted_url
        self.tag = tag
        self.created_on = created_on
    
    @property
    def location(self):
        return self.local_path if self.local_path else self.hosted_url



class InternalProjectObject:
    def __init__(self, uuid, name, user_uuid):
        self.uuid = uuid
        self.name = name
        self.user_uuid = user_uuid


class InternalAIModelObject:
    def __init__(self, uuid, name, user_uuid, version, replicate_model_id, \
                 diffusers_url, category, training_image_list):
        self.uuid = uuid
        self.name = name
        self.user_uuid = user_uuid
        self.version = version
        self.replicate_model_id = replicate_model_id
        self.replicate_url = replicate_model_id
        self.diffusers_url = diffusers_url
        self.category = category
        self.training_image_list = training_image_list  # contains uuid list of images


class InternalFrameTimingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(**kwargs["project"]) if 'project' in kwargs else None
        self.model = InternalAIModelObject(**kwargs["model"]) if 'model' in kwargs else None
        self.source_image = InternalFileObject(**kwargs["source_image"]) if 'source_image' in kwargs else None
        self.interpolated_clip = InternalFileObject(**kwargs["interpolated_clip"]) if 'interpolated_clip' in kwargs else None
        self.timed_clip = InternalFileObject(**kwargs["timed_clip"]) if 'timed_clip' in kwargs else None
        self.mask = InternalFileObject(**kwargs["mask"]) if 'mask' in kwargs else None
        self.canny_image = InternalFileObject(**kwargs["canny_image"]) if 'canny_image' in kwargs else None
        self.preview_video = InternalFileObject(**kwargs["preview_video"]) if 'preview_video' in kwargs else None
        self.primary_image = InternalFileObject(**kwargs["primary_image"]) if 'primary_image' in kwargs else None
        self.custom_model_uuid_list = kwargs['custom_model_uuid_list'] if 'custom_model_uuid_list' in kwargs else []
        self.frame_time = kwargs['frame_time'] if 'frame_time' in kwargs else None
        self.frame_number = kwargs['frame_number'] if 'frame_number' in kwargs else None
        self.alternative_images = kwargs['alternative_images'] if 'alternative_images' in kwargs else []
        self.custom_pipeline = kwargs['custom_pipeline'] if 'custom_pipeline' in kwargs else None
        self.prompt = kwargs['prompt'] if 'prompt' in kwargs else None
        self.negative_prompt = kwargs['negative_prompt'] if 'negative_prompt' in kwargs else None
        self.guidance_scale = kwargs['guidance_scale'] if 'guidance_scale' in kwargs else None
        self.seed = kwargs['seed'] if 'seed' in kwargs else None
        self.num_inteference_steps = kwargs['num_inteference_steps'] if 'num_inteference_steps' in kwargs else None
        self.strength = kwargs['strength'] if 'strength' in kwargs else None
        self.notes = kwargs['notes'] if 'notes' in kwargs else None
        self.adapter_type = kwargs['adapter_type'] if 'adapter_type' in kwargs else None
        self.clip_duration = kwargs['clip_duration'] if 'clip_duration' in kwargs else None
        self.animation_style = kwargs['animation_style'] if 'animation_style' in kwargs else None
        self.interpolation_steps = kwargs['interpolation_steps'] if 'interpolation_steps' in kwargs else None
        self.low_threshold = kwargs['low_threshold'] if 'low_threshold' in kwargs else None
        self.high_threshold = kwargs['high_threshold'] if 'high_threshold' in kwargs else None
        self.aux_frame_index = kwargs['aux_frame_index'] if 'aux_frame_index' in kwargs else None

    @property
    def alternative_images_list(self):
        image_id_list = json.loads(self.alternative_images) if self.alternative_images else []
        return InternalFileObject.objects.filter(id__in=image_id_list, is_disabled=False).all()
    
    @property
    def primary_variant_location(self):
        if not len(self.alternative_images_list):
            return ""
        else:                         
            return self.alternative_images_list[self.primary_image].location if self.primary_image < len(self.alternative_images_list) else ""
        
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
        self.user = InternalUserObject(**kwargs["user"]) if 'user' in kwargs else None
        self.previous_project = kwargs['previous_project'] if 'previous_project' in kwargs else None
        self.replicate_user_name = kwargs['replicate_user_name'] if 'replicate_user_name' in kwargs else None
        self.welcome_state = kwargs['welcome_state'] if 'welcome_state' in kwargs else None
        self.aws_secret_access_key = kwargs['aws_secret_access_key'] if 'aws_secret_access_key' in kwargs else None
        self.aws_access_key = kwargs['aws_access_key'] if 'aws_access_key' in kwargs else None

class InternalSettingObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(**kwargs["project"]) if 'project' in kwargs else None
        self.default_model = InternalAIModelObject(**kwargs["default_model"]) if 'default_model' in kwargs else None
        self.audio = InternalFileObject(**kwargs["audio"]) if 'audio' in kwargs else None
        self.input_video = InternalFileObject(**kwargs["input_video"]) if 'input_video' in kwargs else None
        self.default_prompt = kwargs['default_prompt'] if 'default_prompt' in kwargs else None
        self.default_strength = kwargs['default_strength'] if 'default_strength' in kwargs else None
        self.default_custom_pipeline = kwargs['default_custom_pipeline'] if 'default_custom_pipeline' in kwargs else None
        self.input_type = kwargs['input_type'] if 'input_type' in kwargs else None
        self.extraction_type = kwargs['extraction_type'] if 'extraction_type' in kwargs else None
        self.width = kwargs['width'] if 'width' in kwargs else None
        self.height = kwargs['height'] if 'height' in kwargs else None
        self.default_negative_prompt = kwargs['default_negative_prompt'] if 'default_negative_prompt' in kwargs else None
        self.default_guidance_scale = kwargs['default_guidance_scale'] if 'default_guidance_scale' in kwargs else None
        self.default_seed = kwargs['default_seed'] if 'default_seed' in kwargs else None
        self.default_num_inference_steps = kwargs['default_num_inference_steps'] if 'default_num_inference_steps' in kwargs else None
        self.default_stage = kwargs['default_stage'] if 'default_stage' in kwargs else None
        self.default_custom_model_uuid_list = kwargs['default_custom_model_uuid_list'] if 'default_custom_model_uuid_list' in kwargs else []
        self.default_adapter_type = kwargs['default_adapter_type'] if 'default_adapter_type' in kwargs else None
        self.guidance_type = kwargs['guidance_type'] if 'guidance_type' in kwargs else None
        self.default_animation_style = kwargs['default_animation_style'] if 'default_animation_style' in kwargs else None
        self.default_low_threshold = kwargs['default_low_threshold'] if 'default_low_threshold' in kwargs else None
        self.default_high_threshold = kwargs['default_high_threshold'] if 'default_high_threshold' in kwargs else None

class InternalBackupObject:
    def __init__(self, **kwargs):
        self.uuid = kwargs['uuid'] if 'uuid' in kwargs else None
        self.project = InternalProjectObject(**kwargs["project"]) if 'project' in kwargs else None
        self.name = kwargs['name'] if 'name' in kwargs else None
        self.data_dump = kwargs['data_dump'] if 'data_dump' in kwargs else None
        self.note = kwargs['note'] if 'note' in kwargs else None

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
        self.project = InternalProjectObject(**kwargs["project"]) if 'project' in kwargs else None
        self.model = InternalAIModelObject(**kwargs["model"]) if 'model' in kwargs else None
        self.input_params = kwargs['input_params'] if 'input_params' in kwargs else None
        self.output_details = kwargs['output_details'] if 'output_details' in kwargs else None
        self.total_inference_time = kwargs['total_inference_time'] if 'total_inference_time' in kwargs else None