from rest_framework import serializers

from backend.models import AIModel, AppSetting, BackupTiming, InferenceLog, InternalFileObject, Project, Setting, Timing, User

class InternalFileDto(serializers.ModelSerializer):
    class Meta:
        model = InternalFileObject
        fields = ('uuid', 'name', 'local_path', 'type',  'hosted_url', 'created_on')

class UserDto(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = (
            'uuid',
            'name',
            'email',
            'type',
            'total_credits'
        )

class ProjectDto(serializers.ModelSerializer):
    user_uuid = serializers.SerializerMethodField()
    class Meta:
        model = Project
        fields = ('uuid', 'name', 'user_uuid', 'created_on')

    def get_user_uuid(self, obj):
        return obj.user.uuid
    

class AIModelDto(serializers.ModelSerializer):
    user_uuid = serializers.SerializerMethodField()
    class Meta:
        model = AIModel
        fields = (
            'uuid',
            'name',
            'user_uuid',
            'custom_trained',
            'version',
            'replicate_model_id',
            'replicate_url',
            'diffusers_url',
            'category',
            'training_image_list',
            'keyword',
            'created_on'
        )

    def get_user_uuid(self, obj):
        return obj.user.uuid
    

class TimingDto(serializers.ModelSerializer):
    project = ProjectDto()
    model = AIModelDto()
    source_image = InternalFileDto()
    interpolated_clip = InternalFileDto()
    timed_clip = InternalFileDto()
    mask = InternalFileDto()
    canny_image = InternalFileDto()
    preview_video = InternalFileDto()
    primary_image  = InternalFileDto()
    
    class Meta:
        model = Timing
        fields = (
            "uuid",
            "project",
            "model",
            "source_image",
            "interpolated_clip",
            "timed_clip",
            "mask",
            "canny_image",
            "preview_video",
            "custom_model_id_list",
            "frame_time",
            "frame_number",
            "primary_image",
            "alternative_images",
            "custom_pipeline",
            "prompt",
            "negative_prompt",
            "guidance_scale",
            "seed",
            "num_inteference_steps",
            "strength",
            "notes",
            "adapter_type",
            "clip_duration",
            "animation_style",
            "interpolation_steps",
            "low_threshold",
            "high_threshold",
            "aux_frame_index",
            "created_on",
            "transformation_stage"
        )


class AppSettingDto(serializers.ModelSerializer):
    user = UserDto()

    class Meta:
        model = AppSetting
        fields = (
            "uuid",
            "user",
            "previous_project",
            "replicate_username",
            "welcome_state",
            "created_on"
        )


class SettingDto(serializers.ModelSerializer):
    project = ProjectDto()
    default_model = AIModelDto()
    audio = InternalFileDto()
    input_video = InternalFileDto()
    class Meta:
        model = Setting
        fields = (
            "uuid",
            "project",
            "default_model",
            "audio",
            "input_video",
            "default_prompt",
            "default_strength",
            "default_custom_pipeline",
            "input_type",
            "extraction_type",
            "width",
            "height",
            "default_negative_prompt",
            "default_guidance_scale",
            "default_seed",
            "default_num_inference_steps",
            "default_stage",
            "default_custom_model_uuid_list",
            "default_adapter_type",
            "guidance_type",
            "default_animation_style",
            "default_low_threshold",
            "default_high_threshold",
            "created_on",
            "zoom_level",
            "x_shift",
            "y_shift",
            "rotation_angle_value"
        )


class InferenceLogDto(serializers.ModelSerializer):
    project = ProjectDto()
    model = AIModelDto()

    class Meta:
        model = InferenceLog
        fields = (
            "project", 
            "model", 
            "input_params", 
            "output_details", 
            "total_inference_time",
            "created_on"
        )

class BackupDto(serializers.ModelSerializer):
    project = ProjectDto()
    class Meta:
        model = BackupTiming
        fields = (
            "name",
            "project",
            "note",
            "data_dump",
            "created_on"
        )

class BackupListDto(serializers.ModelSerializer):
    project = ProjectDto()
    class Meta:
        model = BackupTiming
        fields = (
            "uuid",
            "project",
            "name",
            "note",
            "created_on"
        )