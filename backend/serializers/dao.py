from argparse import FileType
from rest_framework import serializers

from shared.constants import AIModelCategory, AnimationStyleType, GuidanceType, InternalFileType


class CreateUserDao(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    email = serializers.EmailField(max_length=100)
    password = serializers.CharField(max_length=100)
    type = serializers.CharField(max_length=100, default="user")
    third_party_id = serializers.CharField(max_length=100, default=None, required=False)


class CreateFileDao(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    type = serializers.ChoiceField(choices=InternalFileType.value_list())
    local_path = serializers.CharField(max_length=512, required=False)
    hosted_url = serializers.CharField(max_length=512, required=False)
    tag = serializers.CharField(max_length=100, allow_blank=True, required=False)
    project_id = serializers.CharField(max_length=100, required=False)
    shot_uuid = serializers.CharField(max_length=512, required=False, default="", allow_blank=True)
    inference_log_id = serializers.CharField(max_length=100, allow_null=True, required=False)

    def validate(self, data):
        local_path = data.get("local_path")
        hosted_url = data.get("hosted_url")

        if not local_path and not hosted_url:
            raise serializers.ValidationError("At least one of local_path or hosted_url is required.")

        return data


class CreateProjectDao(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    user_id = serializers.CharField(max_length=100)  # this is user UUID


class CreateAIModelDao(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    user_id = serializers.CharField(max_length=100)  # this is user UUID
    custom_trained = serializers.BooleanField(default=False, required=False)
    version = serializers.CharField(max_length=100, allow_null=True, required=False)
    replicate_url = serializers.CharField(max_length=512, allow_blank=True, default="", required=False)
    diffusers_url = serializers.CharField(max_length=512, allow_blank=True, default="", required=False)
    training_image_list = serializers.CharField(max_length=None, default="", allow_blank=True, required=False)
    category = serializers.ChoiceField(choices=AIModelCategory.value_list())
    keyword = serializers.CharField(max_length=255, default="", allow_blank=True, required=False)
    model_type = serializers.CharField(max_length=None)


class UpdateAIModelDao(serializers.Serializer):
    uuid = serializers.CharField(max_length=100)
    name = serializers.CharField(max_length=100, required=False)
    user_id = serializers.CharField(max_length=100, required=False)  # this is user UUID
    custom_trained = serializers.BooleanField(default=False, required=False)
    version = serializers.CharField(max_length=100, required=False)
    replicate_url = serializers.CharField(max_length=512, default="", required=False)
    diffusers_url = serializers.CharField(max_length=512, default="", required=False)
    training_image_list = serializers.CharField(max_length=None, allow_blank=True, required=False)
    category = serializers.ChoiceField(choices=AIModelCategory.value_list(), required=False)
    keyword = serializers.CharField(max_length=255, required=False)
    model_type = serializers.CharField(max_length=None, required=False)


class CreateInferenceLogDao(serializers.Serializer):
    project_id = serializers.CharField(max_length=100, required=False)
    model_id = serializers.CharField(max_length=100, allow_null=True, required=False)
    input_params = serializers.CharField(required=False)
    output_details = serializers.CharField(required=False)
    total_inference_time = serializers.CharField(required=False)
    credits_used = serializers.CharField(required=False)
    status = serializers.CharField(required=False, default="")
    model_name = serializers.CharField(max_length=512, allow_blank=True, required=False, default="")
    generation_source = serializers.CharField(max_length=512, allow_blank=True, required=False, default="")
    generation_tag = serializers.CharField(max_length=512, allow_blank=True, required=False, default="")


class CreateAIModelParamMapDao(serializers.Serializer):
    model_id = serializers.CharField(max_length=100)
    standard_param_key = serializers.CharField(max_length=100, required=False)
    model_param_key = serializers.CharField(max_length=100, required=False)


class CreateTimingDao(serializers.Serializer):
    model_id = serializers.CharField(max_length=100, required=False)
    source_image_id = serializers.CharField(max_length=100, required=False)
    mask_id = serializers.CharField(max_length=100, required=False)
    canny_image_id = serializers.CharField(max_length=100, required=False)
    shot_id = serializers.CharField(max_length=100)
    primary_image_id = serializers.CharField(max_length=100, required=False)
    alternative_images = serializers.CharField(max_length=100, required=False)
    notes = serializers.CharField(max_length=1024, required=False)
    aux_frame_index = serializers.IntegerField(required=False)


class CreateAppSettingDao(serializers.Serializer):
    user_id = serializers.CharField(max_length=100)
    replicate_key = serializers.CharField(max_length=100, default="", required=False)
    aws_access_key = serializers.CharField(max_length=100, required=False)
    previous_project = serializers.CharField(max_length=100, required=False)
    replicate_username = serializers.CharField(max_length=100, default="", required=False)
    welcome_state = serializers.IntegerField(default=0, required=False)


class UpdateAppSettingDao(serializers.Serializer):
    uuid = serializers.CharField(max_length=100, required=False)  # picking the first app setting by defaults
    user_id = serializers.CharField(max_length=100, required=False)
    replicate_key = serializers.CharField(max_length=100, required=False, allow_blank=True)
    aws_access_key = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    aws_secret_access_key = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    stability_key = serializers.CharField(max_length=100, required=False)
    previous_project = serializers.CharField(max_length=100, required=False)
    replicate_username = serializers.CharField(max_length=100, required=False)
    welcome_state = serializers.IntegerField(required=False)


class CreateSettingDao(serializers.Serializer):
    project_id = serializers.CharField(max_length=255)
    default_model_id = serializers.CharField(max_length=255, required=False)
    audio_id = serializers.CharField(max_length=255, required=False)
    input_type = serializers.CharField(max_length=255, required=False)
    width = serializers.IntegerField(default=512, required=False)
    height = serializers.IntegerField(default=512, required=False)


class UpdateSettingDao(serializers.Serializer):
    project_id = serializers.CharField(max_length=255)
    default_model_id = serializers.CharField(max_length=255, required=False)
    audio_id = serializers.CharField(max_length=255, required=False)
    input_type = serializers.CharField(max_length=255, required=False)
    width = serializers.IntegerField(required=False)
    height = serializers.IntegerField(required=False)
    name = serializers.CharField(max_length=255, required=False)
