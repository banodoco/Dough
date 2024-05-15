import json
from rest_framework import serializers

from backend.models import (
    AIModel,
    AppSetting,
    BackupTiming,
    InferenceLog,
    InternalFileObject,
    Project,
    Setting,
    Shot,
    Timing,
    User,
)


class UserDto(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("uuid", "name", "email", "type", "total_credits")


class ProjectDto(serializers.ModelSerializer):
    user_uuid = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = ("uuid", "name", "user_uuid", "created_on", "temp_file_list", "meta_data")

    def get_user_uuid(self, obj):
        return obj.user.uuid


class AIModelDto(serializers.ModelSerializer):
    user_uuid = serializers.SerializerMethodField()

    class Meta:
        model = AIModel
        fields = (
            "uuid",
            "name",
            "user_uuid",
            "custom_trained",
            "version",
            "replicate_model_id",
            "replicate_url",
            "diffusers_url",
            "category",
            "training_image_list",
            "keyword",
            "created_on",
        )

    def get_user_uuid(self, obj):
        return obj.user.uuid


class InferenceLogDto(serializers.ModelSerializer):
    project = ProjectDto()
    model = AIModelDto()

    class Meta:
        model = InferenceLog
        fields = (
            "uuid",
            "project",
            "model",
            "input_params",
            "output_details",
            "total_inference_time",
            "created_on",
            "updated_on",
            "status",
            "model_name",
        )


class InternalFileDto(serializers.ModelSerializer):
    project = ProjectDto()  # TODO: pass this as context to speed up the api
    inference_log = InferenceLogDto()

    class Meta:
        model = InternalFileObject
        fields = (
            "uuid",
            "name",
            "local_path",
            "type",
            "hosted_url",
            "created_on",
            "inference_log",
            "project",
            "tag",
            "shot_uuid",
        )


class BasicShotDto(serializers.ModelSerializer):
    project = ProjectDto()

    class Meta:
        model = Shot
        fields = (
            "uuid",
            "name",
            "project",
            "desc",
            "shot_idx",
            "project",
            "duration",
            "meta_data",
        )


class TimingDto(serializers.ModelSerializer):
    model = AIModelDto()
    source_image = InternalFileDto()
    mask = InternalFileDto()
    canny_image = InternalFileDto()
    primary_image = InternalFileDto()
    shot = BasicShotDto()

    class Meta:
        model = Timing
        fields = (
            "uuid",
            "model",
            "source_image",
            "mask",
            "canny_image",
            "primary_image",
            "alternative_images",
            "notes",
            "aux_frame_index",
            "created_on",
            "shot",
        )


class AppSettingDto(serializers.ModelSerializer):
    user = UserDto()

    class Meta:
        model = AppSetting
        fields = ("uuid", "user", "previous_project", "replicate_username", "welcome_state", "created_on")


class SettingDto(serializers.ModelSerializer):
    project = ProjectDto()
    default_model = AIModelDto()
    audio = InternalFileDto()

    class Meta:
        model = Setting
        fields = (
            "uuid",
            "project",
            "default_model",
            "audio",
            "input_type",
            "width",
            "height",
            "created_on",
        )


class BackupDto(serializers.ModelSerializer):
    project = ProjectDto()

    class Meta:
        model = BackupTiming
        fields = ("name", "project", "note", "data_dump", "created_on")


class BackupListDto(serializers.ModelSerializer):
    project = ProjectDto()

    class Meta:
        model = BackupTiming
        fields = ("uuid", "project", "name", "note", "created_on")


class ShotDto(serializers.ModelSerializer):
    timing_list = serializers.SerializerMethodField()
    interpolated_clip_list = serializers.SerializerMethodField()
    main_clip = InternalFileDto()
    project = ProjectDto()

    class Meta:
        model = Shot
        fields = (
            "uuid",
            "name",
            "desc",
            "shot_idx",
            "project",
            "duration",
            "meta_data",
            "timing_list",
            "interpolated_clip_list",
            "main_clip",
        )

    def get_timing_list(self, obj):
        timing_list = self.context.get("timing_list", [])
        timing_list = [
            TimingDto(timing).data for timing in timing_list if str(timing.shot.uuid) == str(obj.uuid)
        ]
        timing_list.sort(key=lambda x: x["aux_frame_index"])
        return timing_list

    def get_interpolated_clip_list(self, obj):
        id_list = json.loads(obj.interpolated_clip_list) if obj.interpolated_clip_list else []
        file_list = InternalFileObject.objects.filter(uuid__in=id_list, is_disabled=False).all()
        return [InternalFileDto(file).data for file in file_list]
