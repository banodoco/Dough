from django.db import models
import uuid
import json
import requests
from django.db.models import F
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import urllib

from shared.constants import SERVER, InferenceParamType, InferenceStatus, ServerType
from shared.file_upload.s3 import generate_s3_url, is_s3_image_url


class BaseModel(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    is_disabled = models.BooleanField(default=False)

    class Meta:
        app_label = "backend"
        abstract = True


class Lock(BaseModel):
    row_key = models.CharField(max_length=255, unique=True)

    class Meta:
        app_label = "backend"
        db_table = "lock"


class User(BaseModel):
    name = models.CharField(max_length=255, default="")
    email = models.CharField(max_length=255)
    password = models.TextField(default=None, null=True)
    type = models.CharField(max_length=50, default="user")
    third_party_id = models.CharField(max_length=255, default=None, null=True)
    total_credits = models.FloatField(default=0)

    class Meta:
        app_label = "backend"
        db_table = "user"

    def save(self, *args, **kwargs):
        if not self.id:
            self.total_credits = 1000

        super(User, self).save(*args, **kwargs)


class Project(BaseModel):
    name = models.CharField(max_length=255, default="")
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING, null=True)
    temp_file_list = models.TextField(default=None, null=True)
    meta_data = models.TextField(default=None, null=True)

    class Meta:
        app_label = "backend"
        db_table = "project"

    def __init__(self, *args, **kwargs):
        super(Project, self).__init__(*args, **kwargs)
        self.old_project_name = self.name

    def save(self, *args, **kwargs):
        # if either this is a new project or it's name is being updated
        # we check to make sure that it's name is not already present in the app
        if self._state.adding or (self.old_project_name and self.old_project_name != self.name):
            if Project.objects.filter(name=self.name, is_disabled=False).exists():
                raise ValidationError("Project name already exists. Please input unique project name")

        super().save(*args, **kwargs)


class AIModel(BaseModel):
    name = models.CharField(max_length=255, default="")
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING, null=True)
    custom_trained = models.BooleanField(default=False)
    version = models.CharField(max_length=255, default="", blank=True, null=True)
    replicate_model_id = models.CharField(
        max_length=255, default="", blank=True
    )  # for models which were custom created
    replicate_url = models.TextField(default="", blank=True)
    diffusers_url = models.TextField(default="", blank=True)  # for downloading and running models offline
    category = models.CharField(max_length=255, default="", blank=True)  # Lora, Dreambooth..
    model_type = models.TextField(default="", blank=True)  # [txt2img, img2img..] array of types
    training_image_list = models.TextField(
        default="", blank=True
    )  # contains an array of uuid of file objects
    keyword = models.CharField(max_length=255, default="", blank=True)

    class Meta:
        app_label = "backend"
        db_table = "ai_model"


class InferenceLog(BaseModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)
    model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    model_name = models.CharField(max_length=512, default="", blank=True)  # for filtering purposes
    input_params = models.TextField(default="", blank=True)
    output_details = models.TextField(default="", blank=True)
    total_inference_time = models.FloatField(default=0)
    status = models.CharField(max_length=255, default="")  # success, failed, in_progress, queued
    generation_source = models.CharField(max_length=255, default="", blank=True)    # the source of generation
    generation_tag = models.CharField(max_length=255, default="", blank=True)   # review, temp, upscaled etc..

    class Meta:
        app_label = "backend"
        db_table = "inference_log"

    def __init__(self, *args, **kwargs):
        super(InferenceLog, self).__init__(*args, **kwargs)
        self.old_status = self.status

    def save(self, *args, **kwargs):
        # preventing status update if it has been set to one of these values
        if self.old_status in [InferenceStatus.CANCELED.value, InferenceStatus.FAILED.value]:
            self.status = self.old_status

        super().save(*args, **kwargs)


class InternalFileObject(BaseModel):
    name = models.TextField(default="")
    type = models.CharField(max_length=255, default="")  # image, video, audio
    local_path = models.TextField(default="")
    hosted_url = models.TextField(default="")
    tag = models.CharField(max_length=255, default="")  # background_image, mask_image, canny_image etc..
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, default=None, null=True)
    inference_log = models.ForeignKey(InferenceLog, on_delete=models.SET_NULL, default=None, null=True)
    shot_uuid = models.CharField(
        max_length=255, default="", blank=True
    )  # NOTE: this is not a foreignkey and purely for filtering purpose

    class Meta:
        app_label = "backend"
        db_table = "file"

    def save(self, *args, **kwargs):
        # if the online url is not an s3 url and it's a production environment then we need to save the file in s3
        if self.hosted_url and not is_s3_image_url(self.hosted_url) and SERVER == ServerType.PRODUCTION.value:
            self.hosted_url = generate_s3_url(self.hosted_url)

        if self.hosted_url and not self.local_path and SERVER != ServerType.PRODUCTION.value:
            video = "temp"
            if self.project:
                video = self.project.uuid

            file_location = "videos/" + str(video) + "/assets/videos/completed/" + str(uuid.uuid4()) + ".png"
            self.download_and_save_file(file_location)

        super(InternalFileObject, self).save(*args, **kwargs)

        # creating file relation/link if the inference has completed
        if self.inference_log and self.inference_log.status == InferenceStatus.COMPLETED.value:
            parent_entity_data = json.loads(self.inference_log.input_params).get(
                InferenceParamType.FILE_RELATION_DATA.value, None
            )
            if parent_entity_data:
                parent_entity_data = json.loads(parent_entity_data)
                # TODO: optimize such that the entries are not created twice and not
                # created one by one
                for p in parent_entity_data:
                    file_link = FileRelationship()
                    file_link.child_entity_id = self.id
                    file_link.transformation_type = (
                        p["transformation_type"] if "transformation_type" in p else ""
                    )
                    parent_file = InternalFileObject.objects.filter(uuid=p["id"], is_disabled=False).first()
                    if parent_file:
                        file_link.parent_entity_id = parent_file.id
                        file_link.save()

    def get_child_entities(self, transformation_type_list=None):
        query = {"parent_entity_id": self.id, "is_disabled": False}
        if transformation_type_list and len(transformation_type_list):
            query["transformation_type__in"] = transformation_type_list
        entity_list = FileRelationship.objects.filter(**query).all()
        res = []
        for e in entity_list:
            res.append(e.child_entity)
        return res

    def get_parent_entities(self, transformation_type_list=None):
        query = {"parent_entity_id": self.id, "is_disabled": False}
        if transformation_type_list and len(transformation_type_list):
            query["transformation_type__in"] = transformation_type_list
        entity_list = FileRelationship.objects.filter(child_entity_id=self.id, is_disabled=False).all()
        res = []
        for e in entity_list:
            res.append(e.parent_entity)
        return res

    def download_and_save_file(self, file_location):
        try:
            response = requests.get(self.hosted_url)
            response.raise_for_status()

            content = ContentFile(response.content)
            default_storage.save(file_location, content)
            self.local_path = file_location
        except Exception as e:
            print(e)

    @property
    def location(self):
        return self.local_path if self.local_path else self.hosted_url


class FileRelationship(BaseModel):
    """
    for maintaining relationship between files, like upscaled, morphed, stylized, speed change
    this will also help in tracking the series of transformations a file went through
    e.g. if vid_2 is upscaled from vid_1, child will be vid_2 and parent will be vid_1

    for now this relationship is automatically created when a file is created with inference log completed if it has
    the relationship data inside it (given by FILE_RELATION_DATA inside input_params)
    """

    transformation_type = models.TextField(default="")
    child_entity_type = models.CharField(
        max_length=255, default="file"
    )  # rn it will only be used for file relations
    child_entity = models.ForeignKey(
        InternalFileObject, on_delete=models.CASCADE, default=None, null=True, related_name="child_entity"
    )
    parent_entity_type = models.CharField(max_length=255, default="file")
    parent_entity = models.ForeignKey(
        InternalFileObject, on_delete=models.CASCADE, default=None, null=True, related_name="parent_entity"
    )

    class Meta:
        app_label = "backend"
        db_table = "file_relationship"


class AIModelParamMap(BaseModel):
    model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    standard_param_key = models.CharField(max_length=255, blank=True)
    model_param_key = models.CharField(max_length=255, blank=True)

    class Meta:
        app_label = "backend"
        db_table = "model_param_map"


class BackupTiming(BaseModel):
    name = models.CharField(max_length=255, default="")
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)
    note = models.TextField(default="", blank=True)
    data_dump = models.TextField(default="", blank=True)

    class Meta:
        app_label = "backend"
        db_table = "backup_timing"

    @property
    def data_dump_dict(self):
        return json.loads(self.data_dump) if self.data_dump else None


class Shot(BaseModel):
    name = models.CharField(max_length=255, default="", blank=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    main_clip = models.ForeignKey(
        InternalFileObject, default=None, null=True, on_delete=models.DO_NOTHING
    )  # main clip has the correct duration
    desc = models.TextField(default="", blank=True)
    shot_idx = models.IntegerField()
    duration = models.FloatField(default=2.5)
    meta_data = models.TextField(default="", blank=True)
    interpolated_clip_list = models.TextField(default=None, null=True)

    class Meta:
        app_label = "backend"
        db_table = "shot"

    @property
    def meta_data_dict(self):
        return json.loads(self.meta_data) if self.meta_data else None

    def __init__(self, *args, **kwargs):
        super(Shot, self).__init__(*args, **kwargs)
        self.old_shot_idx = self.shot_idx
        self.old_is_disabled = self.is_disabled
        self.old_duration = self.duration

    def add_interpolated_clip_list(self, clip_uuid_list):
        cur_list = json.loads(self.interpolated_clip_list) if self.interpolated_clip_list else []
        cur_list.extend(clip_uuid_list)
        cur_list = list(set(cur_list))
        self.interpolated_clip_list = json.dumps(cur_list)

    def save(self, *args, **kwargs):
        # --------------- handling shot_idx change --------------
        # if the shot is being deleted (disabled)
        if self.old_is_disabled != self.is_disabled:
            shot_list = Shot.objects.filter(project_id=self.project_id, is_disabled=False).all()

            # if this is disabled then shifting every shot backwards one step
            if self.is_disabled:
                shot_list = shot_list.filter(shot_idx__gt=self.shot_idx).order_by("shot_idx")
                shot_list.update(shot_idx=F("shot_idx") - 1)
            else:
                shot_list = shot_list.filter(shot_idx__gte=self.shot_idx).order_by("shot_idx")
                shot_list.update(shot_idx=F("shot_idx") + 1)

        # if this is a newly created shot or assigned new shot_idx (and not disabled)
        if (not self.id or self.old_shot_idx != self.shot_idx) and not self.is_disabled:
            # newly created shot
            if not self.id:
                # if a shot already exists at this place then moving everything one step forward
                if Shot.objects.filter(
                    project_id=self.project_id, shot_idx=self.shot_idx, is_disabled=False
                ).exists():
                    shot_list = Shot.objects.filter(
                        project_id=self.project_id, shot_idx__gte=self.shot_idx, is_disabled=False
                    )
                    shot_list.update(shot_idx=F("shot_idx") + 1)
            elif self.old_shot_idx != self.shot_idx:
                if self.shot_idx >= self.old_shot_idx:
                    shots_to_move = Shot.objects.filter(
                        project_id=self.project_id,
                        shot_idx__gt=self.old_shot_idx,
                        shot_idx__lte=self.shot_idx,
                        is_disabled=False,
                    ).order_by("shot_idx")
                    # moving the frames between old and new index one step backwards
                    shots_to_move.update(shot_idx=F("shot_idx") - 1)
                else:
                    shots_to_move = Shot.objects.filter(
                        project_id=self.project_id,
                        shot_idx__gte=self.shot_idx,
                        shot_idx__lt=self.old_shot_idx,
                        is_disabled=False,
                    ).order_by("shot_idx")
                    shots_to_move.update(shot_idx=F("shot_idx") + 1)

        super(Shot, self).save(*args, **kwargs)


class Timing(BaseModel):
    model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    source_image = models.ForeignKey(
        InternalFileObject, related_name="source_image", on_delete=models.DO_NOTHING, null=True
    )
    mask = models.ForeignKey(InternalFileObject, related_name="mask", on_delete=models.DO_NOTHING, null=True)
    canny_image = models.ForeignKey(
        InternalFileObject, related_name="canny_image", on_delete=models.DO_NOTHING, null=True
    )
    primary_image = models.ForeignKey(
        InternalFileObject, related_name="primary_image", on_delete=models.DO_NOTHING, null=True
    )  # variant number that is currently selected (among alternative images) NONE if none is present
    shot = models.ForeignKey(Shot, on_delete=models.CASCADE, null=True)
    alternative_images = models.TextField(default=None, null=True)
    notes = models.TextField(default="", blank=True)
    clip_duration = models.FloatField(default=None, null=True)
    aux_frame_index = models.IntegerField(default=0)

    class Meta:
        app_label = "backend"
        db_table = "frame_timing"

    def __init__(self, *args, **kwargs):
        super(Timing, self).__init__(*args, **kwargs)
        self.old_is_disabled = self.is_disabled
        self.old_aux_frame_index = self.aux_frame_index
        self.old_shot = self.shot

    def save(self, *args, **kwargs):
        # TODO: updating details of every frame this way can be slow - implement a better strategy

        # ------ handling aux_frame_index ------
        # if the frame is being deleted (disabled)
        if self.old_is_disabled != self.is_disabled and self.is_disabled:
            timing_list = Timing.objects.filter(
                shot_id=self.shot_id, aux_frame_index__gte=self.aux_frame_index, is_disabled=False
            ).order_by("aux_frame_index")

            # shifting aux_frame_index of all frames after this frame one backwards
            if self.is_disabled:
                timing_list.update(aux_frame_index=F("aux_frame_index") - 1)
            else:
                # shifting aux_frame_index of all frames after this frame one forward
                timing_list.update(aux_frame_index=F("aux_frame_index") + 1)

        # if this is a newly created frame or assigned a new aux_frame_index (and not disabled)
        if (not self.id or self.old_aux_frame_index != self.aux_frame_index) and not self.is_disabled:
            if not self.id:
                # shifting aux_frame_index of all frames after this frame one forward
                if Timing.objects.filter(
                    shot_id=self.shot_id, aux_frame_index=self.aux_frame_index, is_disabled=False
                ).exists():
                    timing_list = Timing.objects.filter(
                        shot_id=self.shot_id, aux_frame_index__gte=self.aux_frame_index, is_disabled=False
                    )
                    timing_list.update(aux_frame_index=F("aux_frame_index") + 1)
            elif self.old_aux_frame_index != self.aux_frame_index:
                if self.aux_frame_index >= self.old_aux_frame_index:
                    timings_to_move = Timing.objects.filter(
                        shot_id=self.shot_id,
                        aux_frame_index__gt=self.old_aux_frame_index,
                        aux_frame_index__lte=self.aux_frame_index,
                        is_disabled=False,
                    ).order_by("aux_frame_index")

                    # moving the frames between old and new index one step backwards
                    timings_to_move.update(aux_frame_index=F("aux_frame_index") - 1)
                else:
                    timings_to_move = Timing.objects.filter(
                        shot_id=self.shot_id,
                        aux_frame_index__gte=self.aux_frame_index,
                        aux_frame_index__lt=self.old_aux_frame_index,
                        is_disabled=False,
                    ).order_by("aux_frame_index")
                    timings_to_move.update(aux_frame_index=F("aux_frame_index") + 1)

        # --------------- handling shot change -------------------
        if not self.is_disabled and self.id and self.old_shot != self.shot:
            # moving all frames ahead of this frame, one step backwards
            timing_list = Timing.objects.filter(
                shot_id=self.old_shot.id, aux_frame_index__gt=self.aux_frame_index, is_disabled=False
            ).order_by("aux_frame_index")
            # changing the aux_frame_index of this frame to be the last one in the new shot
            new_index = Timing.objects.filter(shot_id=self.shot.id, is_disabled=False).count()
            self.aux_frame_index = new_index
            timing_list.update(aux_frame_index=F("aux_frame_index") - 1)

        # --------------- adding alternative images ----------
        if not (self.alternative_images and len(self.alternative_images)) and self.primary_image:
            self.alternative_images = json.dumps([str(self.primary_image.uuid)])

        super().save(*args, **kwargs)

    @property
    def alternative_images_list(self):
        image_id_list = json.loads(self.alternative_images) if self.alternative_images else []
        return InternalFileObject.objects.filter(uuid__in=image_id_list, is_disabled=False).all()

    @property
    def primary_variant_location(self):
        if self.primary_image:
            return self.primary_image.location

        return ""

    # gives the next entry in the shot timings
    @property
    def next_timing(self):
        next_timing = (
            Timing.objects.filter(shot=self.shot, id__gt=self.id, is_disabled=False).order_by("id").first()
        )
        return next_timing

    @property
    def prev_timing(self):
        prev_timing = (
            Timing.objects.filter(shot=self.shot, id__lt=self.id, is_disabled=False).order_by("id").first()
        )
        return prev_timing


class AppSetting(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    replicate_key = models.CharField(max_length=255, default="", blank=True)
    aws_secret_access_key = models.CharField(max_length=255, default="", blank=True)
    aws_access_key = models.CharField(max_length=255, default="", blank=True)
    stability_key = models.CharField(max_length=255, default="", blank=True)
    previous_project = models.CharField(
        max_length=255, default="", blank=True
    )  # contains the uuid of the previous project
    replicate_username = models.CharField(max_length=255, default="", blank=True)
    welcome_state = models.IntegerField(default=0)

    class Meta:
        app_label = "backend"
        db_table = "app_setting"

    def __init__(self, *args, **kwargs):
        super(AppSetting, self).__init__(*args, **kwargs)
        self.old_replicate_key = self.replicate_key
        self.old_aws_access_key = self.aws_access_key
        self.old_stability_key = self.stability_key

    def save(self, *args, **kwargs):
        from utils.encryption import Encryptor

        encryptor = Encryptor()

        new_access_key = not self.id or (self.old_aws_access_key != self.aws_access_key)
        new_replicate_key = not self.id or (self.old_replicate_key != self.replicate_key)
        new_stability_key = not self.id or (self.old_stability_key != self.stability_key)

        if new_access_key and self.aws_access_key:
            encrypted_access_key = encryptor.encrypt(self.aws_access_key)
            self.aws_access_key = encrypted_access_key

        if new_replicate_key and self.replicate_key:
            encrypted_replicate_key = encryptor.encrypt(self.replicate_key)
            self.replicate_key = encrypted_replicate_key

        if new_stability_key and self.stability_key:
            encrypted_stability_key = encryptor.encrypt(self.stability_key)
            self.stability_key = encrypted_stability_key

        super(AppSetting, self).save(*args, **kwargs)

    @property
    def aws_access_key_decrypted(self):
        from utils.encryption import Encryptor

        encryptor = Encryptor()
        return encryptor.decrypt(self.aws_access_key) if self.aws_access_key else None

    @property
    def aws_secret_access_key_decrypted(self):
        from utils.encryption import Encryptor

        encryptor = Encryptor()
        return encryptor.decrypt(self.aws_secret_access_key) if self.aws_secret_access_key else None

    @property
    def replicate_key_decrypted(self):
        from utils.encryption import Encryptor

        encryptor = Encryptor()
        return encryptor.decrypt(self.replicate_key) if self.replicate_key else None

    @property
    def stability_key_decrypted(self):
        from utils.encryption import Encryptor

        encryptor = Encryptor()
        return encryptor.decrypt(self.stability_key) if self.stability_key else None


class Setting(BaseModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    default_model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    audio = models.ForeignKey(
        InternalFileObject, related_name="audio", on_delete=models.DO_NOTHING, null=True
    )
    input_type = models.CharField(max_length=255)  # video, image, audio
    width = models.IntegerField(default=512)
    height = models.IntegerField(default=512)

    class Meta:
        app_label = "backend"
        db_table = "setting"
