from django.utils import timezone
from django.db import models
import uuid
import json
from django.db.models import F
import urllib

from shared.constants import SERVER, ServerType
from shared.file_upload.s3 import generate_s3_url, is_s3_image_url

class BaseModel(models.Model):
    uuid = models.UUIDField(default=uuid.uuid4)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)
    is_disabled = models.BooleanField(default=False)

    class Meta:
        app_label = 'backend'
        abstract = True


class Lock(BaseModel):
    row_key = models.CharField(max_length=255, unique=True)

    class Meta:
        app_label = 'backend'
        db_table = 'lock'


class User(BaseModel):
    name = models.CharField(max_length=255, default="")
    email = models.CharField(max_length=255)
    password = models.TextField(default=None, null=True)
    type = models.CharField(max_length=50, default="user")
    third_party_id = models.CharField(max_length=255, default=None, null=True)
    total_credits = models.FloatField(default=0)

    class Meta:
        app_label = 'backend'
        db_table = 'user'

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
        app_label = 'backend'
        db_table = 'project'


class AIModel(BaseModel):
    name = models.CharField(max_length=255, default="")
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING, null=True)
    custom_trained = models.BooleanField(default=False)
    version = models.CharField(max_length=255, default="", blank=True, null=True)
    replicate_model_id = models.CharField(max_length=255, default="", blank=True)      # for models which were custom created
    replicate_url = models.TextField(default="", blank=True)
    diffusers_url = models.TextField(default="", blank=True)    # for downloading and running models offline
    category = models.CharField(max_length=255,default="", blank=True)     # Lora, Dreambooth..
    model_type = models.TextField(default="", blank=True)   # [txt2img, img2img..] array of types
    training_image_list = models.TextField(default="", blank=True)      # contains an array of uuid of file objects
    keyword = models.CharField(max_length=255,default="", blank=True)

    class Meta:
        app_label = 'backend'
        db_table = 'ai_model'


class InferenceLog(BaseModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)
    model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    input_params = models.TextField(default="", blank=True)
    output_details = models.TextField(default="", blank=True)
    total_inference_time = models.FloatField(default=0)
    status = models.CharField(max_length=255, default="")   # success, failed, in_progress, queued

    class Meta:
        app_label = 'backend'
        db_table = 'inference_log'


class InternalFileObject(BaseModel):
    name = models.TextField(default="")
    type = models.CharField(max_length=255, default="")     # image, video, audio
    local_path = models.TextField(default="")
    hosted_url = models.TextField(default="")
    tag = models.CharField(max_length=255,default="")  # background_image, mask_image, canny_image etc..
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, default=None, null=True)
    inference_log = models.ForeignKey(InferenceLog, on_delete=models.SET_NULL, default=None, null=True)

    class Meta:
        app_label = 'backend'
        db_table = 'file'

    def save(self, *args, **kwargs):
        # if the online url is not an s3 url and it's a production environment then we need to save the file in s3
        if self.hosted_url and not is_s3_image_url(self.hosted_url) and SERVER == ServerType.PRODUCTION.value:
            self.hosted_url = generate_s3_url(self.hosted_url)
        
        if self.hosted_url and not self.local_path and SERVER != ServerType.PRODUCTION.value:
            video = 'temp'
            if self.project:
                video = self.project.uuid

            file_location = "videos/" + str(video) + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".png"
            try:
                urllib.request.urlretrieve(self.hosted_url, file_location)
                self.local_path = file_location
            except Exception as e:
                print(e)
            
        super(InternalFileObject, self).save(*args, **kwargs)

    @property
    def location(self):
        return self.local_path if self.local_path else self.hosted_url


class AIModelParamMap(BaseModel):
    model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    standard_param_key = models.CharField(max_length=255, blank=True)
    model_param_key = models.CharField(max_length=255, blank=True)

    class Meta:
        app_label = 'backend'
        db_table = 'model_param_map'

class BackupTiming(BaseModel):
    name = models.CharField(max_length=255, default="")
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)
    note = models.TextField(default="", blank=True)
    data_dump = models.TextField(default="", blank=True)

    class Meta:
        app_label = 'backend'
        db_table = 'backup_timing'

    @property
    def data_dump_dict(self):
        return json.loads(self.data_dump) if self.data_dump else None

class Timing(BaseModel):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True)
    model = models.ForeignKey(AIModel, on_delete=models.DO_NOTHING, null=True)
    source_image = models.ForeignKey(InternalFileObject, related_name="source_image", on_delete=models.DO_NOTHING, null=True)
    interpolated_clip_list = models.TextField(default=None, null=True)
    timed_clip = models.ForeignKey(InternalFileObject, related_name="timed_clip", on_delete=models.DO_NOTHING, null=True)
    mask = models.ForeignKey(InternalFileObject, related_name="mask", on_delete=models.DO_NOTHING, null=True)
    canny_image = models.ForeignKey(InternalFileObject, related_name="canny_image", on_delete=models.DO_NOTHING, null=True)
    preview_video = models.ForeignKey(InternalFileObject, related_name="preview_video", on_delete=models.DO_NOTHING, null=True)
    primary_image = models.ForeignKey(InternalFileObject, related_name="primary_image", on_delete=models.DO_NOTHING, null=True)   # variant number that is currently selected (among alternative images) NONE if none is present
    custom_model_id_list = models.TextField(default=None, null=True, blank=True)    
    frame_time = models.FloatField(default=None, null=True)
    frame_number = models.IntegerField(default=None, null=True)
    alternative_images = models.TextField(default=None, null=True)
    custom_pipeline = models.CharField(max_length=255, default=None, null=True, blank=True)
    prompt = models.TextField(default='', blank=True)
    negative_prompt = models.TextField(default="", blank=True)
    guidance_scale = models.FloatField(default=7.5)
    seed = models.IntegerField(default=0)
    num_inteference_steps = models.IntegerField(default=50)
    strength = models.FloatField(default=1)
    notes = models.TextField(default="", blank=True)
    adapter_type = models.CharField(max_length=255, default=None, null=True, blank=True)
    clip_duration = models.FloatField(default=None, null=True)     # clip duration of the timed_clip
    animation_style = models.CharField(max_length=255, default=None, null=True)
    interpolation_steps = models.IntegerField(default=0)
    low_threshold = models.FloatField(default=0)
    high_threshold = models.FloatField(default=0)
    aux_frame_index = models.IntegerField(default=0)    # starts with 0 # TODO: udpate this
    transformation_stage = models.CharField(max_length=255, default=None, null=True)

    class Meta:
        app_label = 'backend'
        db_table = 'frame_timing'

    def __init__(self, *args, **kwargs):
        super(Timing, self).__init__(*args, **kwargs)
        self.old_is_disabled = self.is_disabled
        self.old_aux_frame_index = self.aux_frame_index
        self.old_timed_clip = self.timed_clip
        self.old_frame_time = self.frame_time

    def save(self, *args, **kwargs):
        # TODO: updating details of every frame this way can be slow - implement a better strategy

        # ------ handling aux_frame_index ------
        # if the frame is being deleted (disabled)
        if self.old_is_disabled != self.is_disabled and self.is_disabled:
            timing_list = Timing.objects.filter(project_id=self.project_id, \
                                            aux_frame_index__gte=self.aux_frame_index, is_disabled=False).order_by('aux_frame_index')
            
            # shifting aux_frame_index of all frames after this frame one backwards
            if self.is_disabled:
                timing_list.update(aux_frame_index=F('aux_frame_index') - 1)
            else:
                # shifting aux_frame_index of all frames after this frame one forward
                timing_list.update(aux_frame_index=F('aux_frame_index') + 1)

        # if this is a newly created frame or assigned a new aux_frame_index (and not disabled)
        if (not self.id or self.old_aux_frame_index != self.aux_frame_index) and not self.is_disabled:
            if not self.id:
                # shifting aux_frame_index of all frames after this frame one forward
                if Timing.objects.filter(project_id=self.project_id, aux_frame_index=self.aux_frame_index, is_disabled=False).exists():
                    timing_list = Timing.objects.filter(project_id=self.project_id, \
                                            aux_frame_index__gte=self.aux_frame_index, is_disabled=False)
                    timing_list.update(aux_frame_index=F('aux_frame_index') + 1)
            elif self.old_aux_frame_index != self.aux_frame_index:
                if self.aux_frame_index >= self.old_aux_frame_index:
                    timings_to_move = Timing.objects.filter(project_id=self.project_id, aux_frame_index__gt=self.old_aux_frame_index, \
                                    aux_frame_index__lte=self.aux_frame_index, is_disabled=False).order_by('aux_frame_index')
                    frame_time_list = [int(self.frame_time * 100) / 100]
                    for t in timings_to_move:
                        frame_time_list.append(t.frame_time)
                    # updating frame time
                    for idx, t in enumerate(timings_to_move):
                        Timing.objects.filter(uuid=t.uuid, is_disabled=False).update(frame_time=int(frame_time_list[idx] * 100) / 100)
                    self.frame_time = frame_time_list[-1]

                    # moving the frames between old and new index one step backwards
                    timings_to_move.update(aux_frame_index=F('aux_frame_index') - 1)
                else:
                    timings_to_move = Timing.objects.filter(project_id=self.project_id, aux_frame_index__gte=self.aux_frame_index, \
                                       aux_frame_index__lt=self.old_aux_frame_index, is_disabled=False).order_by('aux_frame_index')
                    
                    frame_time_list = [self.frame_time]
                    for t in reversed(timings_to_move):
                        frame_time_list.append(t.frame_time)
                    # updating frame time
                    frame_time_list.reverse()
                    idx = 0
                    self.frame_time = int(frame_time_list[idx] * 100) / 100
                    idx += 1
                    for t in timings_to_move:
                        Timing.objects.filter(uuid=t.uuid, is_disabled=False).update(frame_time=int(frame_time_list[idx] * 100) / 100)
                        idx += 1
                    # moving frames
                    timings_to_move.update(aux_frame_index=F('aux_frame_index') + 1, timed_clip=None, preview_video=None)
                    
                    
                self.interpolated_video_id = None
                self.timed_clip_id = None

                # updating clip_duration
                timing_list = Timing.objects.filter(project_id=self.project_id, is_disabled=False).order_by('aux_frame_index')
                length_of_list = len(timing_list)

                for idx, timing_item in enumerate(timing_list):
                    # last frame
                    if idx == (length_of_list - 1):
                        time_of_frame = timing_item.frame_time
                        duration_of_static_time = 0.0
                        end_duration_of_frame = float(time_of_frame) + float(duration_of_static_time)
                        total_duration_of_frame = float(end_duration_of_frame) - float(time_of_frame)
                    else:
                        time_of_frame = timing_item.frame_time
                        next_timing = timing_list[idx + 1]
                        time_of_next_frame = next_timing.frame_time
                        total_duration_of_frame = float(time_of_next_frame) - float(time_of_frame)

                    Timing.objects.filter(uuid=timing_item.uuid, is_disabled=False).update(clip_duration=total_duration_of_frame)
                

        # ------ handling timed_clip ------
        # if timed_clip is deleted/changed then preview_video will be deleted
        if self.old_timed_clip and (not self.timed_clip or self.old_timed_clip != self.timed_clip):
            self.preview_video = None

        super().save(*args, **kwargs)

    def add_interpolated_clip_list(self, clip_uuid_list):
        cur_list = json.loads(self.interpolated_clip_list) if self.interpolated_clip_list else []
        cur_list.extend(clip_uuid_list)
        cur_list = list(set(cur_list))
        self.interpolated_clip_list = json.dumps(cur_list)

    @property
    def alternative_images_list(self):
        image_id_list = json.loads(self.alternative_images) if self.alternative_images else []
        return InternalFileObject.objects.filter(uuid__in=image_id_list, is_disabled=False).all()
    
    @property
    def primary_variant_location(self):
        if self.primary_image:
            return self.primary_image.location

        return ""
    
    # gives the next entry in the project timings
    @property
    def next_timing(self):
        next_timing = Timing.objects.filter(project=self.project, id__gt=self.id, is_disabled=False).order_by('id').first()
        return next_timing
    
    @property
    def prev_timing(self):
        prev_timing = Timing.objects.filter(project=self.project, id__lt=self.id, is_disabled=False).order_by('id').first()
        return prev_timing


class AppSetting(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    replicate_key = models.CharField(max_length=255, default="", blank=True)
    aws_secret_access_key = models.CharField(max_length=255, default="", blank=True)
    aws_access_key = models.CharField(max_length=255, default="", blank=True)
    stability_key = models.CharField(max_length=255, default="", blank=True)
    previous_project = models.CharField(max_length=255, default="", blank=True)     # contains the uuid of the previous project
    replicate_username = models.CharField(max_length=255, default="", blank=True)
    welcome_state = models.IntegerField(default=0)

    class Meta:
        app_label = 'backend'
        db_table = 'app_setting'

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
    audio = models.ForeignKey(InternalFileObject, related_name="audio", on_delete=models.DO_NOTHING, null=True)
    input_video = models.ForeignKey(InternalFileObject, related_name="input_video", on_delete=models.DO_NOTHING, null=True)
    default_prompt = models.TextField(default="")
    default_strength = models.FloatField(default=0.7)
    default_custom_pipeline = models.CharField(max_length=255, default="", blank=True)
    input_type = models.CharField(max_length=255)   # video, image, audio
    extraction_type = models.CharField(max_length=255)   # Extract manually
    width = models.IntegerField(default=512)
    height = models.IntegerField(default=512)
    default_negative_prompt = models.TextField(default="")
    default_guidance_scale = models.FloatField(default=7.5)
    default_seed = models.IntegerField(default=0)
    default_num_inference_steps = models.IntegerField(default=50)
    default_stage = models.CharField(max_length=255)    # extracted_key_frames
    default_custom_model_uuid_list = models.TextField(default=None, null=True, blank=True)
    default_adapter_type = models.CharField(max_length=255, default="", blank=True)
    guidance_type = models.CharField(max_length=255)   # "Drawing", "Images", "Video"
    default_animation_style = models.CharField(max_length=255)  # "Interpolation", "Direct Morphing"
    default_low_threshold = models.FloatField(default=0)
    default_high_threshold = models.FloatField(default=0)
    zoom_level = models.IntegerField(default=100)
    x_shift = models.IntegerField(default=0)
    y_shift = models.IntegerField(default=0)
    rotation_angle_value = models.FloatField(default=0.0)

    class Meta:
        app_label = 'backend'
        db_table = 'setting'


