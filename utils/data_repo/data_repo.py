# this repo serves as a middlerware between API backend and the frontend
import json
import time
from shared.constants import (
    SECRET_ACCESS_TOKEN,
    InferenceParamType,
    InferenceStatus,
    InternalFileType,
    InternalResponse,
)
from shared.constants import SERVER, ServerType
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.models import (
    InferenceLogObject,
    InternalAIModelObject,
    InternalAppSettingObject,
    InternalBackupObject,
    InternalFrameTimingObject,
    InternalProjectObject,
    InternalFileObject,
    InternalSettingObject,
    InternalShotObject,
    InternalUserObject,
)
from utils.cache.cache_methods import cache_data

from utils.data_repo.api_repo import APIRepo


@cache_data
class DataRepo:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        if not self._initialized:
            if SERVER == ServerType.DEVELOPMENT.value:
                from backend.db_repo import DBRepo

                self.db_repo = DBRepo()
            else:
                self.db_repo = APIRepo()

            self._initialized = True

    def refresh_auth_token(self, refresh_token):
        data = self.db_repo.refresh_auth_token(refresh_token).data["data"]
        user = InternalUserObject(**data["user"]) if data and data["user"] else None
        token = data["token"] if data and data["token"] else None
        refresh_token = data["refresh_token"] if data and data["refresh_token"] else None
        return user, token, refresh_token

    def user_password_login(self, **kwargs):
        data = self.db_repo.user_password_login(**kwargs).data["data"]
        user = InternalUserObject(**data["user"]) if data and data["user"] else None
        token = data["token"] if data and data["token"] else None
        refresh_token = data["refresh_token"] if data and data["refresh_token"] else None
        return user, token, refresh_token

    def google_user_login(self, **kwargs):
        data = self.db_repo.google_user_login(**kwargs).data["data"]
        user = InternalUserObject(**data["user"]) if data and data["user"] else None
        token = data["token"] if data and data["token"] else None
        refresh_token = data["refresh_token"] if data and data["refresh_token"] else None
        return user, token, refresh_token

    def create_user(self, **kwargs):
        user = self.db_repo.create_user(**kwargs).data["data"]
        return InternalUserObject(**user) if user else None

    def get_first_active_user(self, invalidate_cache=False):
        res: InternalResponse = self.db_repo.get_first_active_user()
        user = res.data["data"] if res.status else None
        return InternalUserObject(**user) if user else None

    def get_user_by_email(self, email):
        user = self.db_repo.get_user_by_email(email).data["data"]
        return InternalUserObject(**user) if user else None

    def get_total_user_count(self):
        return self.db_repo.get_total_user_count().data

    def get_all_user_list(self):
        user_list = self.db_repo.get_all_user_list().data["data"]
        return [InternalUserObject(**user) for user in user_list] if user_list else None

    def update_user(self, user_id, **kwargs):
        res = self.db_repo.update_user(user_id, **kwargs)
        user = res.data["data"] if res.status else None
        return InternalUserObject(**user) if user else None

    def delete_user_by_email(self, email):
        res = self.db_repo.delete_user_by_email(email)
        return res.status

    # internal file object
    def get_file_from_name(self, name):
        file = self.db_repo.get_file_from_name(name).data["data"]
        return InternalFileObject(**file) if file else None

    def get_file_from_uuid(self, uuid):
        res = self.db_repo.get_file_from_uuid(uuid)
        file = res.data["data"] if res.status else None
        return InternalFileObject(**file) if file else None

    def get_file_list_from_log_uuid_list(self, log_uuid_list):
        res = self.db_repo.get_file_list_from_log_uuid_list(log_uuid_list)
        file_list = res.data["data"] if res.status else []
        return [InternalFileObject(**file) for file in file_list]

    # kwargs -  file_type: InternalFileType, tag = None, shot_uuid = "", project_id = None, page=None, data_per_page=None, sort_order=None
    def get_all_file_list(self, **kwargs):
        kwargs["type"] = kwargs["file_type"]
        del kwargs["file_type"]

        res = self.db_repo.get_all_file_list(**kwargs)
        file_list = res.data["data"] if res.status else None

        return ([InternalFileObject(**file) for file in file_list] if file_list else [], res.data)

    def create_or_update_file(self, uuid, type=InternalFileType.IMAGE.value, **kwargs):
        file = self.db_repo.create_or_update_file(uuid, type, **kwargs).data["data"]
        return InternalFileObject(**file) if file else None

    def upload_file(self, file_content, ext):
        res = self.db_repo.upload_file(file_content, ext)
        file_url = res.data["data"] if res.status else None
        return file_url

    def create_file(self, **kwargs) -> InternalFileObject:
        if "hosted_url" not in kwargs and SERVER != ServerType.DEVELOPMENT.value:
            file_content = ("file", open(kwargs["local_path"], "rb"))
            uploaded_file_url = self.upload_file(file_content)
            kwargs.update({"hosted_url": uploaded_file_url})

        # handling the case of local inference.. will fix later
        if "hosted_url" in kwargs and not kwargs["hosted_url"].startswith("http"):
            kwargs["local_path"] = kwargs["hosted_url"]
            del kwargs["hosted_url"]

        res = self.db_repo.create_file(**kwargs)
        file = res.data["data"] if res.status else None
        file = InternalFileObject(**file) if file else None

        if file and file.type == InternalFileType.IMAGE.value:
            from ui_components.methods.file_methods import normalize_size_internal_file_obj

            file = normalize_size_internal_file_obj(file, **kwargs)

        return file

    def delete_file_from_uuid(self, uuid):
        res = self.db_repo.delete_file_from_uuid(uuid)
        return res.status

    def get_image_list_from_uuid_list(self, image_uuid_list, file_type=InternalFileType.IMAGE.value):
        if not (image_uuid_list and len(image_uuid_list)):
            return []
        image_list = self.db_repo.get_image_list_from_uuid_list(image_uuid_list, file_type=file_type).data[
            "data"
        ]
        return [InternalFileObject(**image) for image in image_list] if image_list else []

    def update_file(self, file_uuid, **kwargs):
        # TODO: we are updating hosted_url whenever local_path is updated but we
        # are not checking if the local_path is a different one - handle this correctly
        if "local_path" in kwargs and SERVER != ServerType.DEVELOPMENT.value:
            file_content = ("file", open(kwargs["local_path"], "rb"))
            uploaded_file_url = self.upload_file(file_content)
            kwargs.update({"hosted_url": uploaded_file_url})

        res = self.db_repo.update_file(uuid=file_uuid, **kwargs)
        file = res.data["data"] if res.status else None
        return InternalFileObject(**file) if file else None

    def get_file_count_from_type(self, file_tag=None, project_uuid=None):
        return self.db_repo.get_file_count_from_type(file_tag, project_uuid).data["data"]

    def update_temp_gallery_images(self, project_uuid):
        self.db_repo.update_temp_gallery_images(project_uuid)
        return True

    # project
    def get_project_from_uuid(self, uuid):
        project = self.db_repo.get_project_from_uuid(uuid).data["data"]
        return InternalProjectObject(**project) if project else None

    def get_all_project_list(self, user_id):
        project_list = self.db_repo.get_all_project_list(user_id).data["data"]
        project_list.sort(key=lambda x: x["created_on"])
        return [InternalProjectObject(**project) for project in project_list] if project_list else None

    def create_project(self, **kwargs):
        project = self.db_repo.create_project(**kwargs).data["data"]
        return InternalProjectObject(**project) if project else None

    def delete_project_from_uuid(self, uuid):
        res = self.db_repo.delete_project_from_uuid(uuid)
        return res.status

    def update_project(self, **kwargs):
        project = self.db_repo.update_project(**kwargs).data["data"]
        return InternalProjectObject(**project) if project else None

    # ai model (custom ai model)
    def get_ai_model_from_uuid(self, uuid):
        res = self.db_repo.get_ai_model_from_uuid(uuid)
        model = res.data["data"] if res.status else None
        return InternalAIModelObject(**model) if model else None

    def get_ai_model_from_name(self, name, user_id):
        res = self.db_repo.get_ai_model_from_name(name, user_id)
        model = res.data["data"] if res.status else None
        return InternalAIModelObject(**model) if model else None

    def get_all_ai_model_list(
        self, model_category_list=None, user_id=None, custom_trained=None, model_type_list=None
    ):
        from utils.common_utils import get_current_user_uuid

        if not user_id:
            user_id = get_current_user_uuid()

        model_list = self.db_repo.get_all_ai_model_list(
            model_category_list, user_id, custom_trained, model_type_list
        ).data["data"]
        return [InternalAIModelObject(**model) for model in model_list] if model_list else []

    def create_ai_model(self, **kwargs):
        model = self.db_repo.create_ai_model(**kwargs).data["data"]
        return InternalAIModelObject(**model) if model else None

    def update_ai_model(self, **kwargs):
        model = self.db_repo.update_ai_model(**kwargs).data["data"]
        return InternalAIModelObject(**model) if model else None

    def delete_ai_model_from_uuid(self, uuid):
        res = self.db_repo.delete_ai_model_from_uuid(uuid)
        return res.status

    # inference log
    def get_inference_log_from_uuid(self, uuid):
        res = self.db_repo.get_inference_log_from_uuid(uuid)
        log = res.data["data"] if res else None
        return InferenceLogObject(**log) if log else None

    def get_all_inference_log_list(self, **kwargs):
        res = self.db_repo.get_all_inference_log_list(**kwargs)
        log_list = res.data["data"] if res.status else None
        total_page_count = res.data["total_pages"] if res.status else None

        return ([InferenceLogObject(**log) for log in log_list] if log_list else None, total_page_count)

    def create_inference_log(self, **kwargs):
        res = self.db_repo.create_inference_log(**kwargs)
        log = res.data["data"] if res else None
        return InferenceLogObject(**log) if log else None

    def delete_inference_log_from_uuid(self, uuid):
        res = self.db_repo.delete_inference_log_from_uuid(uuid)
        return res.status

    def update_inference_log(self, uuid, **kwargs):
        res = self.db_repo.update_inference_log(uuid, **kwargs)
        return res.status

    def update_inference_log_list(self, uuid_list, **kwargs):
        res = self.db_repo.update_inference_log_list(uuid_list, **kwargs)
        return res.status

    def update_inference_log_origin_data(self, uuid, **kwargs):
        res = self.get_inference_log_from_uuid(uuid)
        if not res:
            return False

        input_params_data = json.loads(res.input_params)
        input_params_data[InferenceParamType.ORIGIN_DATA.value] = dict(kwargs)

        status = self.update_inference_log(uuid, input_params=json.dumps(input_params_data))
        return status

    # ai model param map
    # TODO: add DTO in the output
    def get_ai_model_param_map_from_uuid(self, uuid):
        pass

    def get_all_ai_model_param_map_list(self, model_id=None):
        pass

    def create_ai_model_param_map(self, **kwargs):
        pass

    def delete_ai_model(self, uuid):
        pass

    # timing
    def get_timing_from_uuid(self, uuid, **kwargs):
        timing = self.db_repo.get_timing_from_uuid(uuid).data["data"]
        return InternalFrameTimingObject(**timing) if timing else None

    def get_timing_from_frame_number(self, shot_uuid, frame_number):
        res = self.db_repo.get_timing_from_frame_number(shot_uuid, frame_number)
        timing = res.data["data"] if res.status else None
        return InternalFrameTimingObject(**timing) if timing else None

    # this is based on the aux_frame_index and not the order in the db
    def get_next_timing(self, uuid):
        next_timing = self.db_repo.get_next_timing(uuid).data["data"]
        return InternalFrameTimingObject(**next_timing) if next_timing else None

    def get_prev_timing(self, uuid):
        prev_timing = self.db_repo.get_prev_timing(uuid).data["data"]
        return InternalFrameTimingObject(**prev_timing) if prev_timing else None

    def get_timing_list_from_project(self, project_uuid=None):
        res = self.db_repo.get_timing_list_from_project(project_uuid)
        timing_list = res.data["data"] if res.status else None
        return [InternalFrameTimingObject(**timing) for timing in timing_list] if timing_list else []

    def get_timing_list_from_shot(self, shot_uuid=None):
        res = self.db_repo.get_timing_list_from_shot(shot_uuid)
        timing_list = res.data["data"] if res.status else None
        return [InternalFrameTimingObject(**timing) for timing in timing_list] if timing_list else []

    def create_timing(self, **kwargs):
        res = self.db_repo.create_timing(**kwargs)
        timing = res.data["data"] if res.status else None
        return InternalFrameTimingObject(**timing) if timing else None

    def update_specific_timing(self, uuid, **kwargs):
        res = self.db_repo.update_specific_timing(uuid, **kwargs)
        return res.status

    # NOTE: this method focuses on speed and therefore bypasses aux_frame update for individual saves
    # only use it for updating timings if their relative position is not affected
    def update_bulk_timing(self, timing_uuid_list, data_list):
        res = self.db_repo.update_bulk_timing(timing_uuid_list, data_list)
        return res.status

    # NOTE: this method focuses on speed and therefore bypasses aux_frame update for individual saves
    # only use it for updating timings if their relative position is not affected
    def bulk_create_timing(self, data_list):
        res = self.db_repo.bulk_create_timing(data_list)
        return res.status

    def delete_timing_from_uuid(self, uuid):
        res = self.db_repo.delete_timing_from_uuid(uuid)
        return res.status

    # removes all timing frames from the project
    def remove_existing_timing(self, project_uuid):
        res = self.db_repo.remove_existing_timing(project_uuid)
        return res.status

    def remove_primary_frame(self, timing_uuid):
        res = self.db_repo.remove_primary_frame(timing_uuid)
        return res.status

    def remove_source_image(self, timing_uuid):
        res = self.db_repo.remove_source_image(timing_uuid)
        return res.status

    # app setting
    def get_app_setting_from_uuid(self, uuid=None):
        res = self.db_repo.get_app_setting_from_uuid(uuid)
        app_setting = res.data["data"] if res.status else None
        return InternalAppSettingObject(**app_setting) if app_setting else None

    def get_app_secrets_from_user_uuid(self, uuid=None):
        from utils.common_utils import get_current_user_uuid

        # if user is not defined then take the current user
        if not uuid:
            uuid = get_current_user_uuid()

        app_secrets = self.db_repo.get_app_secrets_from_user_uuid(
            uuid, secret_access=SECRET_ACCESS_TOKEN
        ).data["data"]
        return app_secrets

    def get_all_app_setting_list(self):
        app_setting_list = self.db_repo.get_all_app_setting_list().data["data"]
        return (
            [InternalAppSettingObject(**app_setting) for app_setting in app_setting_list]
            if app_setting_list
            else None
        )

    def update_app_setting(self, **kwargs):
        res = self.db_repo.update_app_setting(**kwargs)
        return res.status

    def create_app_setting(self, **kwargs):
        app_setting = self.db_repo.create_app_setting(**kwargs).data["data"]
        return InternalAppSettingObject(**app_setting) if app_setting else None

    def delete_app_setting(self, user_id):
        res = self.db_repo.delete_app_setting(user_id)
        return res.status

    # setting
    def get_project_setting(self, project_id):
        res = self.db_repo.get_project_setting(project_id)
        project_setting = res.data["data"] if res.status else None
        return InternalSettingObject(**project_setting) if project_setting else None

    # TODO: add valid model_id check throughout dp_repo
    def create_project_setting(self, **kwargs):
        res = self.db_repo.create_project_setting(**kwargs)
        project_setting = res.data["data"] if res.status else None
        return InternalSettingObject(**project_setting) if project_setting else None

    # TODO: remove db calls for updating guidance_type
    def update_project_setting(self, project_uuid, **kwargs):
        kwargs["project_id"] = project_uuid
        project_setting = self.db_repo.update_project_setting(**kwargs).data["data"]
        return InternalSettingObject(**project_setting) if project_setting else None

    def bulk_update_project_setting(self, **kwargs):
        res = self.db_repo.bulk_update_project_setting(**kwargs)
        return res.status

    # backup
    def get_backup_from_uuid(self, uuid):
        backup = self.db_repo.get_backup_from_uuid(uuid).data["data"]
        return InternalBackupObject(**backup) if backup else None

    def create_backup(self, project_uuid, version_name):
        backup = self.db_repo.create_backup(project_uuid, version_name).data["data"]
        return InternalBackupObject(**backup) if backup else None

    def get_backup_list(self, project_id=None):
        backup_list = self.db_repo.get_backup_list(project_id).data["data"]
        return [InternalBackupObject(**backup) for backup in backup_list] if backup_list else []

    def delete_backup(self, uuid):
        res = self.db_repo.delete_backup(uuid)
        return res.status

    def restore_backup(self, uuid):
        res = self.db_repo.restore_backup(uuid)
        return res.status

    # update user credits - updates the credit of the user calling the API
    def update_usage_credits(self, credits_to_add, log_uuid=None):
        user_id = None
        if log_uuid:
            log = self.get_inference_log_from_uuid(log_uuid)
            user_id = log.project.user_uuid

        user = self.update_user(user_id=user_id, credits_to_add=credits_to_add)
        return user

    def generate_payment_link(self, amount):
        res = self.db_repo.generate_payment_link(amount)
        link = res.data["data"] if res.status else None
        return link

    # shot
    def get_shot_from_uuid(self, shot_uuid):
        res = self.db_repo.get_shot_from_uuid(shot_uuid)
        shot = res.data["data"] if res.status else None
        return InternalShotObject(**shot) if shot else None

    def get_shot_from_number(self, project_uuid, shot_number=0):
        res = self.db_repo.get_shot_from_number(project_uuid, shot_number)
        shot = res.data["data"] if res.status else None
        return InternalShotObject(**shot) if shot else None

    def get_shot_list(self, project_uuid, invalidate_cache=False):
        res = self.db_repo.get_shot_list(project_uuid)
        shot_list = res.data["data"] if res.status else None
        return [InternalShotObject(**shot) for shot in shot_list] if shot_list else []

    def create_shot(self, project_uuid, duration, name="", meta_data="", desc=""):
        res = self.db_repo.create_shot(project_uuid, duration, name, meta_data, desc)
        shot = res.data["data"] if res.status else None
        return InternalShotObject(**shot) if shot else None

    # shot_uuid, shot_idx, name, duration, meta_data, desc, main_clip_id
    def update_shot(self, **kwargs):
        res = self.db_repo.update_shot(**kwargs)
        return res.status

    def delete_shot(self, shot_uuid):
        res = self.db_repo.delete_shot(shot_uuid)
        return res.status

    def duplicate_shot(self, shot_uuid):
        res = self.db_repo.duplicate_shot(shot_uuid)
        return res.status

    def add_interpolated_clip(self, shot_uuid, **kwargs):
        res = self.db_repo.add_interpolated_clip(shot_uuid, **kwargs)
        return res.status

    # combined
    # gives the count of 1. temp generated images 2. inference logs with in-progress/pending status
    def get_explorer_pending_stats(self, project_uuid):
        log_status_list = [InferenceStatus.IN_PROGRESS.value, InferenceStatus.QUEUED.value]
        res = self.db_repo.get_explorer_pending_stats(project_uuid, log_status_list)
        count_data = res.data["data"] if res.status else {"temp_image_count": 0, "pending_image_count": 0}
        return count_data
