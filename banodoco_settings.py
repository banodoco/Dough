import os
import shutil
from dotenv import dotenv_values

from shared.constants import SERVER, GuidanceType, ServerType
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.constants import AnimationStyleType
from ui_components.models import InternalUserObject
from utils.common_methods import copy_sample_assets, create_working_assets
from utils.data_repo.data_repo import DataRepo

REPLICATE_API_TOKEN = None
REPLICATE_USERNAME = None

ENCRYPTION_KEY = 'J2684nBgNUYa_K0a6oBr5H8MpSRW0EJ52Qmq7jExE-w='

logger = AppLogger()

def project_init():
    global REPLICATE_API_TOKEN
    global REPLICATE_USERNAME
    global AWS_ACCESS_KEY_ID
    global AWS_SECRET_ACCESS_KEY

    data_repo = DataRepo()

    # create a user if not already present (if dev mode)
    # if this is the local server with no user than create one and related data (it's handled by the backend)
    user_count = data_repo.get_total_user_count()
    if SERVER != ServerType.PRODUCTION.value and not user_count:
        user_data = {
            "name" : "banodoco_user",
            "email" : "banodoco@tempuser.com",
            "password" : "123",
            "type" : "user"
        }
        user: InternalUserObject = data_repo.create_user(**user_data)
        logger.log(LoggingType.INFO, "new temp user created: " + user.name)

        create_new_user_data(user)

    app_secret = data_repo.get_app_secrets_from_user_uuid()

    REPLICATE_API_TOKEN = app_secret["replicate_key"]
    REPLICATE_USERNAME = app_secret["replicate_username"]

    AWS_ACCESS_KEY_ID = app_secret["aws_access_key"]
    AWS_SECRET_ACCESS_KEY = app_secret["aws_secret_key"]

    # create encryption key if not already present (not applicable in dev mode)
    # env_vars = dotenv_values('.env')
    # desired_key = 'FERNET_KEY'
    # global ENCRYPTION_KEY
    # if desired_key in env_vars:
    #     ENCRYPTION_KEY = env_vars[desired_key].decode()
    # else:
    #     from cryptography.fernet import Fernet

    #     secret_key = Fernet.generate_key()
    #     with open('.env', 'a') as env_file:
    #         env_file.write(f'FERNET_KEY={secret_key.decode()}\n')
        
    #     ENCRYPTION_KEY = secret_key.decode()


# sample data required for starting the project like app settings, project settings
def create_new_user_data(user: InternalUserObject):
    data_repo = DataRepo()
    
    # TODO: write this logic in a separate signal + make it an atomic transaction
    # creating it's app setting
    setting_data = {
        "user_id": user.uuid,
        "welcome_state": 0
    }
    app_setting = data_repo.create_app_setting(**setting_data)

    # creating a new project for this user
    project_data = {
        "user_id": user.uuid,
        "name": "my_first_project",
        'width': 512,
        'height': 512
    }
    project = data_repo.create_project(**project_data)

    # creating a project settings for this
    project_setting_data = {
        "project_id" : project.uuid,
        "input_type" : "video",
        "default_strength": 0.63,
        "extraction_type" : "Extract manually",
        "width" : 512,
        "height" : 512,
        "default_negative_prompt" : "",
        "default_guidance_scale" : 7.5,
        "default_seed" : 1234,
        "default_num_inference_steps" : 30,
        "default_stage" : "Extracted Key Frames",
        "default_custom_model_id_list" : "[]",
        "default_adapter_type" : "N",
        "guidance_type" : GuidanceType.DRAWING.value,
        "default_animation_style" : AnimationStyleType.INTERPOLATION.value,
        "default_low_threshold" : 0,
        "default_high_threshold" : 0
    }

    project_setting = data_repo.create_project_setting(**project_setting_data)

    create_working_assets(project.uuid)