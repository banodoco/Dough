import os
from dotenv import dotenv_values

from shared.constants import SERVER, ServerType
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.models import InternalUserObject
from utils.common_methods import create_working_assets
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
    # if this is the local server with no user than create one
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

        # creating it's app setting as well
        setting_data = {
            "user_id": user.uuid,
            "welcome_state": 0
        }
        app_setting = data_repo.create_app_setting(**setting_data)

        # creating a new project for this user
        project_data = {
            "user_id": user.uuid,
            "name": "my_first_project",
        }
        project = data_repo.create_project(**project_data)

    app_secret = data_repo.get_app_secrets_from_user_uuid()

    REPLICATE_API_TOKEN = app_secret["replicate_key"]
    REPLICATE_USERNAME = app_secret["replicate_username"]

    AWS_ACCESS_KEY_ID = app_secret["aws_access_key"]
    AWS_SECRET_ACCESS_KEY = app_secret["aws_secret_key"]

    # create asset directories
    create_working_assets('controlnet_test')

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
    