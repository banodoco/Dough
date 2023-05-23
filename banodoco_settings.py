from dotenv import dotenv_values

from repository.local_repo.csv_repo import get_app_settings
from utils.common_methods import create_working_assets
from utils.enum import ExtendedEnum


class ServerType(ExtendedEnum):
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'

REPLICATE_API_TOKEN = None
REPLICATE_USERNAME = None

AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None

SERVER = ServerType.STAGING.value

LOCAL_DATABASE_LOCATION = 'banodoco_local.db'
ENCRYPTION_KEY = None

def project_init():
    app_settings = get_app_settings()

    global REPLICATE_API_TOKEN
    global REPLICATE_USERNAME
    global AWS_ACCESS_KEY_ID
    global AWS_SECRET_ACCESS_KEY

    REPLICATE_API_TOKEN = app_settings["replicate_com_api_key"]
    REPLICATE_USERNAME = app_settings["replicate_user_name"]

    AWS_ACCESS_KEY_ID = app_settings["aws_access_key_id"]
    AWS_SECRET_ACCESS_KEY = app_settings["aws_secret_access_key"]

    # create asset directories
    create_working_assets('controlnet_test')

    # create encryption key if not already present
    env_vars = dotenv_values('.env')
    desired_key = 'FERNET_KEY'
    global ENCRYPTION_KEY
    if desired_key in env_vars:
        ENCRYPTION_KEY = env_vars[desired_key].decode()
    else:
        from cryptography.fernet import Fernet

        secret_key = Fernet.generate_key()
        with open('.env', 'a') as env_file:
            env_file.write(f'FERNET_KEY={secret_key.decode()}\n')
        
        ENCRYPTION_KEY = secret_key.decode()


