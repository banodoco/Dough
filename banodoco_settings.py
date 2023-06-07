import os
from dotenv import dotenv_values

from shared.constants import SERVER, ServerType
from utils.common_methods import create_working_assets
from utils.data_repo.data_repo import DataRepo

REPLICATE_API_TOKEN = None
REPLICATE_USERNAME = None

ENCRYPTION_KEY = None

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
        data_repo.create_user(**user_data)

    app_secret = data_repo.get_app_secrets_from_user_uuid()

    REPLICATE_API_TOKEN = app_secret["replicate_key"]
    REPLICATE_USERNAME = app_secret["replicate_username"]

    AWS_ACCESS_KEY_ID = app_secret["aws_access_key"]
    AWS_SECRET_ACCESS_KEY = app_secret["aws_secret_key"]

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
    