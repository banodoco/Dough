from repository.local_repo.csv_repo import get_app_settings
from utils.common_methods import create_working_assets


REPLICATE_API_TOKEN = None
REPLICATE_USERNAME = None

AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None

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
