import boto3
import uuid
import shutil
from repository.local_repo.csv_repo import CSVProcessor, get_app_settings
from utils.logging.constants import LoggingPayload, LoggingType
from utils.logging.logging import AppLogger


def upload_image(image_location):
    app_settings = get_app_settings()

    url = None

    unique_file_name = str(uuid.uuid4()) + ".png"
    try:
        s3_file = f"input_images/{unique_file_name}"
        s3 = boto3.client('s3', aws_access_key_id=app_settings['aws_access_key_id'],
                        aws_secret_access_key=app_settings['aws_secret_access_key'])
        s3.upload_file(image_location, "banodoco", s3_file)
        s3.put_object_acl(ACL='public-read', Bucket='banodoco', Key=s3_file)
        url = f"https://s3.amazonaws.com/banodoco/{s3_file}"
    except Exception as e:
        # saving locally if S3 upload fails
        logger = AppLogger()
        logger.log(LoggingType.ERROR, LoggingPayload(message=str(e), data={}))

        dest = "videos/controlnet_test/assets/frames/1_selected/" + unique_file_name
        shutil.copy(image_location, dest)
        url = dest

    return url