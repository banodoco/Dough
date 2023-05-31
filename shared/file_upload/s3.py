import boto3
import uuid
import os
import shutil
from shared.constants import AWS_S3_BUCKET, SERVER, ServerType
from shared.logging.logging import AppLogger
from shared.logging.constants import LoggingPayload, LoggingType


def upload_file(file_location, aws_access_key, aws_secret_key, bucket=AWS_S3_BUCKET):
    url = None
    ext = os.path.splitext(file_location)[1]
    unique_file_name = str(uuid.uuid4()) + ext
    try:
        s3_file = f"input_images/{unique_file_name}"
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                          aws_secret_access_key=aws_secret_key)
        s3.upload_file(file_location, bucket, s3_file)
        s3.put_object_acl(ACL='public-read', Bucket=bucket, Key=s3_file)
        url = f"https://s3.amazonaws.com/{bucket}/{s3_file}"
    except Exception as e:
        # saving locally in the code directory if S3 upload fails (ONLY for LOCAL/DEV SERVER)
        if SERVER != ServerType.PRODUCTION.value:
            logger = AppLogger()
            logger.log(LoggingType.ERROR, LoggingPayload(
                message=str(e), data={}))

            # TODO: fix the local destinations for different files
            dest = "videos/controlnet_test/assets/frames/1_selected/" + unique_file_name
            shutil.copy(file_location, dest)
            url = dest

    return url