import mimetypes
from urllib.parse import urlparse
import boto3
import uuid
import os
import shutil

import requests
from shared.constants import AWS_S3_BUCKET, AWS_S3_REGION
from shared.logging.logging import AppLogger
from shared.logging.constants import LoggingPayload, LoggingType
logger = AppLogger()

# TODO: fix proper paths for file uploads 

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
        logger.log(LoggingType.ERROR, 'unable to upload to s3')

    return url

def upload_file_from_obj(file, aws_access_key, aws_secret_key, bucket=AWS_S3_BUCKET):
    folder = 'test/'
    unique_tag = str(uuid.uuid4())
    file_extension = os.path.splitext(file.name)[1]
    filename = unique_tag + file_extension

    # Upload the file
    content_type = mimetypes.guess_type(file.name)[0]
    data = {
        "Body": file,
        "Bucket": bucket,
        "Key": folder + filename,
        "ACL": "public-read"
    }
    if content_type:
        data['ContentType'] = content_type
    
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key,
                          aws_secret_access_key=aws_secret_key)
    resp = s3_client.put_object(**data)
    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        AWS_S3_REGION,
        bucket,
        folder + filename)
    return object_url

def upload_file_from_bytes(file_bytes, aws_access_key, aws_secret_key, key=None, bucket=AWS_S3_BUCKET):
    if not key:
        key = 'test/' + str(uuid.uuid4()) + '.png'
    
    content_type = 'image/png'
    data = {
        "Body": file_bytes,
        "Bucket": bucket,
        "Key": key,
        "ACL": "public-read"
    }
    if content_type:
        data['ContentType'] = content_type
    
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key,
                          aws_secret_access_key=aws_secret_key)
    resp = s3_client.put_object(**data)
    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        AWS_S3_REGION,
        bucket,
        key)
    return object_url

# TODO: fix the structuring of s3 for different users and different files
def generate_s3_url(image_url, aws_access_key, aws_secret_key, bucket=AWS_S3_BUCKET, file_ext='png', folder='posts/'):
    if object_name is None:
        object_name = str(uuid.uuid4()) + '.' + file_ext

    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception("Failed to download the image from the given URL")

    file = response.content

    content_type = mimetypes.guess_type(object_name)[0]
    data = {
        "Body": file,
        "Bucket": bucket,
        "Key": folder + object_name,
        "ACL": "public-read"
    }
    if content_type:
        data['ContentType'] = content_type
    else:
        data['ContentType'] = 'image/png'

    s3_client = boto3.client(
            service_name='s3',
            region_name=AWS_S3_REGION,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    resp = s3_client.put_object(**data)

    extension = os.path.splitext(object_name)[1]
    disposition = f'inline; filename="{object_name}"'
    if extension:
        disposition += f'; filename="{object_name}"'
    resp['ResponseMetadata']['HTTPHeaders']['Content-Disposition'] = disposition

    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        AWS_S3_REGION,
        AWS_S3_BUCKET,
        folder + object_name)
    return object_url

def is_s3_image_url(url):
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()

    if netloc.endswith('.amazonaws.com'):
        subdomain = netloc[:-len('.amazonaws.com')].split('-')
        if len(subdomain) > 1 and subdomain[0] == 's3':
            return True

    return False