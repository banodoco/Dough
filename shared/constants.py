from utils.enum import ExtendedEnum


class ServerType(ExtendedEnum):
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'

SERVER = ServerType.STAGING.value

AUTOMATIC_FILE_HOSTING = SERVER == ServerType.PRODUCTION.value  # automatically upload project files to s3 (images, videos, gifs)
AWS_S3_BUCKET = 'banodoco'

LOCAL_DATABASE_NAME = 'banodoco_local.db'