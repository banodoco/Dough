from utils.enum import ExtendedEnum
from dataclasses import dataclass


##################### enums #####################
class ServerType(ExtendedEnum):
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'

class InternalResponse:
    def __init__(self, data, message, status):
        self.status = status
        self.message = message
        self.data = data

class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

class AIModelType(ExtendedEnum):
    LORA = 'LoRA'
    DREAMBOOTH = 'Dreambooth'
    BASE_SD = 'Base_SD'
    CONTROLNET = 'controlnet'
    STYLEGAN_NADA = "StyleGAN-NADA"
    PIX_2_PIX = 'pix2pix'

class GuidanceType(ExtendedEnum):
    DRAWING = 'drawing'
    IMAGE = 'image'

class InternalFileType(ExtendedEnum):
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    GIF = 'gif'

# Internal file tags
class InternalFileTag(ExtendedEnum):
    BACKGROUND_IMAGE = 'background_image'
    GENERATED_VIDEO = 'generated_video'
    INPUT_VIDEO = 'input_video'
    TEMP_IMAGE = 'temp'

class AnimationStyleType(ExtendedEnum):
    INTERPOLATION = "Interpolation"
    DIRECT_MORPHING = "Direct Morphing"


##################### global constants #####################
SERVER = ServerType.STAGING.value

AUTOMATIC_FILE_HOSTING = SERVER != ServerType.DEVELOPMENT.value  # automatically upload project files to s3 (images, videos, gifs)
AWS_S3_BUCKET = 'banodoco'
AWS_S3_REGION = 'ap-south-1'    # TODO: discuss this

LOCAL_DATABASE_NAME = 'banodoco_local.db'
