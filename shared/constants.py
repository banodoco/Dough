import os
from utils.enum import ExtendedEnum
from dotenv import load_dotenv

load_dotenv()


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

class AIModelCategory(ExtendedEnum):
    LORA = 'LoRA'
    DREAMBOOTH = 'Dreambooth'
    BASE_SD = 'Base_SD'
    CONTROLNET = 'controlnet'
    STYLEGAN_NADA = "StyleGAN-NADA"
    PIX_2_PIX = 'pix2pix'

class AIModelType(ExtendedEnum):
    TXT2IMG = 'txt2img'
    IMG2IMG = 'img2img'
    TXT2VID = 'txt2vid'
    IMG2VID = 'img2vid'
    VID2VID = 'vid2vid'

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
    COMPLETE_GENERATED_VIDEO = 'complete_generated_video'
    INPUT_VIDEO = 'input_video'
    TEMP_IMAGE = 'temp'

class AnimationStyleType(ExtendedEnum):
    INTERPOLATION = "Interpolate to next"
    IMAGE_TO_VIDEO = "Image to video"
    DIRECT_MORPHING = "None"

class AnimationToolType(ExtendedEnum):
    ANIMATEDIFF = 'Animatediff'
    G_FILM = "Google FiLM"


##################### global constants #####################
SERVER = os.getenv('SERVER', ServerType.PRODUCTION.value)

AUTOMATIC_FILE_HOSTING = SERVER != ServerType.DEVELOPMENT.value  # automatically upload project files to s3 (images, videos, gifs)
AWS_S3_BUCKET = 'banodoco'
AWS_S3_REGION = 'ap-south-1'    # TODO: discuss this
OFFLINE_MODE = os.getenv('OFFLINE_MODE', False)     # for picking up secrets and file storage

LOCAL_DATABASE_NAME = 'banodoco_local.db'

REPLICATE_USER = "piyushk52"