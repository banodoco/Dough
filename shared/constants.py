import os
from typing import Any
import toml
from utils.enum import ExtendedEnum
from dotenv import load_dotenv

load_dotenv(override=True)


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = {}
            cls._instance.toml_file = "app_settings.toml"
            cls._instance.load_from_toml()
        return cls._instance

    def _get_toml_app_settings(self, key=None):
        # print("----- fresh toml load")
        default_settings_dict = {"automatic_update": True, "gpu_inference": True}
        toml_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", self.toml_file)
        )

        toml_data = {}
        with open(toml_config_path, "r") as f:
            toml_data = toml.load(f)

        if key and key in toml_data:
            return toml_data[key]

        for k, v in default_settings_dict.items():
            if k not in toml_data:
                toml_data[k] = v

        return toml_data

    def _update_toml_app_settings(self):
        toml_dict = self.config
        toml_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "scripts", self.toml_file)
        )

        with open(toml_config_path, "wb") as f:
            toml_content = toml.dumps(toml_dict)
            f.write(toml_content.encode())

    def load_from_toml(self):
        self.config = self._get_toml_app_settings()

    def get(self, key: str, default: Any = None, fresh_pull=False) -> Any:
        if fresh_pull:
            self.load_from_toml()
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self._update_toml_app_settings()


singleton_config_manager = ConfigManager()


##################### enums #####################
class ServerType(ExtendedEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


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
    LORA = "LoRA"
    DREAMBOOTH = "Dreambooth"
    BASE_SD = "Base_SD"
    CONTROLNET = "controlnet"
    STYLEGAN_NADA = "StyleGAN-NADA"
    PIX_2_PIX = "pix2pix"


class AIModelType(ExtendedEnum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    TXT2VID = "txt2vid"
    IMG2VID = "img2vid"
    VID2VID = "vid2vid"


class GuidanceType(ExtendedEnum):
    DRAWING = "drawing"
    IMAGE = "image"


class InternalFileType(ExtendedEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    GIF = "gif"


# Internal file tags
class InternalFileTag(ExtendedEnum):
    BACKGROUND_IMAGE = "background_image"
    GENERATED_VIDEO = "generated_video"
    COMPLETE_GENERATED_VIDEO = "complete_generated_video"
    INPUT_VIDEO = "input_video"
    TEMP_IMAGE = "temp"
    GALLERY_IMAGE = "gallery_image"
    SHORTLISTED_GALLERY_IMAGE = "shortlisted_gallery_image"
    TEMP_GALLERY_IMAGE = (
        "temp_gallery_image"  # these generations are complete but not yet being shown in the gallery
    )
    SHORTLISTED_VIDEO = "shortlisted_video"


class AnimationStyleType(ExtendedEnum):
    CREATIVE_INTERPOLATION = "Creative Interpolation"
    IMAGE_TO_VIDEO = "Image to video"
    DIRECT_MORPHING = "None"


class AnimationToolType(ExtendedEnum):
    ANIMATEDIFF = "Animatediff"
    G_FILM = "Google FiLM"


class ViewType(ExtendedEnum):
    SINGLE = "Single"
    LIST = "List"


class InferenceType(ExtendedEnum):
    FRAME_TIMING_IMAGE_INFERENCE = "frame_timing_inference"  # for generating variants of a frame
    FRAME_TIMING_VIDEO_INFERENCE = "frame_timing_video_inference"  # for generating variants of a video
    FRAME_INTERPOLATION = "frame_interpolation"  # for generating single/multiple interpolated videos
    GALLERY_IMAGE_GENERATION = "gallery_image_generation"  # for generating gallery images
    FRAME_INPAINTING = "frame_inpainting"  # for generating inpainted frames
    MOTION_LORA_TRAINING = "motion_lora_training"  # for training new motion loras


class InferenceLogTag(ExtendedEnum):
    UPSCALED_VIDEO = "upscaled_video"
    PREVIEW_VIEW = "preview"


class FileTransformationType(ExtendedEnum):
    UPSCALE = "upscale"


class InferenceStatus(ExtendedEnum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    BACKLOG = "backlog"


class InferenceParamType(ExtendedEnum):
    REPLICATE_INFERENCE = "replicate_inference"  # replicate url for queue inference and other data
    QUERY_DICT = "query_dict"  # query dict of standardized inference params
    ORIGIN_DATA = "origin_data"  # origin data - used to store file once inference is completed
    GPU_INFERENCE = "gpu_inference"  # gpu inference data
    SAI_INFERENCE = "sai_inference"  # stablity ai inference data
    FILE_RELATION_DATA = (
        "file_relation"  # file relationship data such as what parent a particular file was upscaled from
    )
    API_INFERENCE_DATA = "api_inference"  # banodoco's hosted service


class ProjectMetaData(ExtendedEnum):
    DATA_UPDATE = "data_update"  # info regarding cache/data update when runner updates the db
    GALLERY_UPDATE = "gallery_update"
    BACKGROUND_IMG_LIST = "background_img_list"
    SHOT_VIDEO_UPDATE = "shot_video_update"
    ACTIVE_SHOT = "active_shot"  # the most recent shot where the generation took place or where "load settings" was clicked (settings of this will be used as default)
    INSP_VALUES = "insp_values"


class SortOrder(ExtendedEnum):
    ASCENDING = "asc"
    DESCENDING = "desc"


class CreativeProcessPage(ExtendedEnum):
    SHOTS = "Shots"
    INSPIRATION_ENGINE = "Inspiration Engine"
    ADJUST_SHOT = "Adjust Shot"
    ANIMATE_SHOT = "Animate Shot"
    UPSCALING = "Upscaling"


# these can be one of the main creative process page or some other sub page inside it
class AppSubPage(ExtendedEnum):
    SHOTS = "Shots"
    INSPIRATION_ENGINE = "Inspiration Engine"
    ADJUST_SHOT = "Adjust Shot"
    ANIMATE_SHOT = "Animate Shot"
    KEYFRAME = "Key Frames"
    UPSCALING = "Upscaling"
    SHOT = "Shots"


STEERABLE_MOTION_WORKFLOWS = [
    {"name": "Slurshy Realistiche", "order": 5, "display": True},
    {"name": "Smooth n' Steady", "order": 1, "display": True},
    {"name": "Chocky Realistiche", "order": 2, "display": True},
    {"name": "Liquidy Loop", "order": 3, "display": True},
    {"name": "Fast With A Price", "order": 4, "display": True},
    {"name": "Rad Attack", "order": 0, "display": True},
]
##################### global constants #####################
SERVER = os.getenv("SERVER", ServerType.PRODUCTION.value)

AUTOMATIC_FILE_HOSTING = (
    SERVER != ServerType.DEVELOPMENT.value
)  # automatically upload project files to s3 (images, videos, gifs)
AWS_S3_BUCKET = "banodoco-data-bucket-public"
AWS_S3_REGION = "ap-south-1"
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "")
OFFLINE_MODE = os.getenv("OFFLINE_MODE", False)  # for picking up secrets and file storage
COMFY_BASE_PATH = os.getenv("COMFY_MODELS_BASE_PATH", "ComfyUI") or "ComfyUI"
SERVER_URL = os.getenv("SERVER_URL", "https://api.banodoco.ai")

LOCAL_DATABASE_NAME = "banodoco_local.db"
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "J2684nBgNUYa_K0a6oBr5H8MpSRW0EJ52Qmq7jExE-w=")

QUEUE_INFERENCE_QUERIES = True
HOSTED_BACKGROUND_RUNNER_MODE = os.getenv("HOSTED_BACKGROUND_RUNNER_MODE", False)

GPU_INFERENCE_ENABLED_KEY = "gpu_inference"
AUTOMATIC_UPDATE_KEY = "automatic_update"


if OFFLINE_MODE:
    SECRET_ACCESS_TOKEN = os.getenv("SECRET_ACCESS_TOKEN", None)
else:
    import boto3

    ssm = boto3.client("ssm", region_name="ap-south-1")

    SECRET_ACCESS_TOKEN = ssm.get_parameter(Name="/backend/banodoco/secret-access-token")["Parameter"][
        "Value"
    ]

COMFY_PORT = 4333
