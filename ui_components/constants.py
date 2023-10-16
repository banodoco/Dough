from shared.constants import AnimationStyleType, AnimationToolType
from utils.constants import ImageStage
from utils.enum import ExtendedEnum


class WorkflowStageType(ExtendedEnum):
    SOURCE = "source"
    STYLED = "styled"
    

class VideoQuality(ExtendedEnum):
    HIGH = "High-Quality"
    PREVIEW = "Preview"
    LOW = "Low"

class CreativeProcessType(ExtendedEnum):
    STYLING = "Key Frames"
    MOTION = "Videos"

class DefaultTimingStyleParams:
    prompt = ""
    negative_prompt = "bad image, worst quality"
    strength = 1
    guidance_scale = 7.5
    seed = 0
    num_inference_steps = 25
    low_threshold = 100
    high_threshold = 200
    adapter_type = None
    interpolation_steps = 3
    transformation_stage = ImageStage.SOURCE_IMAGE.value
    custom_model_id_list = []
    animation_tool = AnimationToolType.G_FILM.value
    animation_style = AnimationStyleType.INTERPOLATION.value
    model = None

class DefaultProjectSettingParams:
    batch_prompt = ""
    batch_negative_prompt = "bad image, worst quality"
    batch_strength = 1
    batch_guidance_scale = 0.5
    batch_seed = 0
    batch_num_inference_steps = 25
    batch_low_threshold = 100
    batch_high_threshold = 200
    batch_adapter_type = None
    batch_interpolation_steps = 3
    batch_transformation_stage = ImageStage.SOURCE_IMAGE.value
    batch_custom_model_id_list = []
    batch_animation_tool = AnimationToolType.G_FILM.value
    batch_animation_style = AnimationStyleType.INTERPOLATION.value
    batch_model = None
    total_log_pages = 1

# TODO: make proper paths for every file
CROPPED_IMG_LOCAL_PATH = "videos/temp/cropped.png"

MASK_IMG_LOCAL_PATH = "videos/temp/mask.png"
TEMP_MASK_FILE = 'temp_mask_file'

SECOND_MASK_FILE_PATH = 'videos/temp/second_mask.png'
SECOND_MASK_FILE = 'second_mask_file'

AUDIO_FILE_PATH = 'videos/temp/audio.mp3'
AUDIO_FILE = 'audio_file'