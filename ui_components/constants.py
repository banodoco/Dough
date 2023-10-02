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
    STYLING = "Styling"
    MOTION = "Motion"

class DefaultTimingStyleParams:
    prompt = ""
    negative_prompt = "bad image, worst quality"
    strength = 1
    guidance_scale = 0.5
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

# TODO: make proper paths for every file
CROPPED_IMG_LOCAL_PATH = "videos/temp/cropped.png"

MASK_IMG_LOCAL_PATH = "videos/temp/mask.png"
TEMP_MASK_FILE = 'temp_mask_file'

SECOND_MASK_FILE_PATH = 'videos/temp/second_mask.png'
SECOND_MASK_FILE = 'second_mask_file'

AUDIO_FILE_PATH = 'videos/temp/audio.mp3'
AUDIO_FILE = 'audio_file'