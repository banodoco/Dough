from utils.enum import ExtendedEnum


class WorkflowStageType(ExtendedEnum):
    SOURCE = "source"
    STYLED = "styled"
    

class VideoQuality(ExtendedEnum):
    HIGH = "High-Quality"
    PREVIEW = "Preview"
    LOW = "Low"


# TODO: make proper paths for every file
CROPPED_IMG_LOCAL_PATH = "videos/temp/cropped.png"

MASK_IMG_LOCAL_PATH = "videos/temp/mask.png"
TEMP_MASK_FILE = 'temp_mask_file'

SECOND_MASK_FILE_PATH = 'videos/temp/second_mask.png'
SECOND_MASK_FILE = 'second_mask_file'

AUDIO_FILE_PATH = 'videos/temp/audio.mp3'
AUDIO_FILE = 'audio_file'