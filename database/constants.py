from utils.enum import ExtendedEnum

class InternalFileType(ExtendedEnum):
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'

class UserType(ExtendedEnum):
    USER = 'user'
    ADMIN = 'admin'

class AIModelType(ExtendedEnum):
    LORA = 'LoRA'
    DREAMBOOTH = 'Dreambooth'

class GuidanceType(ExtendedEnum):
    DRAWING = 'drawing'
    IMAGE = 'image'
    VIDEO = 'video'