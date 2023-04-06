from dataclasses import dataclass
from utils.enum import ExtendedEnum


@dataclass
class LoggingPayload:
    message: str
    data: dict = {}
    
class LoggingType(ExtendedEnum):
    INFO = 'info'
    INFERENCE_CALL = 'inference_call'
    INFERENCE_RESULT = 'inference_result'
    ERROR = 'error'
    DEBUG = 'debug'

class LoggingMode(ExtendedEnum):
    OFFLINE = 'offline'
    ONLINE = 'online'