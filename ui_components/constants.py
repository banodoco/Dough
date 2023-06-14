from utils.enum import ExtendedEnum


class WorkflowStageType(ExtendedEnum):
    SOURCE = "source"
    STYLED = "styled"
    

class VideoQuality(ExtendedEnum):
    HIGH = "High-Quality"
    PREVIEW = "Preview"
    LOW = "Low"