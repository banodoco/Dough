from utils.enum import ExtendedEnum


class WorkflowStageType(ExtendedEnum):
    SOURCE = "source"
    STYLED = "styled"

class AnimationStyleType(ExtendedEnum):
    INTERPOLATION = "Interpolation"
    DIRECT_MORPHING = "Direct Morphing"

class VideoQuality(ExtendedEnum):
    HIGH = "High-Quality"
    PREVIEW = "Preview"
    LOW = "Low"