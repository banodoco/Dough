from dataclasses import dataclass


@dataclass
class MotionModuleCheckpoint:
    name: str


# make sure to have unique names (streamlit limitation)
class AnimateDiffCheckpoint:
    mm_v15 = MotionModuleCheckpoint(name="mm_sd_v15.ckpt")
    mm_v14 = MotionModuleCheckpoint(name="mm_sd_v14.ckpt")

    @staticmethod
    def get_name_list():
        checkpoint_names = [
            getattr(AnimateDiffCheckpoint, attr).name
            for attr in dir(AnimateDiffCheckpoint)
            if not callable(getattr(AnimateDiffCheckpoint, attr))
            and not attr.startswith("__")
            and isinstance(getattr(AnimateDiffCheckpoint, attr), MotionModuleCheckpoint)
        ]
        return checkpoint_names

    @staticmethod
    def get_model_from_name(name):
        checkpoint_list = [
            getattr(AnimateDiffCheckpoint, attr)
            for attr in dir(AnimateDiffCheckpoint)
            if not callable(getattr(AnimateDiffCheckpoint, attr))
            and not attr.startswith("__")
            and isinstance(getattr(AnimateDiffCheckpoint, attr), MotionModuleCheckpoint)
        ]

        for ckpt in checkpoint_list:
            if ckpt.name == name:
                return ckpt

        return None
