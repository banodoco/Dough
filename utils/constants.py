# streamlit state constants
import json
from shared.constants import AIModelCategory, AIModelType
from utils.enum import ExtendedEnum
from utils.ml_processor.constants import ML_MODEL
import streamlit as st


AUTH_TOKEN = "auth_details"
REFRESH_AUTH_TOKEN = "refresh_auth_details"
RUNNER_PROCESS_NAME = "banodoco_runner"
RUNNER_PROCESS_PORT = 12345


class TomlConfig(ExtendedEnum):
    FILE_HASH = "file_hash"
    NODE_VERSION = "node_version"
    COMFY_VERSION = "comfy"


class ImageStage(ExtendedEnum):
    SOURCE_IMAGE = "Source Image"
    MAIN_VARIANT = "Main Variant"
    NONE = "None"


class T2IModel(ExtendedEnum):
    SDXL = "SDXL"
    SD3 = "SD3"


class AnimateShotMethod(ExtendedEnum):  # remove this and have a common nomenclature throughout
    BATCH_CREATIVE_INTERPOLATION = "Batch Creative Interpolation"
    DYNAMICRAFTER_INTERPOLATION = "2-Image Realistic Interpolation (beta)"


# single template for passing query params
# TODO: remove the named params and pass everything through the data object (as different models require very different inputs)
class MLQueryObject:
    def __init__(
        self,
        timing_uuid,
        guidance_scale,
        seed,
        num_inference_steps,
        strength,
        adapter_type=None,
        prompt="",
        negative_prompt="",
        height=512,
        width=512,
        low_threshold=100,  # update these default values
        high_threshold=200,
        file_data={},  # {'file_name': {'uuid': file_uuid, 'dest': file_dest}}
        relation_data={},  # json.dumps([{"type": "file", "id": video_file.uuid, "transformation_type": "upscale"}])
        **kwargs,
    ):
        self.timing_uuid = timing_uuid
        self.prompt = prompt
        self.strength = strength
        self.height = height
        self.width = width
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.adapter_type = adapter_type
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.data = kwargs
        self.relation_data = relation_data
        self.file_data = file_data

        self._validate_params()

    def _validate_params(self):
        if not (self.prompt or len(self.file_data.keys())):
            st.error("Prompt or image is required to run the model")
            raise Exception("Prompt or image is required to run the model")

    @property
    def file_list(self):
        from utils.data_repo.data_repo import DataRepo

        data_repo = DataRepo()

        file_uuid_list = []
        for _, v in self.file_data.items():
            file_uuid_list.append(v["uuid"])

        return data_repo.get_all_file_list(uuid__in=file_uuid_list)[0]

    def to_json(self):
        return json.dumps(self.__dict__)


# NOTE: old code, not is use
ML_MODEL_LIST = [
    {
        "name": "stable-diffusion-img2img-v2.1",
        "version": ML_MODEL.img2img_sd_2_1.version,
        "replicate_url": ML_MODEL.img2img_sd_2_1.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": False,
    },
    {
        "name": "depth2img",
        "version": ML_MODEL.jagilley_controlnet_depth2img.version,
        "replicate_url": ML_MODEL.jagilley_controlnet_depth2img.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": False,
    },
    {
        "name": "pix2pix",
        "version": ML_MODEL.arielreplicate.version,
        "replicate_url": ML_MODEL.arielreplicate.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": False,
    },
    {
        "name": "controlnet",
        "category": AIModelCategory.CONTROLNET.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "Dreambooth",
        "category": AIModelCategory.DREAMBOOTH.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "LoRA",
        "category": AIModelCategory.LORA.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "StyleGAN-NADA",
        "version": ML_MODEL.stylegan_nada.version,
        "replicate_url": ML_MODEL.stylegan_nada.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": False,
    },
    {
        "name": "real-esrgan-upscaling",
        "version": ML_MODEL.real_esrgan_upscale.version,
        "replicate_url": ML_MODEL.real_esrgan_upscale.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": False,
    },
    {
        "name": "controlnet_1_1_x_realistic_vision_v2_0",
        "version": ML_MODEL.controlnet_1_1_x_realistic_vision_v2_0.version,
        "replicate_url": ML_MODEL.controlnet_1_1_x_realistic_vision_v2_0.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": False,
    },
    {
        "name": "urpm-v1.3",
        "version": ML_MODEL.urpm.version,
        "replicate_url": ML_MODEL.urpm.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "stable_diffusion_xl",
        "version": ML_MODEL.sdxl.version,
        "replicate_url": ML_MODEL.sdxl.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "realistic_vision_5",
        "version": ML_MODEL.realistic_vision_v5.version,
        "replicate_url": ML_MODEL.realistic_vision_v5.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.TXT2IMG.value]),
        "enabled": True,
    },
    {
        "name": "deliberate_v3",
        "version": ML_MODEL.deliberate_v3.version,
        "replicate_url": ML_MODEL.deliberate_v3.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "dreamshaper_v7",
        "version": ML_MODEL.dreamshaper_v7.version,
        "replicate_url": ML_MODEL.dreamshaper_v7.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "epic_realism_v5",
        "version": ML_MODEL.epicrealism_v5.version,
        "replicate_url": ML_MODEL.epicrealism_v5.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "sdxl_controlnet",
        "version": ML_MODEL.sdxl_controlnet.version,
        "replicate_url": ML_MODEL.sdxl_controlnet.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "sdxl_controlnet_openpose",
        "version": ML_MODEL.sdxl_controlnet_openpose.version,
        "replicate_url": ML_MODEL.sdxl_controlnet_openpose.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
    {
        "name": "realistic_vision_img2img",
        "version": ML_MODEL.realistic_vision_v5_img2img.version,
        "replicate_url": ML_MODEL.realistic_vision_v5_img2img.name,
        "category": AIModelCategory.BASE_SD.value,
        "keyword": "",
        "model_type": json.dumps([AIModelType.IMG2IMG.value]),
        "enabled": True,
    },
]
