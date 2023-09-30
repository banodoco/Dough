# streamlit state constants
import json
from shared.constants import AIModelCategory, AIModelType
from utils.enum import ExtendedEnum
from utils.ml_processor.replicate.constants import REPLICATE_MODEL
import streamlit as st


LOGGED_USER = 'logged_user'
AUTH_TOKEN = 'auth_details'

class ImageStage(ExtendedEnum):
    SOURCE_IMAGE = 'Source Image'
    MAIN_VARIANT = 'Main Variant'
    NONE = 'None'


# single template for passing query params
class MLQueryObject:
    def __init__(
            self,
            timing_uuid,
            model_name,
            prompt,
            strength,
            negative_prompt,
            guidance_scale,
            seed,
            num_inference_steps,
            adapter_type,
            height=512,
            width=512,
            low_threshold=100,  # update these default values
            high_threshold=200,
            image_uuid=None,
            mask_uuid=None,
            **kwargs
        ):
        self.timing_uuid = timing_uuid
        self.model_name = model_name
        self.prompt = prompt
        self.image_uuid = image_uuid
        self.mask_uuid = mask_uuid
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

        self._validate_params()
    
    def _validate_params(self):
        if not (self.prompt or self.image_uuid):
            st.error("Prompt or image is required to run the model")
            raise Exception("Prompt or image is required to run the model")
        
    def to_json(self):
        return json.dumps(self.__dict__)

ML_MODEL_LIST = [
        {
            "name" : 'stable-diffusion-img2img-v2.1',
            "version": REPLICATE_MODEL.img2img_sd_2_1.version,
            "replicate_url" : REPLICATE_MODEL.img2img_sd_2_1.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": False
        },
        {
            "name" : 'depth2img',
            "version": REPLICATE_MODEL.jagilley_controlnet_depth2img.version,
            "replicate_url" : REPLICATE_MODEL.jagilley_controlnet_depth2img.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": False
        },
        {
            "name" : 'pix2pix',
            "version": REPLICATE_MODEL.arielreplicate.version,
            "replicate_url" : REPLICATE_MODEL.arielreplicate.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": False
        },
        {
            "name" : 'controlnet',
            "category" : AIModelCategory.CONTROLNET.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name" : 'Dreambooth',
            "category" : AIModelCategory.DREAMBOOTH.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name" : 'LoRA',
            "category" : AIModelCategory.LORA.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name" : 'StyleGAN-NADA',
            "version": REPLICATE_MODEL.stylegan_nada.version,
            "replicate_url" : REPLICATE_MODEL.stylegan_nada.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": False
        },
        {
            "name" : 'real-esrgan-upscaling',
            "version": REPLICATE_MODEL.real_esrgan_upscale.version,
            "replicate_url" : REPLICATE_MODEL.real_esrgan_upscale.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name" : 'controlnet_1_1_x_realistic_vision_v2_0',
            "version": REPLICATE_MODEL.controlnet_1_1_x_realistic_vision_v2_0.version,
            "replicate_url" : REPLICATE_MODEL.controlnet_1_1_x_realistic_vision_v2_0.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name" : 'urpm-v1.3',
            "version": REPLICATE_MODEL.urpm.version,
            "replicate_url" : REPLICATE_MODEL.urpm.name,
            "category" : AIModelCategory.BASE_SD.value,
            "keyword" : "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name": "stable_diffusion_xl",
            "version": REPLICATE_MODEL.sdxl.version,
            "replicate_url": REPLICATE_MODEL.sdxl.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name": "realistic_vision_5",
            "version": REPLICATE_MODEL.realistic_vision_v5.version,
            "replicate_url": REPLICATE_MODEL.realistic_vision_v5.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.TXT2IMG.value]),
            "enabled": True
        },
        {
            "name": "deliberate_v3",
            "version": REPLICATE_MODEL.deliberate_v3.version,
            "replicate_url": REPLICATE_MODEL.deliberate_v3.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name": "dreamshaper_v7",
            "version": REPLICATE_MODEL.dreamshaper_v7.version,
            "replicate_url": REPLICATE_MODEL.dreamshaper_v7.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name": "epic_realism_v5",
            "version": REPLICATE_MODEL.epicrealism_v5.version,
            "replicate_url": REPLICATE_MODEL.epicrealism_v5.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.TXT2IMG.value, AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name": "sdxl_controlnet",
            "version": REPLICATE_MODEL.sdxl_controlnet.version,
            "replicate_url": REPLICATE_MODEL.sdxl_controlnet.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        },
        {
            "name": "realistic_vision_img2img",
            "version": REPLICATE_MODEL.realistic_vision_v5_img2img.version,
            "replicate_url": REPLICATE_MODEL.realistic_vision_v5_img2img.name,
            "category": AIModelCategory.BASE_SD.value,
            "keyword": "",
            "model_type": json.dumps([AIModelType.IMG2IMG.value]),
            "enabled": True
        }
    ]