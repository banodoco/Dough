import os
import shutil
import time
import uuid
from dotenv import dotenv_values

from PIL import Image
from shared.constants import SERVER, AIModelType, GuidanceType, InternalFileType, ServerType
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from shared.constants import AnimationStyleType
from ui_components.common_methods import add_image_variant
from ui_components.models import InternalAppSettingObject, InternalFrameTimingObject, InternalUserObject
from utils.common_utils import copy_sample_assets, create_working_assets, save_or_host_file
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.replicate.constants import REPLICATE_MODEL

ENCRYPTION_KEY = 'J2684nBgNUYa_K0a6oBr5H8MpSRW0EJ52Qmq7jExE-w='

logger = AppLogger()

def project_init():
    data_repo = DataRepo()

    # db initialization takes some time
    # time.sleep(2)
    # create a user if not already present (if dev mode)
    # if this is the local server with no user than create one and related data
    user_count = data_repo.get_total_user_count()
    if SERVER == ServerType.DEVELOPMENT.value and not user_count:
        user_data = {
            "name" : "banodoco_user",
            "email" : "banodoco@tempuser.com",
            "password" : "123",
            "type" : "user"
        }
        user: InternalUserObject = data_repo.create_user(**user_data)
        logger.log(LoggingType.INFO, "new temp user created: " + user.name)

        create_new_user_data(user)
    # creating data for online user
    else:
        app_settings: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()
        if not app_settings:
            online_user = data_repo.get_first_active_user()
            create_new_user_data(online_user)

    # create encryption key if not already present (not applicable in dev mode)
    # env_vars = dotenv_values('.env')
    # desired_key = 'FERNET_KEY'
    # global ENCRYPTION_KEY
    # if desired_key in env_vars:
    #     ENCRYPTION_KEY = env_vars[desired_key].decode()
    # else:
    #     from cryptography.fernet import Fernet

    #     secret_key = Fernet.generate_key()
    #     with open('.env', 'a') as env_file:
    #         env_file.write(f'FERNET_KEY={secret_key.decode()}\n')
        
    #     ENCRYPTION_KEY = secret_key.decode()


# sample data required for starting the project like app settings, project settings
def create_new_user_data(user: InternalUserObject):
    data_repo = DataRepo()
    
    # TODO: disable updating aws or replicate settings in staging or production
    setting_data = {
        "user_id": user.uuid,
        "welcome_state": 0
    }

    app_setting = data_repo.create_app_setting(**setting_data)

    # creating a new project for this user
    project_data = {
        "user_id": user.uuid,
        "name": "my_first_project",
        'width': 704,
        'height': 512
    }
    project = data_repo.create_project(**project_data)

    # create a sample timing frame
    sample_file_location = "sample_assets/frames/selected_sample/3vlb4mr7d95c42i4.png"
    img = Image.open(sample_file_location)
    hosted_url = save_or_host_file(img, sample_file_location)
    file_data = {
        "name": str(uuid.uuid4()),
        "type": InternalFileType.IMAGE.value,
        "project_id": project.uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': sample_file_location})

    source_image = data_repo.create_file(**file_data)

    timing_data = {
        "project_id": project.uuid,
        "frame_time": 0.0,
        "animation_style": AnimationStyleType.INTERPOLATION.value,
        "aux_frame_index": 0,
        "source_image_id": source_image.uuid
    }
    timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)

    add_image_variant(source_image.uuid, timing.uuid)

    # create default ai models
    model_list = create_predefined_models(user)

    # creating a project settings for this
    project_setting_data = {
        "project_id" : project.uuid,
        "input_type" : "video",
        "default_strength": 0.63,
        "extraction_type" : "Extract manually",
        "width" : 512,
        "height" : 512,
        "default_prompt": "an oil painting",
        "default_model_id": model_list[0].uuid,
        "default_negative_prompt" : "",
        "default_guidance_scale" : 7.5,
        "default_seed" : 1234,
        "default_num_inference_steps" : 30,
        "default_stage" : "Source Image",
        "default_custom_model_id_list" : "[]",
        "default_adapter_type" : "N",
        "guidance_type" : GuidanceType.DRAWING.value,
        "default_animation_style" : AnimationStyleType.INTERPOLATION.value,
        "default_low_threshold" : 0,
        "default_high_threshold" : 0
    }

    project_setting = data_repo.create_project_setting(**project_setting_data)

    # TODO: remove this from the hosted version
    create_working_assets(project.uuid)

    

def create_predefined_models(user):
    # create predefined models
    data = [
        {
            "name" : 'stable-diffusion-img2img-v2.1',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.img2img_sd_2_1.version,
            "replicate_url" : REPLICATE_MODEL.img2img_sd_2_1.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
        {
            "name" : 'depth2img',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.jagilley_controlnet_depth2img.version,
            "replicate_url" : REPLICATE_MODEL.jagilley_controlnet_depth2img.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
        {
            "name" : 'pix2pix',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.arielreplicate.version,
            "replicate_url" : REPLICATE_MODEL.arielreplicate.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
        {
            "name" : 'controlnet',
            "user_id" : user.uuid,
            "category" : AIModelType.CONTROLNET.value,
            "keyword" : ""
        },
        {
            "name" : 'Dreambooth',
            "user_id" : user.uuid,
            "category" : AIModelType.DREAMBOOTH.value,
            "keyword" : ""
        },
        {
            "name" : 'LoRA',
            "user_id" : user.uuid,
            "category" : AIModelType.LORA.value,
            "keyword" : ""
        },
        {
            "name" : 'StyleGAN-NADA',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.stylegan_nada.version,
            "replicate_url" : REPLICATE_MODEL.stylegan_nada.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
        {
            "name" : 'real-esrgan-upscaling',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.real_esrgan_upscale.version,
            "replicate_url" : REPLICATE_MODEL.real_esrgan_upscale.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
        {
            "name" : 'controlnet_1_1_x_realistic_vision_v2_0',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.controlnet_1_1_x_realistic_vision_v2_0.version,
            "replicate_url" : REPLICATE_MODEL.controlnet_1_1_x_realistic_vision_v2_0.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
        {
            "name" : 'urpm-v1.3',
            "user_id" : user.uuid,
            "version": REPLICATE_MODEL.urpm.version,
            "replicate_url" : REPLICATE_MODEL.urpm.name,
            "category" : AIModelType.BASE_SD.value,
            "keyword" : ""
        },
    ]

    model_list = []
    data_repo = DataRepo()
    for model in data:
        model_list.append(data_repo.create_ai_model(**model))

    return model_list

    

    