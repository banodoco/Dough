from utils.common_utils import user_credits_available
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.replicate.constants import REPLICATE_MODEL


def check_user_credits(method):
    def wrapper(self, *args, **kwargs):
        if user_credits_available():
            res = method(self, *args, **kwargs)
            return res
        else:
            raise RuntimeError("Insufficient credits. Please recharge")
    
    return wrapper

def check_user_credits_async(method):
    async def wrapper(self, *args, **kwargs):
        if user_credits_available():
            res = await method(self, *args, **kwargs)
            return res
        else:
            raise RuntimeError("Insufficient credits. Please recharge")
    
    return wrapper

# TODO: add data validation (like prompt can't be empty...)
def get_model_params_from_query_obj(model,  query_obj: MLQueryObject):
    data_repo = DataRepo()

    input_image, mask = None, None
    if query_obj.image_uuid:
        image = data_repo.get_file_from_uuid(query_obj.image_uuid)
        if image:
            input_image = image.location
            if not input_image.startswith('http'):
                input_image = open(input_image, 'rb')

    if query_obj.mask_uuid:
        mask = data_repo.get_file_from_uuid(query_obj.mask_uuid)
        if mask:
            mask = mask.location
            if not mask.startswith('http'):
                mask = open(mask, 'rb')

    if model == REPLICATE_MODEL.img2img_sd_2_1:
        data = {
            "prompt_strength" : query_obj.strength,
            "prompt" : query_obj.prompt,
            "negative_prompt" : query_obj.negative_prompt,
            "width" : query_obj.width,
            "height" : query_obj.height,
            "guidance_scale" : query_obj.guidance_scale,
            "seed" : query_obj.seed,
            "num_inference_steps" : query_obj.num_inteference_steps
        }

        if input_image:
            data['image'] = input_image

    elif model == REPLICATE_MODEL.real_esrgan_upscale:
        data = {
            "image": input_image,
            "upscale": query_obj.data.get('upscale', 2),
        }
    elif model == REPLICATE_MODEL.stylegan_nada:
        data = {
            "input": input_image,
            "output_style": query_obj.prompt
        }
    elif model == REPLICATE_MODEL.sdxl:
        data = {
            "prompt" : query_obj.prompt,
            "negative_prompt" : query_obj.negative_prompt,
            "width" : query_obj.width,
            "height" : query_obj.height,
            "mask": mask
        }

        if input_image:
            data['image'] = input_image
            
    elif model == REPLICATE_MODEL.jagilley_controlnet_depth2img:
        data = {
            "prompt_strength" : query_obj.strength,
            "prompt" : query_obj.prompt,
            "negative_prompt" : query_obj.negative_prompt,
            "num_inference_steps" : query_obj.num_inference_steps,
            "guidance_scale" : query_obj.guidance_scale
        }

        if input_image:
            data['input_image'] = input_image

    elif model == REPLICATE_MODEL.arielreplicate:
        data = {
            "instruction_text" : query_obj.prompt,
            "seed" : query_obj.seed, 
            "cfg_image" : query_obj.data.get("cfg", 1.2), 
            "cfg_text" : query_obj.guidance_scale, 
            "resolution" : 704
        }

        if input_image:
            data['input_image'] = input_image

    elif model  == REPLICATE_MODEL.urpm:
        data = {
            'prompt': query_obj.prompt,
            'negative_prompt': query_obj.negative_prompt,
            'strength': query_obj.strength,
            'guidance_scale': query_obj.guidance_scale,
            'num_inference_steps': query_obj.num_inference_steps,
            'upscale': 1,
            'seed': query_obj.seed,
        }

        if input_image:
            data['image'] = input_image

    elif model == REPLICATE_MODEL.controlnet_1_1_x_realistic_vision_v2_0:
        data = {
            'prompt': query_obj.prompt,
            'ddim_steps': query_obj.num_inference_steps,
            'strength': query_obj.strength,
            'scale': query_obj.guidance_scale,
            'seed': query_obj.seed
        }

        if input_image:
            data['image'] = input_image

    elif model == REPLICATE_MODEL.realistic_vision_v5:
        if not (query_obj.guidance_scale >= 3.5 and query_obj.guidance_scale <= 7.0):
            raise ValueError("Guidance scale must be between 3.5 and 7.0")

        data = {
            'prompt': query_obj.prompt,
            'negative_prompt': query_obj.negative_prompt,
            'guidance': query_obj.guidance_scale,
            'width': query_obj.width,
            'height': query_obj.height,
            'steps': query_obj.num_inference_steps,
            'seed': query_obj.seed
        }
    elif model == REPLICATE_MODEL.deliberate_v3 or model == REPLICATE_MODEL.dreamshaper_v7 or model == REPLICATE_MODEL.epicrealism_v5:
        data = {
            'prompt': query_obj.prompt,
            'negative_prompt': query_obj.negative_prompt,
            'width': query_obj.width,
            'height': query_obj.height,
            'prompt_strength': query_obj.strength,
            'guidance_scale': query_obj.guidance_scale,
            'num_inference_steps': query_obj.num_inference_steps,
            'safety_checker': False
        }

        if input_image:
            data['image'] = input_image
        if mask:
            data['mask'] = mask

    elif model == REPLICATE_MODEL.sdxl_controlnet:
        data = {
            'prompt': query_obj.prompt,
            'negative_prompt': query_obj.negative_prompt,
            'num_inference_steps': query_obj.num_inference_steps,
            'condition_scale': query_obj.data.get('condition_scale', 0.5),
        }

        if input_image:
            data['image'] = input_image

    elif model == REPLICATE_MODEL.realistic_vision_v5_img2img:
        data = {
            'prompt': query_obj.prompt,
            'negative_prompt': query_obj.negative_prompt,
            'image': input_image,
            'steps': query_obj.num_inference_steps,
            'strength': query_obj.strength
        }

        if input_image:
            data['image'] = input_image

    else:
        data = query_obj.to_json()

    return data