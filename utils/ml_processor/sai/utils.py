from copy import deepcopy
import json
import os
import random
import uuid

import requests
from shared.constants import InferenceParamType
from utils.constants import MLQueryObject
from utils.encryption import Encryptor
from utils.ml_processor.constants import ML_MODEL, MLModel


def predict_sai_output(data):
    if not data:
        return None
    
    # TODO: decouple encryptor from this function
    encryptor = Encryptor()
    sai_key = encryptor.decrypt_json(data['data']['data']["stability_key"])
    input_params = deepcopy(data)
    del input_params["data"]

    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "authorization": f"Bearer {sai_key}",
            "accept": "image/*"
        },
        files={"none": ''},
        data=input_params,
    )

    if response.status_code == 200:
        unique_filename = os.path.join("output", str(uuid.uuid4()) + ".png")
        if not os.path.exists("output"):
            os.makedirs("output")
        with open(unique_filename, 'wb') as file:
            file.write(response.content)
        
        return unique_filename
    else:
        raise Exception(str(response.json()))

def get_closest_aspect_ratio(width, height):
    aspect_ratios = ["16:9","1:1","21:9","2:3","3:2","4:5","5:4","9:16","9:21"]
    ratio = width / height
    closest_ratio = None
    min_difference = float('inf')

    for aspect_ratio in aspect_ratios:
        aspect_width, aspect_height = aspect_ratio.split(":")
        aspect_ratio_value = int(aspect_width) / int(aspect_height)

        difference = abs(ratio - aspect_ratio_value)
        if difference < min_difference:
            min_difference = difference
            closest_ratio = aspect_ratio

    return closest_ratio

def get_model_params_from_query_obj(model: MLModel, query_obj: MLQueryObject):
    if model == ML_MODEL.sd3:
        return {
                "mode": "text-to-image",
                "prompt": query_obj.prompt,
                "negative_prompt": query_obj.negative_prompt,
                "model": "sd3",
                "seed": random_seed(),
                "output_format": "png",
                "data": query_obj.data,
                "aspect_ratio" : get_closest_aspect_ratio(query_obj.width, query_obj.height)
            }
    else:
        return None

def random_seed():
    return random.randint(10**6, 10**8 - 1)