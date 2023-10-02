import asyncio
import io
import time
from shared.file_upload.s3 import upload_file
from utils.common_utils import get_current_user_uuid
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import MachineLearningProcessor
import replicate
import os
import requests as r
import json
import zipfile
from PIL import Image

from utils.ml_processor.replicate.constants import REPLICATE_MODEL, ReplicateModel
from repository.data_logger import log_model_inference
import utils.local_storage.local_storage as local_storage
from utils.ml_processor.replicate.utils import check_user_credits, check_user_credits_async, get_model_params_from_query_obj


class ReplicateProcessor(MachineLearningProcessor):
    def __init__(self):
        data_repo = DataRepo()
        self.app_settings = data_repo.get_app_secrets_from_user_uuid(uuid=get_current_user_uuid())

        self.logger = None
        try:
            os.environ["REPLICATE_API_TOKEN"] = self.app_settings['replicate_key']
        except Exception as e:
            print('no replicate key found')
        self._set_urls()
        super().__init__()

    def _set_urls(self):
        self.dreambooth_training_url = "https://dreambooth-api-experimental.replicate.com/v1/trainings"
        self.training_data_upload_url = "https://dreambooth-api-experimental.replicate.com/v1/upload/data.zip"
        self.model_version_url = "https://api.replicate.com/v1/models"

    def _update_usage_credits(self, time_taken):
        data_repo = DataRepo()
        cost = round((time_taken / 60) * 0.17, 3)
        data_repo.update_usage_credits(-cost)

    def get_model(self, input_model: ReplicateModel):
        model = replicate.models.get(input_model.name)
        model_version = model.versions.get(input_model.version) if input_model.version else model
        return model_version
    
    def get_model_by_name(self, model_name, model_version=None):
        model = replicate.models.get(model_name)
        model_version = model.versions.get(model_version) if model_version else model
        return model_version
    
    # it converts the standardized query_obj into params required by replicate
    def predict_model_output_standardized(self, model: ReplicateModel, query_obj: MLQueryObject):
        params = get_model_params_from_query_obj(model, query_obj)
        params['query_dict'] = query_obj.to_json()
        return self.predict_model_output(model, **params)
    
    @check_user_credits
    def predict_model_output(self, model: ReplicateModel, **kwargs):
        model_version = self.get_model(model)
        start_time = time.time()
        output = model_version.predict(**kwargs)
        end_time = time.time()
        log = log_model_inference(model, end_time - start_time, **kwargs)
        self._update_usage_credits(end_time - start_time)

        return [output[-1]], log
    
    @check_user_credits
    def predict_model_output_async(self, model: ReplicateModel, **kwargs):
        res = asyncio.run(self._multi_async_prediction(model, **kwargs))

        output_list = []
        for (output, time_taken) in  res:
            log = log_model_inference(model, time_taken, **kwargs)
            self._update_usage_credits(time_taken)
            output_list.append((output, log))

        return output_list
    
    async def _multi_async_prediction(self, model: ReplicateModel, **kwargs):
        variant_count = kwargs['variant_count'] if ('variant_count' in kwargs and kwargs['variant_count']) else 1
        res = await asyncio.gather(*[self._async_model_prediction(model, **kwargs) for _ in range(variant_count)])
        return res
    
    async def _async_model_prediction(self, model: ReplicateModel, **kwargs):
        model_version = self.get_model(model)
        start_time = time.time()
        output = await asyncio.to_thread(model_version.predict, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        return output, time_taken

    @check_user_credits
    def inpainting(self, video_name, input_image, prompt, negative_prompt):
        model = self.get_model(REPLICATE_MODEL.andreas_sd_inpainting)
        
        mask = "mask.png"
        mask = upload_file("mask.png", self.app_settings['aws_access_key'], self.app_settings['aws_secret_key'])
            
        if not input_image.startswith("http"):        
            input_image = open(input_image, "rb")

        start_time = time.time()
        output = model.predict(mask=mask, image=input_image,prompt=prompt, invert_mask=True, negative_prompt=negative_prompt,num_inference_steps=25)    
        end_time = time.time()
        log = log_model_inference(model, end_time - start_time, prompt=prompt, invert_mask=True, negative_prompt=negative_prompt,num_inference_steps=25)
        self._update_usage_credits(end_time - start_time)

        return output[0], log
    
    # TODO: separate image compression from this function
    @check_user_credits
    def upload_training_data(self, images_list):
        # compressing images in zip file
        for i in range(len(images_list)):
            images_list[i] = 'videos/training_data/' + images_list[i]

        with zipfile.ZipFile('images.zip', 'w') as zip:
            for image in images_list:
                zip.write(image, arcname=os.path.basename(image))

        headers = {
            "Authorization": "Token " + os.environ.get("REPLICATE_API_TOKEN"),
            "Content-Type": "application/zip"
        }
        response = r.post(self.training_data_upload_url, headers=headers)
        if response.status_code != 200:
            raise Exception(str(response.content))
        upload_url = response.json()["upload_url"]  # this is where data will be uploaded
        serving_url = response.json()["serving_url"]    # this is where the data will be available
        with open("images.zip", 'rb') as f:
            r.put(upload_url, data=f, headers=headers)
        
        os.remove('images.zip')
        return serving_url
    
    # for uploading temp files
    @check_user_credits
    def upload_training_data(self, images_list):
        with zipfile.ZipFile('images.zip', 'w') as zip_file:
            for i, image in enumerate(images_list):
                image = Image.open(image)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes.seek(0)

                filename = f'image_{i}.png'

                zip_file.writestr(filename, image_bytes.read())

        headers = {
            "Authorization": "Token " + os.environ.get("REPLICATE_API_TOKEN"),
            "Content-Type": "application/zip"
        }
        response = r.post(self.training_data_upload_url, headers=headers)
        if response.status_code != 200:
            raise Exception(str(response.content))
        upload_url = response.json()["upload_url"]  # this is where data will be uploaded
        serving_url = response.json()["serving_url"]    # this is where the data will be available
        with open("images.zip", 'rb') as f:
            r.put(upload_url, data=f, headers=headers)
        
        os.remove('images.zip')
        return serving_url
    
    # TODO: figure how to resolve model location setting, right now it's hardcoded to peter942/modnet
    @check_user_credits
    def dreambooth_training(self, training_file_url, instance_prompt, \
                            class_prompt, max_train_steps, model_name, controller_type, image_len, replicate_user):
        if controller_type == "normal":
            template_version = "b65d36e378a01ef81d81ba49be7deb127e9bb8b74a28af3aa0eaca16b9bcd0eb"
        elif controller_type == "canny":
            template_version = "3c60cbfce253b1d82fea02c7692d13c1e96b36a22da784470fcbedc603a1ed4b"
        elif controller_type == "hed":
            template_version = "bef0803be223ecb38361097771dbea7cd166514996494123db27907da53d75cd"
        elif controller_type == "scribble":
            template_version = "346b487d77a0bdd150c4bbb8f162f7cd4a4491bca5f309105e078556d0789f11"
        elif controller_type == "seg":
            template_version = "a0266713f8c30b35a3f4fc8212fc9450cecea61e4181af63cfb54e5a152ecb24"
        elif controller_type == "openpose":
            template_version = "141b8753e2973933441880e325fd21404923d0877014c9f8903add05ff530e52"
        elif controller_type == "depth":
            template_version = "6cf8fc430894121f2f91867978780011e6859b6956b499b43273afc25ed21121"
        elif controller_type == "mlsd":
            template_version == "04982e9aa6d3998c2a2490f92e7ccfab2dbd93f5be9423cdf0405c7b86339022"

        headers = {
            "Authorization": "Token " + os.environ.get("REPLICATE_API_TOKEN"),
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "instance_prompt": instance_prompt,
                "class_prompt": class_prompt,
                "instance_data": training_file_url,
                "max_train_steps": max_train_steps
            },
            "model": replicate_user + "/" + str(model_name),
            "trainer_version": "cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa",
            "template_version": template_version,
            "webhook_completed": "https://example.com/dreambooth-webhook"
        }

        response = r.post(self.dreambooth_training_url, headers=headers, data=json.dumps(payload))
        response = (response.json())

        # TODO: currently approximating total credit cost of training based on image len, will fix this in the future
        time_taken = image_len * 3 * 60 # per image 3 mins
        self._update_usage_credits(time_taken)

        return response
    
    def get_model_version_from_id(self, model_id):
        api_key = os.environ.get("REPLICATE_API_TOKEN")
        headers = {"Authorization": f"Token {api_key}"}
        url = f"{self.model_version_url}/{model_id}/versions"
        response = r.get(url, headers=headers)
        # version = (response.json()["version"])

        version = (response.json())['results'][0]['id']
        return version