from distutils.file_util import copy_file
import json
import os
from shared.constants import InferenceParamType
from shared.logging.logging import AppLogger
from ui_components.methods.data_logger import log_model_inference
from ui_components.methods.file_methods import copy_local_file, normalize_size_internal_file_obj
from utils.common_utils import padded_integer
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.comfy_data_transform import get_file_list_from_query_obj, get_model_workflow_from_query
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow, MLModel
from utils.ml_processor.gpu.utils import predict_gpu_output, setup_comfy_runner
from utils.ml_processor.ml_interface import MachineLearningProcessor
import time


# NOTE: add credit management methods such update_usage_credits, check_usage_credits etc.. for hosting
class GPUProcessor(MachineLearningProcessor):
    def __init__(self):
        setup_comfy_runner()
        data_repo = DataRepo()
        self.app_settings = data_repo.get_app_secrets_from_user_uuid()
        super().__init__()

    def predict_model_output_standardized(self, model: MLModel, query_obj: MLQueryObject, queue_inference=False):
        data_repo = DataRepo()
        workflow_json, output_node_ids, extra_model_list = get_model_workflow_from_query(model, query_obj)
        file_uuid_list = []

        file_uuid_list = get_file_list_from_query_obj(query_obj)
        file_list = data_repo.get_image_list_from_uuid_list(file_uuid_list)

        models_using_sdxl = [
                ComfyWorkflow.SDXL.value, 
                ComfyWorkflow.SDXL_IMG2IMG.value,
                ComfyWorkflow.SDXL_CONTROLNET.value, 
                ComfyWorkflow.SDXL_INPAINTING.value,
                ComfyWorkflow.IP_ADAPTER_FACE.value,
                ComfyWorkflow.IP_ADAPTER_FACE_PLUS.value,
                ComfyWorkflow.IP_ADAPTER_PLUS.value
            ]

        # maps old_file_name : new_resized_file_name
        new_file_map = {}
        if model.display_name() in models_using_sdxl:
            res = []
            for file in file_list:
                new_width, new_height = 1024 if query_obj.width == 512 else 768, 1024 if query_obj.height == 512 else 768
                # although the new_file created using create_new_file has the same location as the original file, it is 
                # scaled to the original resolution after inference save (so resize has no effect)
                new_file = normalize_size_internal_file_obj(file, dim=[new_width, new_height], create_new_file=True)
                res.append(new_file)
                new_file_map[file.filename] = new_file.filename

            file_list = res

        file_path_list = []
        for idx, file in enumerate(file_list):
            _, filename = os.path.split(file.local_path)
            new_filename = f"{padded_integer(idx+1)}_" + filename
            copy_local_file(file.local_path, "videos/temp/", new_filename)
            file_path_list.append("videos/temp/" + new_filename)

        # replacing old files with resized files
        # if len(new_file_map.keys()):
        #     workflow_json = json.loads(workflow_json)
        #     for node in workflow_json:
        #         if "inputs" in workflow_json[node]:
        #             for k, v in workflow_json[node]["inputs"].items():
        #                 if isinstance(v, str) and v in new_file_map:
        #                     workflow_json[node]["inputs"][k] = new_file_map[v]

        #     workflow_json = json.dumps(workflow_json)

        data = {
            "workflow_input": workflow_json,
            "file_path_list": file_path_list,
            "output_node_ids": output_node_ids,
            "extra_model_list": extra_model_list
        }

        params = {
            "prompt": query_obj.prompt,     # hackish sol
            InferenceParamType.QUERY_DICT.value: query_obj.to_json(),
            InferenceParamType.GPU_INFERENCE.value: json.dumps(data)
        }
        return self.predict_model_output(model, **params) if not queue_inference else self.queue_prediction(model, **params)

    def predict_model_output(self, replicate_model: MLModel, **kwargs):
        queue_inference = kwargs.get('queue_inference', False)
        if queue_inference:
            return self.queue_prediction(replicate_model, **kwargs)
        
        data = kwargs.get(InferenceParamType.GPU_INFERENCE.value, None)
        data = json.loads(data)
        start_time = time.time()
        output = predict_gpu_output(data['workflow_input'], data['file_path_list'], data['output_node_ids'])
        end_time = time.time()

        log = log_model_inference(replicate_model, end_time - start_time, **kwargs)
        return output, log

    def queue_prediction(self, replicate_model, **kwargs):
        log = log_model_inference(replicate_model, None, **kwargs)
        return None, log

    def upload_training_data(self, zip_file_name, delete_after_upload=False):
        # TODO: fix for online hosting
        # return the local file path as it is
        return zip_file_name