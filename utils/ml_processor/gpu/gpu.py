import json
from shared.constants import InferenceParamType
from shared.logging.logging import AppLogger
from ui_components.methods.data_logger import log_model_inference
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.comfy_data_transform import get_model_workflow_from_query
from utils.ml_processor.constants import ML_MODEL, MLModel
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
        workflow_json, output_node_ids = get_model_workflow_from_query(model, query_obj)
        file_uuid_list = []

        if query_obj.image_uuid:
            file_uuid_list.append(query_obj.image_uuid)
        
        for k, v in query_obj.data.get('data', {}).items():
            if k.startswith("file_"):
                file_uuid_list.append(v)

        file_list = data_repo.get_image_list_from_uuid_list(file_uuid_list)
        file_path_list = [f.location for f in file_list]
        
        data = {
            "workflow_input": workflow_json,
            "file_path_list": file_path_list,
            "output_node_ids": output_node_ids
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