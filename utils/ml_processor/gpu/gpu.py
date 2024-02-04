from shared.constants import InferenceParamType
from shared.logging.logging import AppLogger
from ui_components.methods.data_logger import log_model_inference
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import MLModel
from utils.ml_processor.gpu.utils import get_model_workflow_from_query, predict_gpu_output
from utils.ml_processor.ml_interface import MachineLearningProcessor
import time


# NOTE: add credit management methods such update_usage_credits, check_usage_credits etc..
class GPUProcessor(MachineLearningProcessor):
    def __init__(self):
        data_repo = DataRepo()
        self.app_settings = data_repo.get_app_secrets_from_user_uuid()
        self.logger = AppLogger()
        super().__init__()

    def predict_model_output_standardized(self, model: MLModel, query_obj: MLQueryObject, queue_inference=False):
        model_workflow = get_model_workflow_from_query(model, query_obj)    # stringified json
        params = {
            InferenceParamType.QUERY_DICT.value: query_obj.to_json(),
            InferenceParamType.GPU_INFERENCE.value: model_workflow
        }
        return self.predict_model_output(model, **params) if not queue_inference else self.queue_prediction(model, **params)

    def predict_model_output(self, replicate_model: MLModel, **kwargs):
        queue_inference = kwargs.get('queue_inference', False)
        if queue_inference:
            return self.queue_prediction(replicate_model, **kwargs)
        
        start_time = time.time()
        output = predict_gpu_output(kwargs.get(InferenceParamType.GPU_INFERENCE.value, None))
        end_time = time.time()

        # TODO: lg_complete update this method to take in the current params
        log = log_model_inference(replicate_model, end_time - start_time, **kwargs)
        return output, log

    def queue_prediction(self, replicate_model, **kwargs):
        log = log_model_inference(replicate_model, None, **kwargs)
        return None, log

    def get_model_version_from_id(self, model_id):
        # TODO: lg_complete check exactly where this is being used and why
        pass

    def upload_training_data(self, *args, **kwargs):
        # TODO: lg_complete check exactly where this is being used and why 
        # most probably make a local copy and return it's path
        pass