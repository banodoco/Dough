import time
from shared.constants import InferenceParamType
from ui_components.methods.data_logger import log_model_inference
from utils.constants import MLQueryObject
from utils.ml_processor.constants import MLModel
from utils.ml_processor.ml_interface import MachineLearningProcessor
from utils.ml_processor.sai.utils import get_model_params_from_query_obj, predict_sai_output


# rn only used for sd3 and doesn't have all methods of MachineLearningProcessor
class StabilityProcessor(MachineLearningProcessor):
    def __init__(self):
        pass

    def predict_model_output_standardized(
        self, model: MLModel, query_obj: MLQueryObject, queue_inference=False, backlog=False
    ):
        params = get_model_params_from_query_obj(model, query_obj)

        if params:
            new_params = {}
            new_params[InferenceParamType.QUERY_DICT.value] = params
            new_params[InferenceParamType.SAI_INFERENCE.value] = params
            return (
                self.predict_model_output(model, **new_params)
                if not queue_inference
                else self.queue_prediction(model, **new_params)
            )  # add backlog later

    def predict_model_output(self, model: MLModel, **kwargs):
        queue_inference = kwargs.get("queue_inference", False)
        if queue_inference:
            del kwargs["queue_inference"]
            return self.queue_prediction(model, **kwargs)

        start_time = time.time()
        output = predict_sai_output(kwargs.get(InferenceParamType.SAI_INFERENCE.value, None))
        end_time = time.time()

        if "model" in kwargs:
            kwargs["inf_model"] = kwargs["model"]
            del kwargs["model"]

        log = log_model_inference(model, end_time - start_time, **kwargs)
        # TODO: update usage credits in the api mode
        # self.update_usage_credits(end_time - start_time)

        return output, log

    def queue_prediction(self, model, **kwargs):
        log = log_model_inference(model, None, **kwargs)
        return None, log
