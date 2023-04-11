import json
import time
from repository.local_repo.csv_repo import log_inference_data_in_csv
from utils.logging.constants import LoggingPayload, LoggingType
from utils.logging.logging import AppLogger
from utils.ml_processor.replicate.constants import ReplicateModel


def log_model_inference(model: ReplicateModel, time_taken, **kwargs):
    kwargs_dict = dict(kwargs)

    # removing object like bufferedreader, image_obj ..
    for key, value in dict(kwargs_dict).items():
        if not isinstance(value, (int, str, list, dict)):
            del kwargs_dict[key]

    data_str = json.dumps(kwargs_dict)

    data = {
        'model_name': model.name,
        'model_version': model.version,
        'total_inference_time': time_taken,
        'input_params': data_str,
        'created_on': int(time.time())
    }

    system_logger = AppLogger()
    logging_payload = LoggingPayload(message="logging inference data", data=data)

    # logging in console
    system_logger.log(LoggingType.INFERENCE_CALL, logging_payload)

    # logging data
    log_inference_data_in_csv(logging_payload.data)