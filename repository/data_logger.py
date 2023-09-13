import json
import streamlit as st
import time
from shared.logging.constants import LoggingPayload, LoggingType
from shared.logging.logging import AppLogger
from utils.data_repo.data_repo import DataRepo

from utils.ml_processor.replicate.constants import ReplicateModel


def log_model_inference(model: ReplicateModel, time_taken, **kwargs):
    kwargs_dict = dict(kwargs)

    # removing object like bufferedreader, image_obj ..
    for key, value in dict(kwargs_dict).items():
        if not isinstance(value, (int, str, list, dict)):
            del kwargs_dict[key]

    data_str = json.dumps(kwargs_dict)
    time_taken = round(time_taken, 2)

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

    # storing the log in db
    data_repo = DataRepo()
    ai_model = data_repo.get_ai_model_from_name(model.name)

    log_data = {
        "project_id" : st.session_state["project_uuid"],
        "model_id" : ai_model.uuid if ai_model else None,
        "input_params" : data_str,
        "output_details" : json.dumps({"model_name": model.name, "version": model.version}),
        "total_inference_time" : time_taken,
    }
    
    data_repo.create_inference_log(**log_data)