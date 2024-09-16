from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
from urllib.parse import urlparse

import requests
from shared.constants import SERVER_URL, InferenceParamType
from ui_components.methods.data_logger import log_model_inference
from utils.common_utils import get_auth_token
from utils.constants import MLQueryObject
from utils.ml_processor.comfy_data_transform import get_file_path_list, get_model_workflow_from_query
from utils.ml_processor.constants import MLModel
from utils.ml_processor.ml_interface import MachineLearningProcessor
import time


class APIProcessor(MachineLearningProcessor):
    def __init__(self):
        self.auth_token = None
        super().__init__()

    def predict_model_output_standardized(
        self,
        model: MLModel,
        query_obj: MLQueryObject,
        queue_inference=False,
        backlog=False,
    ):

        token, _ = get_auth_token(validate_through_db=True)
        if not token:
            print("Unable to queue generation, no auth token found")
            return None

        self.auth_token = token

        (
            workflow_type,
            workflow_json,
            output_node_ids,
            extra_model_list,
            ignore_list,
        ) = get_model_workflow_from_query(model, query_obj)

        file_path_list = get_file_path_list(model, query_obj)
        files_to_upload = []
        for fp in file_path_list:
            if isinstance(fp, str):
                files_to_upload.append(fp)
            else:
                files_to_upload.append(fp["filepath"])

        uploaded_dict = self.multi_file_upload(files_to_upload)
        if len(uploaded_dict) != len(files_to_upload):
            raise Exception("unable to upload input files")

        new_file_path_list = []
        for fp in file_path_list:
            if isinstance(fp, str):
                new_file_path_list.append(uploaded_dict[fp])
            else:
                new_file_path_list.append(
                    {
                        "filepath": uploaded_dict[fp["filepath"]],
                        "filename": fp["filepath"].split("/")[-1],
                        "dest_folder": fp["dest_folder"],
                    }
                )

        # this is the format that is expected by comfy_runner
        data = {
            "workflow_type": workflow_type,
            "workflow_input": workflow_json,
            "file_path_list": new_file_path_list,
            "output_node_ids": output_node_ids,
            "extra_model_list": extra_model_list,
            "ignore_model_list": ignore_list,
        }

        params = {
            "prompt": query_obj.prompt,  # hackish sol
            InferenceParamType.QUERY_DICT.value: query_obj.to_json(),
            InferenceParamType.API_INFERENCE_DATA.value: json.dumps(data),
            InferenceParamType.FILE_RELATION_DATA.value: query_obj.relation_data,
        }
        return (
            self.predict_model_output(model, **params)
            if not queue_inference
            else self.queue_prediction(model, **params, backlog=backlog)
        )

    def predict_model_output(self, replicate_model: MLModel, **kwargs):
        queue_inference = kwargs.get("queue_inference", False)
        if queue_inference:
            return self.queue_prediction(replicate_model, **kwargs)

        data = kwargs.get(InferenceParamType.GPU_INFERENCE.value, None)
        data = json.loads(data)
        start_time = time.time()
        output = None  # TODO: real time prediction not implemented
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

    def multi_file_upload(self, file_list):
        results = {}

        if not len(file_list):
            return results

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {
                executor.submit(self._upload_file_to_s3, file_info): file_info for file_info in file_list
            }
            for future in as_completed(future_to_file):
                local_path, hosted_path, success, error = future.result()
                if success:
                    results[local_path] = hosted_path
                else:
                    print(f"Failed to upload {local_path}: {error}")

        return results

    def _upload_file_to_s3(self, file_path):
        local_file_path = file_path
        content_type = self._get_content_type(file_path)
        file_expiration = 172800
        signed_url, public_url = self._get_signed_url(
            {
                "file_path": file_path,
                "content_type": content_type,
                "file_expiration": file_expiration,
            }
        )
        if not signed_url:
            return local_file_path, None, False, "Failed to get signed URL"

        try:
            with open(local_file_path, "rb") as file:
                metadata = {}
                metadata["expire_in"] = str(file_expiration)
                headers = {f"x-amz-meta-{key}": value for key, value in metadata.items()}
                if content_type:
                    headers["Content-Type"] = content_type

                response = requests.put(signed_url, data=file, headers=headers)

                if response.status_code == 200:
                    return local_file_path, public_url, True, None
                else:
                    return (
                        local_file_path,
                        None,
                        False,
                        f"Failed to upload. Status code: {response.status_code}",
                    )
        except Exception as e:
            return local_file_path, None, False, str(e)

    def _get_signed_url(self, file_info):
        backend_url = f"{SERVER_URL}/v1/user/file"
        try:
            response = requests.post(
                backend_url, json=file_info, headers={"Authorization": f"Bearer {self.auth_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                return data["payload"]["data"]["signed_url"], data["payload"]["data"]["public_url"]
            else:
                raise Exception("unable to fetch url: ", response.content)
        except requests.RequestException as e:
            print(f"Failed to get signed URL for {file_info}: {str(e)}")
            return None, None

    def _get_content_type(self, file_path):
        content_type, _ = mimetypes.guess_type(file_path)
        return content_type or "application/octet-stream"
