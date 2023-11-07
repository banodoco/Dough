import json
from typing import List
from shared.constants import AIModelCategory
from utils.common_utils import get_current_user_uuid
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import REPLICATE_MODEL

# NOTE: making an exception for this function, passing just the image urls instead of
# image files
def train_model(images_list, instance_prompt, class_prompt, max_train_steps,
                model_name, type_of_model, type_of_task, resolution, controller_type, model_type_list):
    # prepare and upload the training data (images.zip)
    ml_client = get_ml_client()
    try:
        training_file_url = ml_client.upload_training_data(images_list)
    except Exception as e:
        raise e

    # training the model
    model_name = model_name.replace(" ", "-").lower()
    if type_of_model == "Dreambooth":
        return train_dreambooth_model(instance_prompt, class_prompt, training_file_url,
                                      max_train_steps, model_name, images_list, controller_type, model_type_list)
    elif type_of_model == "LoRA":
        return train_lora_model(training_file_url, type_of_task, resolution, model_name, images_list, model_type_list)


# INFO: images_list passed here are converted to internal files after they are used for training
def train_dreambooth_model(instance_prompt, class_prompt, training_file_url, max_train_steps, model_name, images_list: List[str], controller_type, model_type_list):
    from ui_components.methods.common_methods import convert_image_list_to_file_list
    
    ml_client = get_ml_client()
    app_setting = DataRepo().get_app_setting_from_uuid()

    response = ml_client.dreambooth_training(
        training_file_url, instance_prompt, class_prompt, max_train_steps, model_name, controller_type, len(images_list), app_setting.replicate_username)
    training_status = response["status"]
    
    model_id = response["id"]
    if training_status == "queued":
        file_list = convert_image_list_to_file_list(images_list)
        file_uuid_list = [file.uuid for file in file_list]
        file_uuid_list = json.dumps(file_uuid_list)

        model_data = {
            "name": model_name,
            "user_id": get_current_user_uuid(),
            "replicate_model_id": model_id,
            "replicate_url": response["model"],
            "diffusers_url": "",
            "category": AIModelCategory.DREAMBOOTH.value,
            "training_image_list": file_uuid_list,
            "keyword": instance_prompt,
            "custom_trained": True,
            "model_type": model_type_list
        }

        data_repo = DataRepo()
        data_repo.create_ai_model(**model_data)

        return "Success - Training Started. Please wait 10-15 minutes for the model to be trained."
    else:
        return "Failed"

# INFO: images_list passed here are converted to internal files after they are used for training
def train_lora_model(training_file_url, type_of_task, resolution, model_name, images_list, model_type_list):
    from ui_components.methods.common_methods import convert_image_list_to_file_list

    data_repo = DataRepo()
    ml_client = get_ml_client()
    output = ml_client.predict_model_output(REPLICATE_MODEL.clones_lora_training, instance_data=training_file_url,
                                            task=type_of_task, resolution=int(resolution))

    file_list = convert_image_list_to_file_list(images_list)
    file_uuid_list = [file.uuid for file in file_list]
    file_uuid_list = json.dumps(file_uuid_list)
    model_data = {
        "name": model_name,
        "user_id": get_current_user_uuid(),
        "replicate_url": output,
        "diffusers_url": "",
        "category": AIModelCategory.LORA.value,
        "training_image_list": file_uuid_list,
        "custom_trained": True,
        "model_type": model_type_list
    }

    data_repo.create_ai_model(**model_data)
    return f"Successfully trained - the model '{model_name}' is now available for use!"
