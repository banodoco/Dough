import os
import random
import tempfile
import uuid
from backend.models import InternalFileObject
from shared.constants import COMFY_BASE_PATH, InternalFileType
from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from ui_components.methods.common_methods import combine_mask_and_input_image, random_seed
from ui_components.methods.file_methods import save_or_host_file, zip_images, determine_dimensions_for_sdxl
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow, MLModel
import json


MODEL_PATH_DICT = {
    ComfyWorkflow.SDXL: {"workflow_path": "comfy_workflows/sdxl_workflow_api.json", "output_node_id": [19]},
    ComfyWorkflow.SDXL_IMG2IMG: {
        "workflow_path": "comfy_workflows/sdxl_img2img_workflow_api.json",
        "output_node_id": [31],
    },
    ComfyWorkflow.SDXL_CONTROLNET: {
        "workflow_path": "comfy_workflows/sdxl_controlnet_workflow_api.json",
        "output_node_id": [9],
    },
    ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE: {
        "workflow_path": "comfy_workflows/sdxl_openpose_workflow_api.json",
        "output_node_id": [9],
    },
    ComfyWorkflow.LLAMA_2_7B: {
        "workflow_path": "comfy_workflows/llama_workflow_api.json",
        "output_node_id": [14],
    },
    ComfyWorkflow.SDXL_INPAINTING: {
        "workflow_path": "comfy_workflows/sdxl_inpainting_workflow_api.json",
        "output_node_id": [56],
    },
    ComfyWorkflow.IP_ADAPTER_PLUS: {
        "workflow_path": "comfy_workflows/ipadapter_plus_api.json",
        "output_node_id": [29],
    },
    ComfyWorkflow.IP_ADAPTER_FACE: {
        "workflow_path": "comfy_workflows/ipadapter_face_api.json",
        "output_node_id": [29],
    },
    ComfyWorkflow.IP_ADAPTER_FACE_PLUS: {
        "workflow_path": "comfy_workflows/ipadapter_face_plus_api.json",
        "output_node_id": [29],
    },
    ComfyWorkflow.STEERABLE_MOTION: {
        "workflow_path": "comfy_workflows/steerable_motion_api.json",
        "output_node_id": [281],
    },
    ComfyWorkflow.UPSCALER: {
        "workflow_path": "comfy_workflows/video_upscaler_api.json",
        "output_node_id": [402],
    },
    ComfyWorkflow.MOTION_LORA: {
        "workflow_path": "comfy_workflows/motion_lora_api.json",
        "output_node_id": [11, 14, 26, 30, 34],
    },
    # ComfyWorkflow.MOTION_LORA: {"workflow_path": 'comfy_workflows/motion_lora_test_api.json', "output_node_id": [11, 14]},
    ComfyWorkflow.DYNAMICRAFTER: {
        "workflow_path": "comfy_workflows/dynamicrafter_api.json",
        "output_node_id": [2],
    },
    ComfyWorkflow.IPADAPTER_COMPOSITION: {
        "workflow_path": "comfy_workflows/ipadapter_composition_workflow_api.json",
        "output_node_id": [27],
    },
    ComfyWorkflow.CREATIVE_IMAGE_GEN: {
        "workflow_path": "comfy_workflows/creative_image_gen.json",
        "output_node_id": [9],
    },
    ComfyWorkflow.SD3: {
        "workflow_path": "comfy_workflows/sd3_workflow_api.json",
        "output_node_id": [233],
    },
}


# these methods return the workflow along with the output node class name
class ComfyDataTransform:
    # there are certain files which need to be stored in a subfolder
    # creating a dict of filename <-> subfolder_name
    filename_subfolder_dict = {"structure_control_img": "sci"}

    @staticmethod
    def get_workflow_json(model: ComfyWorkflow):
        json_file_path = "./utils/ml_processor/" + MODEL_PATH_DICT[model]["workflow_path"]
        # Specify encoding as 'utf-8' when opening the file
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            return json_data, MODEL_PATH_DICT[model]["output_node_id"]

    @staticmethod
    def transform_sdxl_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale

        # updating params
        seed = random_seed()
        workflow["10"]["inputs"]["noise_seed"] = seed
        workflow["10"]["inputs"]["noise_seed"] = seed

        workflow["4"]["inputs"]["ckpt_name"] = model

        workflow["5"]["inputs"]["width"], workflow["5"]["inputs"]["height"] = width, height
        workflow["6"]["inputs"]["text"] = workflow["15"]["inputs"]["text"] = positive_prompt
        workflow["7"]["inputs"]["text"] = workflow["16"]["inputs"]["text"] = negative_prompt
        workflow["10"]["inputs"]["steps"], workflow["10"]["inputs"]["cfg"] = steps, cfg
        workflow["11"]["inputs"]["steps"], workflow["11"]["inputs"]["cfg"] = steps, cfg

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_sdxl_img2img_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_IMG2IMG)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = 20, 7  # hardcoding values
        strength = round(query.strength / 100, 1)
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["1"]["inputs"]["ckpt_name"] = model

        workflow["37:0"]["inputs"]["image"] = image_name
        workflow["42:0"]["inputs"]["text"] = positive_prompt
        workflow["42:1"]["inputs"]["text"] = negative_prompt
        workflow["42:2"]["inputs"]["steps"] = steps
        workflow["42:2"]["inputs"]["cfg"] = cfg
        workflow["42:2"]["inputs"]["denoise"] = 1 - strength
        workflow["42:2"]["inputs"]["seed"] = random_seed()

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_sdxl_controlnet_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_CONTROLNET)

        # workflow params
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        low_threshold, high_threshold = query.low_threshold, query.high_threshold
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["17"]["width"], workflow["17"]["height"] = width, height
        workflow["6"]["inputs"]["text"], workflow["7"]["inputs"]["text"] = positive_prompt, negative_prompt
        workflow["12"]["inputs"]["low_threshold"], workflow["12"]["inputs"]["high_threshold"] = (
            low_threshold,
            high_threshold,
        )
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["13"]["inputs"]["image"] = image_name

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadapter_composition_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IPADAPTER_COMPOSITION)

        # workflow params
        width, height = query.width, query.height
        # width, height = determine_dimensions_for_sdxl(width, height)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        # low_threshold, high_threshold = query.low_threshold, query.high_threshold
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["9"]["inputs"]["seed"] = random_seed()
        workflow["10"]["width"], workflow["10"]["height"] = width, height
        # workflow["17"]["width"], workflow["17"]["height"] = width, height
        workflow["7"]["inputs"]["text"], workflow["8"]["inputs"]["text"] = positive_prompt, negative_prompt
        # workflow["12"]["inputs"]["low_threshold"], workflow["12"]["inputs"]["high_threshold"] = low_threshold, high_threshold
        workflow["9"]["inputs"]["steps"], workflow["9"]["inputs"]["cfg"] = steps, cfg
        workflow["6"]["inputs"]["image"] = image_name
        workflow["28"]["inputs"]["weight"] = query.strength

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_sdxl_controlnet_openpose_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(
            ComfyWorkflow.SDXL_CONTROLNET_OPENPOSE
        )

        # workflow params
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)

        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["11"]["width"], workflow["11"]["height"] = width, height
        workflow["6"]["inputs"]["text"], workflow["7"]["inputs"]["text"] = positive_prompt, negative_prompt
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["12"]["inputs"]["image"] = image_name

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_llama_2_7b_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.LLAMA_2_7B)

        # workflow params
        input_text = query.prompt
        temperature = query.data.get("temperature", 0.8)

        # updating params
        workflow["15"]["inputs"]["prompt"] = input_text
        workflow["15"]["inputs"]["temperature"] = temperature

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_sdxl_inpainting_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SDXL_INPAINTING)

        # workflow params
        # node 'get_img_size' automatically fetches the size
        model = query.data["data"].get("sdxl_model", None)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        input_image = query.data.get("data", {}).get("input_image", None)
        mask = query.data.get("data", {}).get("mask", None)
        timing = data_repo.get_timing_from_uuid(query.timing_uuid)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)

        # inpainting workflows takes in an image and inpaints the transparent area
        combined_img = combine_mask_and_input_image(mask, input_image)
        # down combined_img PIL image to the current directory
        filename = str(uuid.uuid4()) + ".png"
        hosted_url = save_or_host_file(combined_img, "videos/temp/" + filename)

        file_data = {
            "name": filename,
            "type": InternalFileType.IMAGE.value,
            "project_id": query.data.get("data", {}).get("project_uuid"),
        }

        if hosted_url:
            file_data.update({"hosted_url": hosted_url})
        else:
            file_data.update({"local_path": "videos/temp/" + filename})

        file = data_repo.create_file(**file_data)
        # adding the combined image in query (and removing io buffers)
        query.data = {"data": {"file_combined_img": file.uuid}}
        # updating params
        workflow["29"]["inputs"]["ckpt_name"] = model
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["20"]["inputs"]["image"] = filename
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["34"]["inputs"]["text_g"] = workflow["34"]["inputs"]["text_l"] = positive_prompt
        workflow["37"]["inputs"]["text_g"] = workflow["37"]["inputs"]["text_l"] = negative_prompt
        workflow["37"]["inputs"]["height"] = height
        workflow["37"]["inputs"]["width"] = width
        workflow["37"]["inputs"]["target_height"] = height
        workflow["37"]["inputs"]["target_width"] = width
        workflow["50"]["inputs"]["height"] = height
        workflow["50"]["inputs"]["width"] = width
        workflow["52"]["inputs"]["height"] = height
        workflow["52"]["inputs"]["width"] = width
        workflow["59"]["inputs"]["width"] = width
        workflow["58"]["inputs"]["width"] = height

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadaptor_plus_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_PLUS)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename

        # updating params
        workflow["4"]["inputs"]["ckpt_name"] = model
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        # workflow["24"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["28"]["inputs"]["image"] = image_name  # dummy image
        workflow["6"]["inputs"]["text"] = query.prompt
        workflow["7"]["inputs"]["text"] = query.negative_prompt
        workflow["27"]["inputs"]["weight"] = query.strength

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadaptor_face_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        strength = query.strength

        # updating params
        workflow["4"]["inputs"]["ckpt_name"] = model
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["24"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["6"]["inputs"]["text"] = query.prompt
        workflow["7"]["inputs"]["text"] = query.negative_prompt
        workflow["36"]["inputs"]["weight"] = query.strength
        workflow["36"]["inputs"]["weight_v2"] = query.strength

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_ipadaptor_face_plus_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE_PLUS)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = data_repo.get_file_from_uuid(query.image_uuid)
        image_name = image.filename
        image_2 = data_repo.get_file_from_uuid(query.data.get("data", {}).get("file_image_2_uuid", None))
        image_name_2 = image_2.filename if image_2 else None

        # updating params
        workflow["4"]["inputs"]["ckpt_name"] = model
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["5"]["width"], workflow["5"]["height"] = width, height
        workflow["3"]["inputs"]["steps"], workflow["3"]["inputs"]["cfg"] = steps, cfg
        workflow["24"]["inputs"]["image"] = image_name  # ipadapter image
        workflow["28"]["inputs"]["image"] = image_name_2  # insight face image
        workflow["6"]["inputs"]["text"] = query.prompt
        workflow["7"]["inputs"]["text"] = query.negative_prompt
        workflow["29"]["inputs"]["weight"] = query.strength[0]
        workflow["27"]["inputs"]["weight"] = query.strength[1]

        return json.dumps(workflow), output_node_ids, [], []

    @staticmethod
    def transform_steerable_motion_workflow(query: MLQueryObject):
        def update_structure_control_image(json, image_uuid, weight):
            # Integrate all updates including new nodes and modifications in a single step
            data_repo = DataRepo()
            image = data_repo.get_file_from_uuid(image_uuid)

            json.update(
                {
                    "560": {
                        "inputs": {
                            "image": "sci/"
                            + image.filename,  # TODO: hardcoding for now, pick a proper flow later
                            "upload": "image",
                        },
                        "class_type": "LoadImage",
                        "_meta": {"title": "Load Image"},
                    },
                    "561": {
                        "inputs": {
                            "interpolation": "LANCZOS",
                            "crop_position": "top",
                            "sharpening": 0,
                            "image": ["560", 0],
                        },
                        "class_type": "PrepImageForClipVision",
                        "_meta": {"title": "Prep Image For ClipVision"},
                    },
                    "578": {
                        "inputs": {"ipadapter_file": "ip_plus_composition_sd15.safetensors"},
                        "class_type": "IPAdapterModelLoader",
                        "_meta": {"title": "IPAdapter Model Loader"},
                    },
                    "581": {
                        "inputs": {
                            "weight": weight,
                            "weight_type": "ease in-out",
                            "combine_embeds": "concat",
                            "start_at": 0,
                            "end_at": 0.75,
                            "embeds_scaling": "V only",
                            "model": ["461", 0],
                            "ipadapter": ["578", 0],
                            "image": ["561", 0],
                            "clip_vision": ["370", 0],
                        },
                        "class_type": "IPAdapterAdvanced",
                        "_meta": {"title": "IPAdapter Advanced"},
                    },
                }
            )

            # Update the "558" node's model pair to point to "581"
            if "558" in json:
                json["558"]["inputs"]["model"] = ["581", 0]

            return json

        def update_json_with_loras(json_data, loras):
            start_id = 536
            new_ids = []

            # Add LoRAs
            for lora in loras:
                new_id = str(start_id)
                json_data[new_id] = {
                    "inputs": {
                        "lora_name": lora["filename"],
                        "strength": lora["lora_strength"],
                    },
                    "class_type": "ADE_AnimateDiffLoRALoader",
                    "_meta": {"title": "Load AnimateDiff LoRA 🎭🅐🅓"},
                }

                if new_ids:
                    json_data[new_ids[-1]]["inputs"]["prev_motion_lora"] = [new_id, 0]

                new_ids.append(new_id)
                start_id += 1

            # Update node 545 if needed and if there are new items
            if "545" in json_data and len(new_ids):
                if "motion_lora" not in json_data["545"]["inputs"]:
                    # If "motion_lora" is not present, add it with the specified values
                    json_data["545"]["inputs"]["motion_lora"] = ["536", 0]
                else:
                    # If "motion_lora" is already present, just update the first value
                    json_data["545"]["inputs"]["motion_lora"][0] = "536"

            return json_data

        def convert_to_specific_workflow(json_data, type_of_generation, extra_models_list):

            if type_of_generation == "Slurshy Realistiche":
                json_data["593"] = {
                    "inputs": {
                        "ipa_starts_at": 0,
                        "ipa_ends_at": 0.5,
                        "ipa_weight_type": "ease in-out",
                        "ipa_weight": 1,
                        "ipa_embeds_scaling": "V only",
                        "ipa_noise_strength": 0.8,
                        "use_image_for_noise": False,
                        "type_of_noise": "gaussian",
                        "noise_blur": 0,
                    },
                    "class_type": "IpaConfiguration",
                    "_meta": {"title": "IPA Configuration  🎞️🅢🅜"},
                }

                json_data["594"] = {
                    "inputs": {
                        "ipa_starts_at": 0,
                        "ipa_ends_at": 1,
                        "ipa_weight_type": "ease out",
                        "ipa_weight": 1,
                        "ipa_embeds_scaling": "K+mean(V) w/ C penalty",
                        "ipa_noise_strength": 0.0,
                        "use_image_for_noise": True,
                        "type_of_noise": "fade",
                        "noise_blur": 1,
                    },
                    "class_type": "IpaConfiguration",
                    "_meta": {"title": "IPA Configuration  🎞️🅢🅜"},
                }

                return json_data, extra_models_list

            elif type_of_generation == "Fast With A Price":
                json_data.update(
                    {
                        "565": {
                            "inputs": {
                                "lora_name": "AnimateLCM_sd15_t2v_lora.safetensors",
                                "strength_model": 0.8,
                                "strength_clip": 1,
                                "model": ["461", 0],
                                "clip": ["461", 1],
                            },
                            "class_type": "LoraLoader",
                            "_meta": {"title": "Load LoRA"},
                        }
                    }
                )

                json_data["558"]["inputs"]["model"] = ["565", 0]
                json_data["541"]["inputs"]["clip"] = ["565", 1]
                json_data["543"]["inputs"]["clip"] = ["565", 1]
                json_data["547"]["inputs"]["beta_schedule"] = "lcm avg(sqrt_linear,linear)"

                json_data["207"]["inputs"]["sampler_name"] = "lcm"
                json_data["207"]["inputs"]["steps"] = 8
                json_data["207"]["inputs"]["cfg"] = 2.2
                json_data["546"]["inputs"]["model_name"] = "AnimateLCM_sd15_t2v.ckpt"
                json_data["207"]["inputs"]["scheduler"] = "sgm_uniform"

                extra_models_list.append(
                    {
                        "filename": "AnimateLCM_sd15_t2v_lora.safetensors",
                        "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors?download=true",
                        "dest": os.path.join(COMFY_BASE_PATH, "models", "loras"),
                    }
                )
                extra_models_list.append(
                    {
                        "filename": "AnimateLCM_sd15_t2v.ckpt",
                        "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt",
                        "dest": os.path.join(COMFY_BASE_PATH, "models", "animatediff_models"),
                    }
                )

                return json_data, extra_models_list

            elif type_of_generation == "Smooth n' Steady":
                return json_data, extra_models_list

            elif type_of_generation == "Liquidy Loop":
                json_data.update(
                    {
                        "565": {
                            "inputs": {
                                "lora_name": "AnimateLCM_sd15_t2v_lora.safetensors",
                                "strength_model": 1.05,
                                "strength_clip": 1,
                                "model": ["461", 0],
                                "clip": ["461", 1],
                            },
                            "class_type": "LoraLoader",
                            "_meta": {"title": "Load LoRA"},
                        }
                    }
                )

                json_data["558"]["inputs"]["model"] = ["565", 0]
                json_data["541"]["inputs"]["clip"] = ["565", 1]
                json_data["543"]["inputs"]["clip"] = ["565", 1]
                json_data["547"]["inputs"]["beta_schedule"] = "lcm avg(sqrt_linear,linear)"

                json_data["207"]["inputs"]["sampler_name"] = "lcm"
                json_data["207"]["inputs"]["steps"] = 20
                json_data["207"]["inputs"]["cfg"] = 1.2
                json_data["546"]["inputs"]["model_name"] = "AnimateLCM_sd15_t2v.ckpt"
                json_data["207"]["inputs"]["scheduler"] = "sgm_uniform"

                json_data["593"]["inputs"]["ipa_starts_at"] = 0
                json_data["593"]["inputs"]["ipa_ends_at"] = 0.3
                json_data["593"]["inputs"]["ipa_weight_type"] = "ease in-out"
                json_data["593"]["inputs"]["ipa_weight"] = 1
                json_data["593"]["inputs"]["ipa_embeds_scaling"] = "V only"
                json_data["593"]["inputs"]["ipa_noise_strength"] = 0.9
                json_data["593"]["inputs"]["use_image_for_noise"] = True
                json_data["593"]["inputs"]["type_of_noise"] = "fade"
                json_data["593"]["inputs"]["noise_blur"] = 0

                json_data["594"]["inputs"]["ipa_starts_at"] = 0
                json_data["594"]["inputs"]["ipa_ends_at"] = 1
                json_data["594"]["inputs"]["ipa_weight_type"] = "strong middle"
                json_data["594"]["inputs"]["ipa_weight"] = 1
                json_data["594"]["inputs"]["ipa_embeds_scaling"] = "K+mean(V) w/ C penalty"
                json_data["594"]["inputs"]["ipa_noise_strength"] = 0.1
                json_data["594"]["inputs"]["use_image_for_noise"] = True
                json_data["594"]["inputs"]["type_of_noise"] = "fade"
                json_data["594"]["inputs"]["noise_blur"] = 0

                extra_models_list.append(
                    {
                        "filename": "AnimateLCM_sd15_t2v_lora.safetensors",
                        "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors?download=true",
                        "dest": os.path.join(COMFY_BASE_PATH, "models", "loras"),
                    }
                )
                extra_models_list.append(
                    {
                        "filename": "AnimateLCM_sd15_t2v.ckpt",
                        "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt",
                        "dest": os.path.join(COMFY_BASE_PATH, "models", "animatediff_models"),
                    }
                )

                return json_data, extra_models_list

            elif type_of_generation == "Chocky Realistiche":

                json_data["593"]["inputs"]["ipa_starts_at"] = 0
                json_data["593"]["inputs"]["ipa_ends_at"] = 0.28
                json_data["593"]["inputs"]["ipa_weight_type"] = "ease in-out"
                json_data["593"]["inputs"]["ipa_weight"] = 1
                json_data["593"]["inputs"]["ipa_embeds_scaling"] = "V only"
                json_data["593"]["inputs"]["ipa_noise_strength"] = 0.30
                json_data["593"]["inputs"]["use_image_for_noise"] = True
                json_data["593"]["inputs"]["type_of_noise"] = "fade"
                json_data["593"]["inputs"]["noise_blur"] = 0

                json_data["594"]["inputs"]["ipa_starts_at"] = 0
                json_data["594"]["inputs"]["ipa_ends_at"] = 1
                json_data["594"]["inputs"]["ipa_weight_type"] = "strong middle"
                json_data["594"]["inputs"]["ipa_weight"] = 1.0
                json_data["594"]["inputs"]["ipa_embeds_scaling"] = "V only"
                json_data["594"]["inputs"]["ipa_noise_strength"] = 0.00
                json_data["594"]["inputs"]["use_image_for_noise"] = True
                json_data["594"]["inputs"]["type_of_noise"] = "fade"
                json_data["594"]["inputs"]["noise_blur"] = 0

                return json_data, extra_models_list

            elif type_of_generation == "Rad Attack":

                json_data.update(
                    {
                        "565": {
                            "inputs": {
                                "lora_name": "AnimateLCM_sd15_t2v_lora.safetensors",
                                "strength_model": 1.05,
                                "strength_clip": 1,
                                "model": ["461", 0],
                                "clip": ["461", 1],
                            },
                            "class_type": "LoraLoader",
                            "_meta": {"title": "Load LoRA"},
                        }
                    }
                )

                json_data["558"]["inputs"]["model"] = ["565", 0]
                json_data["541"]["inputs"]["clip"] = ["565", 1]
                json_data["543"]["inputs"]["clip"] = ["565", 1]

                json_data["207"]["inputs"]["steps"] = 20
                json_data["207"]["inputs"]["cfg"] = 1.2
                json_data["207"]["inputs"]["sampler_name"] = "lcm"
                json_data["207"]["inputs"]["scheduler"] = "sgm_uniform"

                json_data["546"]["inputs"]["model_name"] = "AnimateLCM_sd15_t2v.ckpt"

                json_data["342"]["inputs"]["fuse_method"] = "pyramid"

                json_data["547"]["inputs"]["beta_schedule"] = "lcm avg(sqrt_linear,linear)"

                json_data["593"]["inputs"]["ipa_starts_at"] = 0
                json_data["593"]["inputs"]["ipa_ends_at"] = 0.3
                json_data["593"]["inputs"]["ipa_weight_type"] = "ease in-out"
                json_data["593"]["inputs"]["ipa_weight"] = 1
                json_data["593"]["inputs"]["ipa_embeds_scaling"] = "V only"
                json_data["593"]["inputs"]["ipa_noise_strength"] = 0.5
                json_data["593"]["inputs"]["use_image_for_noise"] = True
                json_data["593"]["inputs"]["type_of_noise"] = "shuffle"
                json_data["593"]["inputs"]["noise_blur"] = 0

                json_data["594"]["inputs"]["ipa_starts_at"] = 0
                json_data["594"]["inputs"]["ipa_ends_at"] = 1
                json_data["594"]["inputs"]["ipa_weight_type"] = "ease in-out"
                json_data["594"]["inputs"]["ipa_weight"] = 1
                json_data["594"]["inputs"]["ipa_embeds_scaling"] = "K+mean(V) w/ C penalty"
                json_data["594"]["inputs"]["ipa_noise_strength"] = 0.0
                json_data["594"]["inputs"]["use_image_for_noise"] = False
                json_data["594"]["inputs"]["type_of_noise"] = "shuffle"
                json_data["594"]["inputs"]["noise_blur"] = 0

                extra_models_list.append(
                    {
                        "filename": "AnimateLCM_sd15_t2v_lora.safetensors",
                        "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors?download=true",
                        "dest": os.path.join(COMFY_BASE_PATH, "models", "loras"),
                    }
                )
                extra_models_list.append(
                    {
                        "filename": "AnimateLCM_sd15_t2v.ckpt",
                        "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt",
                        "dest": os.path.join(COMFY_BASE_PATH, "models", "animatediff_models"),
                    }
                )

                return json_data, extra_models_list

        extra_models_list = []
        sm_data = query.data.get("data", {})
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.STEERABLE_MOTION)

        filename_prefix = sm_data.get("filename_prefix", None)
        if filename_prefix:
            workflow["281"]["inputs"]["filename_prefix"] = filename_prefix

        workflow = update_json_with_loras(workflow, sm_data.get("lora_data"))

        workflow["464"]["inputs"]["height"] = sm_data.get("height")
        workflow["464"]["inputs"]["width"] = sm_data.get("width")
        workflow["584"]["inputs"]["value"] = sm_data.get("width")
        workflow["585"]["inputs"]["value"] = sm_data.get("height")

        ckpt = sm_data.get("ckpt")
        if "ComfyUI/models/checkpoints/" != ckpt and ckpt:
            workflow["461"]["inputs"]["ckpt_name"] = ckpt

        workflow["548"]["inputs"]["text"] = sm_data.get("motion_scales")

        workflow["281"]["inputs"]["format"] = sm_data.get("output_format")

        workflow["559"]["inputs"]["multiplier"] = sm_data.get("stmfnet_multiplier")

        workflow["558"]["inputs"]["buffer"] = sm_data.get("buffer")
        workflow["558"]["inputs"]["type_of_strength_distribution"] = sm_data.get(
            "type_of_strength_distribution"
        )
        workflow["558"]["inputs"]["linear_strength_value"] = sm_data.get("linear_strength_value")
        workflow["558"]["inputs"]["dynamic_strength_values"] = str(sm_data.get("dynamic_strength_values"))[
            1:-1
        ]
        workflow["558"]["inputs"]["linear_frame_distribution_value"] = sm_data.get(
            "linear_frame_distribution_value"
        )
        workflow["558"]["inputs"]["dynamic_frame_distribution_values"] = ", ".join(
            str(int(value)) for value in sm_data.get("dynamic_frame_distribution_values")
        )
        workflow["558"]["inputs"]["type_of_frame_distribution"] = sm_data.get("type_of_frame_distribution")
        workflow["558"]["inputs"]["type_of_key_frame_influence"] = sm_data.get("type_of_key_frame_influence")
        workflow["558"]["inputs"]["linear_key_frame_influence_value"] = sm_data.get(
            "linear_key_frame_influence_value"
        )
        workflow["558"]["inputs"]["high_detail_mode"] = sm_data.get("high_detail_mode")
        workflow["558"]["inputs"]["dynamic_key_frame_influence_values"] = str(
            sm_data.get("dynamic_key_frame_influence_values")
        )[1:-1]

        workflow["342"]["inputs"]["context_length"] = sm_data.get("context_length")
        workflow["342"]["inputs"]["context_stride"] = sm_data.get("context_stride")
        workflow["342"]["inputs"]["context_overlap"] = sm_data.get("context_overlap")

        workflow["468"]["inputs"]["end_percent"] = sm_data.get("multipled_base_end_percent")

        workflow["207"]["inputs"]["noise_seed"] = random_seed()

        workflow["541"]["inputs"]["pre_text"] = sm_data.get("prompt")
        workflow["541"]["inputs"]["text"] = sm_data.get("individual_prompts")
        workflow["541"]["inputs"]["max_frames"] = int(float(sm_data.get("max_frames")))

        workflow["543"]["inputs"]["pre_text"] = sm_data.get("negative_prompt")
        workflow["543"]["inputs"]["max_frames"] = int(float(sm_data.get("max_frames")))
        workflow["543"]["inputs"]["text"] = sm_data.get("individual_negative_prompts")

        if sm_data.get("file_structure_control_img_uuid"):
            workflow = update_structure_control_image(
                workflow,
                sm_data.get("file_structure_control_img_uuid"),
                sm_data.get("strength_of_structure_control_image"),
            )

        workflow, extra_models_list = convert_to_specific_workflow(
            workflow, sm_data.get("type_of_generation", "Fast With A Price"), extra_models_list
        )

        ignore_list = sm_data.get("lora_data", [])
        return json.dumps(workflow), output_node_ids, extra_models_list, ignore_list

    @staticmethod
    def transform_dynamicrafter_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.DYNAMICRAFTER)
        sm_data = query.data.get("data", {})

        image_1 = data_repo.get_file_from_uuid(sm_data.get("file_image_0001_uuid"))
        image_2 = data_repo.get_file_from_uuid(sm_data.get("file_image_0002_uuid"))

        workflow["16"]["inputs"]["image"] = image_1.filename
        workflow["17"]["inputs"]["image"] = image_2.filename
        workflow["12"]["inputs"]["seed"] = random_seed()
        workflow["12"]["inputs"]["steps"] = 50
        workflow["12"]["inputs"]["cfg"] = 4
        workflow["12"]["inputs"]["prompt"] = sm_data.get("prompt")

        extra_models_list = [
            {
                "filename": "dynamicrafter_512_interp_v1.ckpt",
                "url": "https://huggingface.co/Doubiiu/DynamiCrafter_512_Interp/resolve/main/model.ckpt?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "checkpoints"),
            }
        ]

        return json.dumps(workflow), output_node_ids, extra_models_list, []

    @staticmethod
    def transform_video_upscaler_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.UPSCALER)
        data = query.data.get("data", {})
        video_uuid = data.get("file_video", None)
        video = data_repo.get_file_from_uuid(video_uuid)
        model = data.get("model", None)

        upscale_factor = data.get("upscale_factor", None)

        workflow["302"]["inputs"]["video"] = os.path.basename(video.filename)
        workflow["362"]["inputs"]["ckpt_name"] = model
        workflow["391"]["inputs"]["upscale_by"] = upscale_factor

        extra_models_list = [
            {
                "filename": "AnimateLCM_sd15_t2v_lora.safetensors",
                "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "loras"),
            },
            {
                "filename": "AnimateLCM_sd15_t2v.ckpt",
                "url": "https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "animatediff_models"),
            },
            {
                "filename": "4x_RealisticRescaler_100000_G.pth",
                "url": "https://huggingface.co/holwech/realistic-rescaler-real-esrgan/resolve/main/4x_RealisticRescaler_100000_G.pth?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "upscale_models"),
            },
        ]

        return json.dumps(workflow), output_node_ids, extra_models_list, []

    @staticmethod
    def transform_motion_lora_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.MOTION_LORA)
        data = query.data.get("data", {})
        video_uuid = data.get("file_video", None)
        video = data_repo.get_file_from_uuid(video_uuid)
        lora_name = data.get("lora_name", "")

        workflow["5"]["inputs"]["video"] = os.path.basename(video.filename)
        workflow["5"]["inputs"]["custom_width"] = query.width
        workflow["5"]["inputs"]["custom_height"] = query.height
        workflow["4"]["inputs"]["lora_name"] = lora_name
        workflow["4"]["inputs"]["prompt"] = query.prompt
        workflow["15"]["inputs"]["validation_prompt"] = query.prompt

        ckpt = data.get("ckpt")
        if "ComfyUI/models/checkpoints/" != ckpt and ckpt:
            workflow["1"]["inputs"]["ckpt_name"] = ckpt

        extra_models_list = [
            {
                "filename": "v3_sd15_mm.ckpt",
                "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "animatediff_models"),
            },
            {
                "filename": "v3_sd15_adapter.ckpt",
                "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "loras"),
            },
        ]

        return json.dumps(workflow), output_node_ids, extra_models_list, []

    @staticmethod
    def transform_sd3_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.SD3)

        # workflow params
        query_data = query.data["data"]
        shift = query_data.get("shift", 3.0)
        model = query_data.get("model", "")
        height, width = query.height, query.width
        width, height = determine_dimensions_for_sdxl(width, height)
        image_prompt, negative_prompt = query.prompt, query.negative_prompt
        seed = random_seed()

        workflow["6"]["inputs"]["text"] = image_prompt
        workflow["71"]["inputs"]["text"] = negative_prompt

        workflow["13"]["inputs"]["shift"] = shift

        workflow["252"]["inputs"]["ckpt_name"] = model
        workflow["271"]["inputs"]["seed"] = seed
        workflow["135"]["inputs"]["width"] = width
        workflow["135"]["inputs"]["height"] = height
        
        # adding download link if it's the default model
        extra_model_list = []
        if model == "sd3_medium_incl_clips.safetensors":
            extra_model_list = [
                {
                    "url": "https://huggingface.co/lone682/sd3/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors?download=true",
                    "filename": "sd3_medium_incl_clips.safetensors",
                    "dest": os.path.join(COMFY_BASE_PATH, "models", "checkpoints"),
                }
            ]

        return json.dumps(workflow), output_node_ids, extra_model_list, []

    @staticmethod
    def transform_creative_img_gen_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.CREATIVE_IMAGE_GEN)

        # workflow params
        query_data = query.data["data"]
        height, width = query_data.get("height", 512), query_data.get("width", 512)
        image_prompt, negative_prompt = query.prompt, query.negative_prompt
        file_uuid_list = json.loads(query_data.get("img_uuid_list", json.dumps([])))
        lightening = query_data.get("lightening", False)
        additional_description_text = query_data.get("additional_description_text", "")
        additional_style_text = query_data.get("additional_style_text", "")
        model = query_data.get("model", "sd_xl_base_1.0.safetensors")
        seed = random_seed()
        style_strength = query.strength

        def add_nth_node(workflow, n, img_file: InternalFileObject):
            ipa_node_idx_list = []
            for k, v in workflow.items():
                if v["class_type"] == "IPAdapterStyleComposition":
                    ipa_node_idx_list.append(int(k))  # NOTE: in some weird af workflow these are floats

            ipa_node_idx_list.sort(reverse=True)
            if len(ipa_node_idx_list) == n:
                # nth node is already present (not checking for connections, assuming they are correctly set)
                for k, v in workflow.items():
                    if v["class_type"] == "LoadImage":
                        workflow[str(k)]["inputs"]["image"] = img_file.filename
                        break
                return ipa_node_idx_list[0]
            else:
                # creating new nodes (not handling the case if there are multiple nodes)
                # starting idx from 100, just to be safe
                node_idx = 100 + n * 4
                workflow[str(node_idx)] = {
                    "inputs": {"image": img_file.filename, "upload": "image"},
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Image"},
                }
                workflow[str(node_idx + 1)] = {
                    "inputs": {
                        "type": "dissolve",
                        "strength": 0.7000000000000001,
                        "blur": 0,
                        "image_optional": [str(node_idx), 0],
                    },
                    "class_type": "IPAdapterNoise",
                    "_meta": {"title": "IPAdapter Noise"},
                }
                workflow[str(node_idx + 2)] = {
                    "inputs": {
                        "weight_style": 0.7000000000000001,
                        "weight_composition": 0,
                        "expand_style": True,
                        "combine_embeds": "concat",
                        "start_at": 0,
                        "end_at": 0.85,
                        "embeds_scaling": "K+V w/ C penalty",
                        "model": [str(ipa_node_idx_list[0]), 0],
                        "ipadapter": ["11", 1],
                        "image_style": [str(node_idx), 0],
                        "image_composition": [str(node_idx), 0],
                        "image_negative": [str(node_idx + 1), 0],
                    },
                    "class_type": "IPAdapterStyleComposition",
                    "_meta": {"title": "IPAdapter Style & Composition SDXL"},
                }

                workflow["3"]["inputs"]["model"] = [str(node_idx + 2), 0]

                return node_idx + 2  # ipadapter style composition node idx

        # this will create nodes require to feed num_imgs in the workflow
        # all newly created ipa nodes will have the same kwargs settings
        def add_reference_images(workflow, img_list, **kwargs):
            for i in range(len(img_list)):
                # creating a node
                node_idx = add_nth_node(workflow, i + 1, img_list[i])
                # setting params
                for k, v in kwargs.items():
                    if k in workflow[str(node_idx)]["inputs"]:
                        workflow[str(node_idx)]["inputs"][k] = v

        # updating params
        workflow["3"]["inputs"]["seed"] = seed

        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height

        workflow["4"]["inputs"]["ckpt_name"] = model
        workflow["6"]["inputs"]["text"] = (
            image_prompt + ", " + additional_description_text + ", " + additional_style_text
        )

        workflow["7"]["inputs"]["text"] = negative_prompt

        if lightening:
            workflow["3"]["inputs"]["cfg"] = 1.9
            workflow["3"]["inputs"]["steps"] = 12
            workflow["3"]["inputs"]["scheduler"] = "sgm_uniform"

        img_list, _ = data_repo.get_all_file_list(uuid__in=file_uuid_list, is_disabled=False)
        add_reference_images(workflow, img_list, weight_style=style_strength)

        extra_model_list = [
            {
                "filename": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "clip_vision"),
            },
            {
                "filename": "ip-adapter-plus_sdxl_vit-h.safetensors",
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "ipadapter"),
            },
        ]

        # with open("ws.json", "w") as file:
        #     file.write(json.dumps(workflow))

        return json.dumps(workflow), output_node_ids, extra_model_list, []


# NOTE: only populating with models currently in use
MODEL_WORKFLOW_MAP = {
    ML_MODEL.sdxl.workflow_name: ComfyDataTransform.transform_sdxl_workflow,
    ML_MODEL.sdxl_controlnet.workflow_name: ComfyDataTransform.transform_sdxl_controlnet_workflow,
    ML_MODEL.sdxl_controlnet_openpose.workflow_name: ComfyDataTransform.transform_sdxl_controlnet_openpose_workflow,
    ML_MODEL.ipadapter_composition.workflow_name: ComfyDataTransform.transform_ipadapter_composition_workflow,
    ML_MODEL.llama_2_7b.workflow_name: ComfyDataTransform.transform_llama_2_7b_workflow,
    ML_MODEL.sdxl_inpainting.workflow_name: ComfyDataTransform.transform_sdxl_inpainting_workflow,
    ML_MODEL.ipadapter_plus.workflow_name: ComfyDataTransform.transform_ipadaptor_plus_workflow,
    ML_MODEL.ipadapter_face.workflow_name: ComfyDataTransform.transform_ipadaptor_face_workflow,
    ML_MODEL.ipadapter_face_plus.workflow_name: ComfyDataTransform.transform_ipadaptor_face_plus_workflow,
    ML_MODEL.ad_interpolation.workflow_name: ComfyDataTransform.transform_steerable_motion_workflow,
    ML_MODEL.dynamicrafter.workflow_name: ComfyDataTransform.transform_dynamicrafter_workflow,
    ML_MODEL.sdxl_img2img.workflow_name: ComfyDataTransform.transform_sdxl_img2img_workflow,
    ML_MODEL.video_upscaler.workflow_name: ComfyDataTransform.transform_video_upscaler_workflow,
    ML_MODEL.motion_lora_trainer.workflow_name: ComfyDataTransform.transform_motion_lora_workflow,
    ML_MODEL.creative_image_gen.workflow_name: ComfyDataTransform.transform_creative_img_gen_workflow,
    ML_MODEL.sd3_local.workflow_name: ComfyDataTransform.transform_sd3_workflow,
}


# returns stringified json of the workflow
def get_model_workflow_from_query(model: MLModel, query_obj: MLQueryObject) -> str:
    if model.workflow_name not in MODEL_WORKFLOW_MAP:
        app_logger.log(LoggingType.ERROR, f"model {model.workflow_name} not supported for local inference")
        raise ValueError(f"Model {model.workflow_name} not supported for local inference")

    return MODEL_WORKFLOW_MAP[model.workflow_name](query_obj)


def get_workflow_json_url(workflow_json):
    from utils.ml_processor.ml_interface import get_ml_client

    ml_client = get_ml_client()
    temp_fd, temp_json_path = tempfile.mkstemp(suffix=".json")

    with open(temp_json_path, "w") as temp_json_file:
        temp_json_file.write(workflow_json)

    return ml_client.upload_training_data(temp_json_path, delete_after_upload=True)


# TODO: fix this and define a proper interface of passing files through query_obj
def get_file_list_from_query_obj(query_obj: MLQueryObject):
    file_uuid_list = []
    custom_dest = {}

    if query_obj.image_uuid:
        file_uuid_list.append(query_obj.image_uuid)

    if query_obj.mask_uuid:
        file_uuid_list.append(query_obj.mask_uuid)

    for file_key, file_uuid in query_obj.data.get("data", {}).items():
        if file_key.startswith("file_"):
            dest = ""
            for filename in ComfyDataTransform.filename_subfolder_dict.keys():
                if filename in file_key:
                    dest = ComfyDataTransform.filename_subfolder_dict[filename]
                    break

            if dest:
                custom_dest[str(file_uuid)] = dest

            file_uuid_list.append(file_uuid)

    return file_uuid_list, custom_dest


# returns the zip file which can be passed to the comfy_runner replicate endpoint
def get_file_zip_url(file_uuid_list, index_files=False) -> str:
    from utils.ml_processor.ml_interface import get_ml_client

    data_repo = DataRepo()
    ml_client = get_ml_client()

    file_list = data_repo.get_image_list_from_uuid_list(file_uuid_list)
    filename_list = (
        [f.filename for f in file_list] if not index_files else []
    )  # file names would be indexed like 1.png, 2.png ...
    zip_path = zip_images([f.location for f in file_list], "videos/temp/input_images.zip", filename_list)

    return ml_client.upload_training_data(zip_path, delete_after_upload=True)
