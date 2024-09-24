import os
import random
import tempfile
import uuid
from backend.models import InternalFileObject
from shared.constants import COMFY_BASE_PATH, InternalFileType
from shared.logging.constants import LoggingType
from shared.logging.logging import app_logger
from ui_components.methods.common_methods import combine_mask_and_input_image, random_seed
from ui_components.methods.file_methods import (
    copy_local_file,
    normalize_size_internal_file_obj,
    save_or_host_file,
    zip_images,
    determine_dimensions_for_sdxl,
)
from ui_components.widgets.model_selector_element import SD3_MODEL_DOWNLOAD_LIST, SDXL_MODEL_DOWNLOAD_LIST
from ui_components.widgets.sm_animation_style_element import SD_MODEL_DICT
from utils.common_utils import padded_integer
from utils.constants import MLQueryObject, StabliseMotionOption
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow, MLModel
import json


MODEL_PATH_DICT = {
    ComfyWorkflow.SDXL: {
        "workflow_path": "comfy_workflows/sdxl_workflow_api.json",
        "output_node_id": [19],
    },
    ComfyWorkflow.FLUX: {
        "workflow_path": "comfy_workflows/flux_schnell_workflow_api.json",
        "output_node_id": [9],
    },
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
    ComfyWorkflow.IPADAPTER_COMPOSITION: {
        "workflow_path": "comfy_workflows/ipadapter_composition_workflow_api.json",
        "output_node_id": [27],
    },
    ComfyWorkflow.CREATIVE_IMAGE_GEN: {
        "workflow_path": "comfy_workflows/creative_image_gen.json",
        "output_node_id": [27],
    },
    ComfyWorkflow.SD3: {
        "workflow_path": "comfy_workflows/sd3_workflow_api.json",
        "output_node_id": [233],
    },
}


# these methods return the workflow along with the output node class name
class ComfyDataTransform:
    @staticmethod
    def get_workflow_json(model: ComfyWorkflow):
        json_file_path = "./utils/ml_processor/" + MODEL_PATH_DICT[model]["workflow_path"]
        # Specify encoding as 'utf-8' when opening the file
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            return json_data, MODEL_PATH_DICT[model]["output_node_id"]

    @staticmethod
    def transform_flux_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.FLUX)

        # workflow params (not all values are plugged in rn)
        model = query.data["data"].get("sdxl_model", None)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale

        # updating params
        workflow["6"]["inputs"]["text"] = positive_prompt
        if negative_prompt:
            workflow["33"]["inputs"]["text"] = negative_prompt
        workflow["31"]["inputs"]["seed"] = random_seed()

        extra_model_list = [
            {
                "filename": "flux1-schnell-fp8.safetensors",
                "url": "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors?download=true",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "checkpoints"),
            }
        ]

        return json.dumps(workflow), output_node_ids, extra_model_list, []

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

        workflow["4"]["inputs"]["ckpt_name"] = model["filename"]

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
        image = query.file_list[0]
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
        image = query.file_list[0]
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
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IPADAPTER_COMPOSITION)

        # workflow params
        width, height = query.width, query.height
        # width, height = determine_dimensions_for_sdxl(width, height)
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        # low_threshold, high_threshold = query.low_threshold, query.high_threshold
        image = query.file_list[0]
        image_name = image.filename

        # updating params
        workflow["9"]["inputs"]["seed"] = random_seed()
        workflow["10"]["width"], workflow["10"]["height"] = width, height
        workflow["7"]["inputs"]["text"], workflow["8"]["inputs"]["text"] = positive_prompt, negative_prompt
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
        image = query.file_list[0]
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
        model = query.data["data"].get("model", "Juggernaut-XL_v9_v2.safetensors")
        positive_prompt, negative_prompt = query.prompt, query.negative_prompt
        steps, cfg = query.num_inference_steps, query.guidance_scale
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)

        file = query.file_list[0]

        print(f"width: {width}, height: {height}")
        # Resize the input image
        resized_file = normalize_size_internal_file_obj(
            file,
            dim=[width, height],
            create_new_file=True,
        )

        # Update the query with the resized image
        query.data = {"data": {"file_combined_img": resized_file.uuid}}

        # updating params
        workflow["29"]["inputs"]["ckpt_name"] = model
        workflow["3"]["inputs"]["seed"] = random_seed()
        workflow["20"]["inputs"]["image"] = resized_file.filename
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

        extra_model_list = [
            {
                "filename": "Juggernaut-XL_v9_v2.safetensors",
                "url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "checkpoints"),
            }
        ]
        return json.dumps(workflow), output_node_ids, extra_model_list, []

    @staticmethod
    def transform_ipadaptor_plus_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_PLUS)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = query.file_list[0]
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
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = query.file_list[0]
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
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.IP_ADAPTER_FACE_PLUS)

        # workflow params
        model = query.data["data"].get("sdxl_model", None)
        width, height = query.width, query.height
        width, height = determine_dimensions_for_sdxl(width, height)
        steps, cfg = query.num_inference_steps, query.guidance_scale
        image = query.file_list[0]
        image_name = image.filename
        image_2 = query.file_list[1] if len(query.file_list) > 1 else None
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

        def update_json_with_styling_loras(json_data, loras):
            start_id = max(int(key) for key in json_data.keys()) + 1
            new_ids = []

            # Add LoRAs
            for lora in loras:
                new_id = str(start_id)
                json_data[new_id] = {
                    "inputs": {
                        "lora_name": lora["filename"],
                        "strength_model": lora["lora_strength"],
                        "model": ["547", 0],  # Initially connect to ADE_UseEvolvedSampling
                    },
                    "class_type": "LoraLoaderModelOnly",
                    "_meta": {"title": "Load Styling LoRA"},
                }

                if new_ids:
                    json_data[new_ids[-1]]["inputs"]["model"] = [new_id, 0]

                new_ids.append(new_id)
                start_id += 1

            # Update KSampler to use the last LoRA node if there are any new LoRAs
            if new_ids:
                json_data["207"]["inputs"]["model"] = [new_ids[-1], 0]

            return json_data

        def update_json_with_motion_loras(json_data, loras):
            start_id = 536
            new_ids = []

            # Add LoRAs
            for lora in loras:
                new_id = str(start_id)
                json_data[new_id] = {
                    "inputs": {
                        "name": lora["filename"],
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

        def allow_for_looping(workflow):
            # Remove nodes 614, 615, 616, 687, and 354
            nodes_to_remove = ["614", "615", "616", "687", "354"]
            for node in nodes_to_remove:
                if node in workflow:
                    del workflow[node]

            # Modify node 559 (FILM VFI) to connect directly to the KSampler output
            if "559" in workflow:
                workflow["559"]["inputs"]["frames"] = ["207", 5]

            return workflow

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
                json_data["593"]["inputs"]["ipa_ends_at"] = 0.3
                json_data["593"]["inputs"]["ipa_weight_type"] = "ease in-out"
                json_data["593"]["inputs"]["ipa_weight"] = 0.75
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

        workflow = update_json_with_motion_loras(workflow, sm_data.get("motion_lora_data"))
        workflow = update_json_with_styling_loras(workflow, sm_data.get("styling_lora_data"))

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

        # workflow["207"]["inputs"]["noise_seed"] = random_seed()

        workflow["541"]["inputs"]["pre_text"] = sm_data.get("prompt")
        workflow["541"]["inputs"]["text"] = sm_data.get("individual_prompts")
        workflow["541"]["inputs"]["max_frames"] = int(float(sm_data.get("max_frames")))

        workflow["543"]["inputs"]["pre_text"] = sm_data.get("negative_prompt")
        workflow["543"]["inputs"]["max_frames"] = int(float(sm_data.get("max_frames")))
        workflow["543"]["inputs"]["text"] = sm_data.get("individual_negative_prompts")

        workflow, extra_models_list = convert_to_specific_workflow(
            workflow,
            sm_data.get("type_of_generation", "Fast With A Price"),
            extra_models_list,
        )

        for v in SD_MODEL_DICT.values():
            if v["filename"] == ckpt:
                extra_models_list.append(v)

        # maps stablise motion values <-> sparse nonhint multiplier (for normal and lcm models)
        stablise_motion_value_map = {
            StabliseMotionOption.NONE.value: {"normal": 0.15, "lcm": 0.15},
            StabliseMotionOption.LOW.value: {"normal": 0.1, "lcm": 0.2},
            StabliseMotionOption.STANDARD.value: {"normal": 0.2, "lcm": 0.4},
            StabliseMotionOption.HIGH.value: {"normal": 0.3, "lcm": 0.6},
            StabliseMotionOption.VERY_HIGH.value: {"normal": 0.4, "lcm": 0.8},
        }

        workflow["467"]["inputs"]["context_aware"] = "nearest_hint"
        ad_mode = "lcm" if workflow["546"]["inputs"]["model_name"] == "AnimateLCM_sd15_t2v.ckpt" else "normal"
        workflow["467"]["inputs"]["sparse_nonhint_mult"] = stablise_motion_value_map[
            sm_data.get("stabilise_motion", StabliseMotionOption.NONE.value)
        ][ad_mode]

        ignore_list = sm_data.get("motion_lora_data", [])

        if sm_data.get("allow_for_looping", False):
            workflow = allow_for_looping(workflow)

        workflow["207"]["inputs"]["steps"] = sm_data.get("number_of_generation_steps")

        # with open("workflow.json", "w") as f:
        #     json.dump(workflow, f, indent=4)

        return json.dumps(workflow), output_node_ids, extra_models_list, ignore_list

    @staticmethod
    def transform_video_upscaler_workflow(query: MLQueryObject):
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.UPSCALER)
        data = query.data.get("data", {})
        video = query.file_list[0]
        model = data.get("model", None)

        upscale_factor = data.get("upscale_factor", None)

        workflow["302"]["inputs"]["video"] = os.path.basename(video.filename)
        workflow["362"]["inputs"]["ckpt_name"] = model
        workflow["391"]["inputs"]["upscale_by"] = upscale_factor
        workflow["391"]["inputs"]["seed"] = random_seed()

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
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.MOTION_LORA)
        data = query.data.get("data", {})
        video = query.file_list[0]
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
        combined_models = {**SDXL_MODEL_DOWNLOAD_LIST, **SD3_MODEL_DOWNLOAD_LIST}
        for v in combined_models.values():
            if v["filename"] == model:
                extra_model_list.append(v)

        return json.dumps(workflow), output_node_ids, extra_model_list, []

    @staticmethod
    def transform_creative_img_gen_workflow(query: MLQueryObject):
        data_repo = DataRepo()
        workflow, output_node_ids = ComfyDataTransform.get_workflow_json(ComfyWorkflow.CREATIVE_IMAGE_GEN)

        # workflow params
        query_data = query.data["data"]
        width, height = determine_dimensions_for_sdxl(
            query_data.get("width", 512), query_data.get("height", 512)
        )
        image_prompt, negative_prompt = query.prompt, query.negative_prompt
        lightning = query_data.get("lightning", False)
        additional_description_text = query_data.get("additional_description_text", "")
        additional_style_text = query_data.get("additional_style_text", "")
        model = query_data.get("sdxl_model", "sd_xl_base_1.0.safetensors")

        seed = random_seed()
        style_strength = query.strength

        # @Peter you can use this weight, passed from the frontend
        def add_nth_node(workflow, n, img_file, weight):
            style_influence, composition_influence, vibe_influence = map(float, weight[:3])

            ipa_node_idx_list = []
            for k, v in workflow.items():
                if v["class_type"] in ["IPAdapterMS", "IPAdapterAdvanced"]:
                    ipa_node_idx_list.append(int(k))
            ipa_node_idx_list.sort(reverse=True)

            node_idx = 50 + n * 10

            # Load Image
            workflow[str(node_idx)] = {
                "inputs": {"image": img_file.filename, "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"},
            }

            last_model_node = ipa_node_idx_list[0] if ipa_node_idx_list else 11
            original_model_node = last_model_node  # Store the original model node

            if style_influence > 0:
                # Prep Image For ClipVision
                workflow[str(node_idx + 1)] = {
                    "inputs": {
                        "interpolation": "LANCZOS",
                        "crop_position": "center",
                        "sharpening": 0,
                        "image": [str(node_idx), 0],
                    },
                    "class_type": "PrepImageForClipVision",
                    "_meta": {"title": "Prep Image For ClipVision"},
                }

                # IPAdapter Mad Scientist
                workflow[str(node_idx + 2)] = {
                    "inputs": {
                        "weight": style_influence,
                        "weight_faceidv2": 1,
                        "weight_type": "style transfer precise",
                        "combine_embeds": "concat",
                        "start_at": 0,
                        "end_at": 1,
                        "embeds_scaling": "V only",
                        "layer_weights": "3:2.5, 6:1",
                        "model": [str(last_model_node), 0],
                        "ipadapter": ["11", 1],
                        "image": [str(node_idx + 1), 0],
                    },
                    "class_type": "IPAdapterMS",
                    "_meta": {"title": "IPAdapter Mad Scientist"},
                }
                last_model_node = node_idx + 2

            if composition_influence > 0:
                # Load CLIP Vision
                workflow[str(node_idx + 3)] = {
                    "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
                    "class_type": "CLIPVisionLoader",
                    "_meta": {"title": "Load CLIP Vision"},
                }

                # IPAdapter Model Loader
                workflow[str(node_idx + 4)] = {
                    "inputs": {"ipadapter_file": "ip_plus_composition_sdxl.safetensors"},
                    "class_type": "IPAdapterModelLoader",
                    "_meta": {"title": "IPAdapter Model Loader"},
                }

                # Prep Image For ClipVision
                workflow[str(node_idx + 5)] = {
                    "inputs": {
                        "interpolation": "LANCZOS",
                        "crop_position": "pad",
                        "sharpening": 0,
                        "image": [str(node_idx), 0],
                    },
                    "class_type": "PrepImageForClipVision",
                    "_meta": {"title": "Prep Image For ClipVision"},
                }

                # IPAdapter Advanced
                workflow[str(node_idx + 6)] = {
                    "inputs": {
                        "weight": composition_influence,
                        "weight_type": "linear",
                        "combine_embeds": "concat",
                        "start_at": 0,
                        "end_at": 1,
                        "embeds_scaling": "V only",
                        "model": [str(last_model_node), 0],
                        "ipadapter": [str(node_idx + 4), 0],
                        "image": [str(node_idx + 5), 0],
                        "clip_vision": [str(node_idx + 3), 0],
                    },
                    "class_type": "IPAdapterAdvanced",
                    "_meta": {"title": "IPAdapter Advanced"},
                }
                last_model_node = node_idx + 6

            if vibe_influence > 0:
                # IPAdapter Noise (negative)
                workflow[str(node_idx + 5)] = {
                    "inputs": {
                        "type": "dissolve",
                        "strength": 0.7,
                        "blur": 0,
                        "image_optional": [str(node_idx), 0],
                    },
                    "class_type": "IPAdapterNoise",
                    "_meta": {"title": "IPAdapter Noise (negative)"},
                }

                # Prep Image For ClipVision (pad)
                workflow[str(node_idx + 6)] = {
                    "inputs": {
                        "interpolation": "LANCZOS",
                        "crop_position": "pad",
                        "sharpening": 0,
                        "image": [str(node_idx), 0],
                    },
                    "class_type": "PrepImageForClipVision",
                    "_meta": {"title": "Prep Image For ClipVision"},
                }

                # IPAdapter Advanced (linear)
                workflow[str(node_idx + 7)] = {
                    "inputs": {
                        "weight": vibe_influence,
                        "weight_type": "linear",
                        "combine_embeds": "concat",
                        "start_at": 0,
                        "end_at": 1,
                        "embeds_scaling": "V only",
                        "model": [str(original_model_node), 0],  # Use the original model node
                        "ipadapter": ["11", 1],
                        "image": [str(node_idx + 6), 0],
                        "image_negative": [str(node_idx + 5), 0],
                    },
                    "class_type": "IPAdapterAdvanced",
                    "_meta": {"title": "IPAdapter Advanced"},
                }
                last_model_node = node_idx + 7

            return int(last_model_node)

        def add_reference_images(workflow, img_list, weight, **kwargs):

            num_images = len(img_list)

            last_node_index = 4

            for i in range(num_images):
                last_node_index = add_nth_node(workflow, i + 1, img_list[i], weight[i])
                for k, v in kwargs.items():
                    if k in workflow[str(last_node_index)]["inputs"]:
                        workflow[str(last_node_index)]["inputs"][k] = v

            workflow["3"]["inputs"]["model"] = [str(last_node_index), 0]

            return workflow

        workflow["3"]["inputs"]["seed"] = seed
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["4"]["inputs"]["ckpt_name"] = model
        workflow["6"]["inputs"]["text"] = (
            image_prompt + ", " + additional_description_text + ", " + additional_style_text
        )

        workflow["7"]["inputs"]["text"] = negative_prompt

        if lightning:
            workflow["3"]["inputs"]["cfg"] = 1.9
            workflow["3"]["inputs"]["steps"] = 12
            workflow["3"]["inputs"]["scheduler"] = "sgm_uniform"

        img_list = query.file_list

        workflow = add_reference_images(workflow, img_list, weight=style_strength)

        combined_models = {**SDXL_MODEL_DOWNLOAD_LIST, **SD3_MODEL_DOWNLOAD_LIST}
        model_details = None
        for v in combined_models.values():
            if v["filename"] == model:
                model_details = v

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
            {
                "filename": "ip_plus_composition_sdxl.safetensors",
                "url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors",
                "dest": os.path.join(COMFY_BASE_PATH, "models", "ipadapter"),
            },
        ]

        if model_details:
            extra_model_list.append(model_details)

        return json.dumps(workflow), output_node_ids, extra_model_list, []


# NOTE: only populating with models currently in use
MODEL_WORKFLOW_MAP = {
    ML_MODEL.flux.workflow_name: ComfyDataTransform.transform_flux_workflow,
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
    ML_MODEL.sdxl_img2img.workflow_name: ComfyDataTransform.transform_sdxl_img2img_workflow,
    ML_MODEL.video_upscaler.workflow_name: ComfyDataTransform.transform_video_upscaler_workflow,
    ML_MODEL.motion_lora_trainer.workflow_name: ComfyDataTransform.transform_motion_lora_workflow,
    ML_MODEL.creative_image_gen.workflow_name: ComfyDataTransform.transform_creative_img_gen_workflow,
    ML_MODEL.sd3_local.workflow_name: ComfyDataTransform.transform_sd3_workflow,
}


# returns stringified json of the workflow
def get_model_workflow_from_query(model: MLModel, query_obj: MLQueryObject) -> str:
    global MODEL_WORKFLOW_MAP
    if model.workflow_name not in MODEL_WORKFLOW_MAP:
        app_logger.log(LoggingType.ERROR, f"model {model.workflow_name} not supported for local inference")
        print("available models: ", MODEL_WORKFLOW_MAP.keys())
        raise ValueError(f"Model {model.workflow_name} not supported for local inference")

    res = MODEL_WORKFLOW_MAP[model.workflow_name](query_obj)
    return (model.workflow_name.value,) + res


def get_workflow_json_url(workflow_json):
    from utils.ml_processor.ml_interface import get_ml_client

    ml_client = get_ml_client()
    temp_fd, temp_json_path = tempfile.mkstemp(suffix=".json")

    with open(temp_json_path, "w") as temp_json_file:
        temp_json_file.write(workflow_json)

    return ml_client.upload_training_data(temp_json_path, delete_after_upload=True)


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


def get_file_path_list(model: MLModel, query_obj: MLQueryObject):
    """
    file_path_list is required by comfy_runner to copy the provided files in the ComfyUI's
    input folder. The elements of file_path_list can be a string (path) or an object that defines
    the current path and also the destination path
    """
    data_repo = DataRepo()

    file_uuid_list = []
    file_uuid_dest_map = {}
    for _, v in query_obj.file_data.items():
        file_uuid_list.append(v["uuid"])
        file_uuid_dest_map[v["uuid"]] = v["dest"]

    file_list = data_repo.get_image_list_from_uuid_list(file_uuid_list)
    uuid_file_dict = {f.uuid: f for f in file_list}
    sorted_file_list = []
    for _, v in query_obj.file_data.items():
        sorted_file_list.append(uuid_file_dict[v["uuid"]])
    file_list = sorted_file_list

    models_using_sdxl = [
        ComfyWorkflow.SDXL.value,
        ComfyWorkflow.SDXL_IMG2IMG.value,
        ComfyWorkflow.SDXL_CONTROLNET.value,
        # ComfyWorkflow.SDXL_INPAINTING.value,
        ComfyWorkflow.IP_ADAPTER_FACE.value,
        ComfyWorkflow.IP_ADAPTER_FACE_PLUS.value,
        ComfyWorkflow.IP_ADAPTER_PLUS.value,
        # ComfyWorkflow.CREATIVE_IMAGE_GEN.value,
    ]

    # resizing the files to dimensions that work well with SDXL
    new_file_map = {}  # maps old_file_name : new_resized_file_name
    if model.display_name() in models_using_sdxl:

        res = []
        for file in file_list:
            new_width, new_height = determine_dimensions_for_sdxl(query_obj.width, query_obj.height)
            # although the new_file created using create_new_file has the same location as the original file, it is
            # scaled to the original resolution after inference save (so resize has no effect)
            new_file = normalize_size_internal_file_obj(
                file,
                dim=[new_width, new_height],
                create_new_file=True,
            )
            res.append(new_file)
            new_file_map[file.filename] = new_file.filename

        file_list = res

    file_path_list = []
    for idx, file in enumerate(file_list):
        _, filename = os.path.split(file.local_path)
        new_filename = (
            f"{padded_integer(idx+1)}_" + filename
            if model.display_name() == ComfyWorkflow.STEERABLE_MOTION.value
            else filename
        )
        if str(file.uuid) not in file_uuid_dest_map:
            file_path_list.append("videos/temp/" + new_filename)
        else:
            dest = (
                file_uuid_dest_map[str(file.uuid)].replace("input", "")
                if file_uuid_dest_map[str(file.uuid)].startswith("input")
                else file_uuid_dest_map[str(file.uuid)]
            )
            file_path_list.append(
                {
                    "filepath": "videos/temp/" + new_filename,
                    "dest_folder": dest,
                }
            )

        copy_local_file(
            filepath=file.local_path,
            destination_directory="videos/temp/",
            new_name=new_filename,
        )

    return file_path_list
