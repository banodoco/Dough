from collections import defaultdict
import json
import time
import ast
from typing import List
import streamlit as st
import re
import os
from PIL import Image
from shared.constants import (
    COMFY_BASE_PATH,
    GPU_INFERENCE_ENABLED_KEY,
    FileTransformationType,
    InferenceLogTag,
    InferenceParamType,
    InternalFileTag,
    InferenceParamType,
    InferenceStatus,
    InferenceType,
    STEERABLE_MOTION_WORKFLOWS,
    ConfigManager,
)
from ui_components.constants import CreativeProcessType, ShotMetaData
from ui_components.methods.animation_style_methods import get_generation_settings_from_log, load_shot_settings
from ui_components.methods.common_methods import promote_image_variant, promote_video_variant
from ui_components.methods.file_methods import add_file_to_shortlist, create_duplicate_file
from ui_components.methods.video_methods import sync_audio_and_duration, upscale_video
from ui_components.widgets.display_element import individual_video_display_element
from ui_components.widgets.shot_view import create_video_download_button
from ui_components.models import (
    InferenceLogObject,
    InternalAIModelObject,
    InternalFileObject,
    InternalShotObject,
)
from ui_components.widgets.add_key_frame_element import add_key_frame
from ui_components.widgets.sm_animation_style_element import SD_MODEL_DICT, video_shortlist_btn
from utils import st_memory
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL, ComfyWorkflow
from utils.state_refresh import refresh_app
from utils.common_utils import convert_timestamp_1, convert_timestamp_to_relative


config_manager = ConfigManager()
gpu_enabled = config_manager.get(GPU_INFERENCE_ENABLED_KEY, False)


# TODO: very inefficient operation.. add shot_id as a foreign in logs table for better search
def video_generation_counter(shot_uuid):
    data_repo = DataRepo()
    log_list, page_count = data_repo.get_all_inference_log_list(
        status_list=[InferenceStatus.IN_PROGRESS.value, InferenceStatus.QUEUED.value],
        data_per_page=1000,
        page=1,
    )
    log_list = log_list or []
    res = []
    for log in log_list:
        origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, None)
        if not origin_data:
            continue
        inference_type = origin_data.get("inference_type", "")
        if (
            inference_type == InferenceType.FRAME_INTERPOLATION.value
            and origin_data.get("shot_uuid", "") == shot_uuid
        ):
            res.append(log)
    if len(res) > 0:

        h1, _ = st.columns([1, 2])
        with h1:
            if len(res) == 1:
                st.info(f"{len(res)} video generation pending for this shot.")
            else:
                st.info(f"{len(res)} video generations pending for this shot.")


# TODO: very inefficient operation.. (maybe add source_entity_id ? as a foreign key)
# @Peter enter the video_uuid to get the count of upscales in progress (this works very similar to the mthod above)
def upscale_video_generation_counter(video_uuid):
    data_repo = DataRepo()
    log_list, page_count = data_repo.get_all_inference_log_list(
        status_list=[InferenceStatus.IN_PROGRESS.value, InferenceStatus.QUEUED.value],
        data_per_page=1000,
        page=1,
    )
    log_list = log_list or []
    res = []
    for log in log_list:
        relation_data = json.loads(log.input_params).get(InferenceParamType.FILE_RELATION_DATA.value, None)
        if relation_data:
            relation_data = json.loads(relation_data)
            if (
                relation_data[0]["id"] == str(video_uuid)
                and relation_data[0]["transformation_type"] == FileTransformationType.UPSCALE.value
            ):
                res.append(log)

    if len(res) > 0:
        h1, h2 = st.columns([1, 2])
        with h1:
            if len(res) == 1:
                st.info(f"{len(res)} upscale generation pending for this shot.")
            else:
                st.info(f"{len(res)} upscale generations pending for this shot.")


def variant_comparison_grid(ele_uuid, stage=CreativeProcessType.MOTION.value):
    """
    UI element which compares different variant of images/videos. For images ele_uuid has to be timing_uuid
    and for videos it has to be shot_uuid.
    """
    data_repo = DataRepo()

    timing_uuid, shot_uuid = None, None
    if stage == CreativeProcessType.MOTION.value:
        shot_uuid = ele_uuid
        shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
        variants: List[InternalFileObject] = shot.interpolated_clip_list
        timing_list = data_repo.get_timing_list_from_shot(shot.uuid)

    else:
        timing_uuid = ele_uuid
        timing = data_repo.get_timing_from_uuid(timing_uuid)
        variants = timing.alternative_images_list
        shot_uuid = timing.shot.uuid
        timing_list = ""

    upscale_in_progress_arr = []

    col1, col2, col3 = st.columns([1, 0.25, 0.5])
    if stage == CreativeProcessType.MOTION.value:
        # have a toggle for open details
        if len(variants):
            with col2:
                open_generaton_details = st_memory.toggle(
                    "Open generation details", key=f"open_details_{shot_uuid}", value=False
                )
            with col3:
                items_to_show = st_memory.slider(
                    "Items per page:",
                    key=f"items_per_page_{shot_uuid}",
                    value=3,
                    step=3,
                    min_value=3,
                    max_value=9,
                )
        else:
            items_to_show = 5
        num_columns = 3
        with col1:
            st.markdown(f"### 🎞️ '{shot.name}' options")
            st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")
            show_previews = st_memory.checkbox("Show previews", value=True, key=f"show_previews_{shot_uuid}")

        if not show_previews:
            variants = [v for v in variants if v.inference_log.generation_tag != "preview"]

        upscale_in_progress_arr = get_video_upscale_dict(shot.project.uuid)

    else:
        items_to_show = 5
        num_columns = 3

    # Updated logic for pagination
    total_items = (
        len(variants) if stage == CreativeProcessType.MOTION.value else len(variants) - 1
    )  # subtracting 1 in the case of images, as we show the main variant there as well
    num_pages = total_items // items_to_show + (total_items % items_to_show > 0)
    page = 1

    if num_pages > 1:
        if num_pages > 10:
            page = col3.number_input(
                f"Page (out of {num_pages}):", min_value=1, max_value=num_pages, value=1, step=1
            )
        else:
            page = col3.radio(
                f"Page (out of {num_pages}):", options=list(range(1, num_pages + 1)), horizontal=True
            )
    if stage == CreativeProcessType.MOTION.value:
        video_generation_counter(shot_uuid)
    if not len(variants):
        st.info("No options created yet.")
        st.markdown("***")

    else:
        st.markdown("***")

        cols = st.columns(num_columns)
        current_variant = -2  # rand value that won't be filtered in additional_variants
        cur_col = 0
        # ----------------------------- main variant (only showing for images) -----------------------
        if stage != CreativeProcessType.MOTION.value:
            current_variant = (
                shot.primary_interpolated_video_index
                if stage == CreativeProcessType.MOTION.value
                else int(timing.primary_variant_index)
            )

            with cols[cur_col]:
                # st.info(f"###### Variant #{current_variant + 1}")
                st.success("Main variant")

                st.image(variants[current_variant].location, use_column_width=True)
                # image_variant_details(variants[current_variant])
                cur_col += 1

        # ------------------------------------- additional variants on the current page -------------------------
        additional_variants = [idx for idx in range(len(variants) - 1, -1, -1) if idx != current_variant]
        page_start = (page - 1) * items_to_show
        page_end = page_start + items_to_show
        page_indices = additional_variants[page_start:page_end]

        for i, variant_index in enumerate(page_indices):
            with cols[cur_col]:
                h1, h2, h3 = st.columns([1, 1, 1])
                with h1:
                    if variants[variant_index].tag == InternalFileTag.SHORTLISTED_VIDEO.value:
                        st.success(f"###### Variant #{variant_index + 1}")
                    else:
                        st.info(f"###### Variant #{variant_index + 1}")
                with h2:
                    if stage != CreativeProcessType.MOTION.value:
                        if st.button(
                            f"Promote variant #{variant_index + 1}",
                            key=f"Promote Variant #{variant_index + 1} for {st.session_state['current_frame_index']}",
                            help="Promote this variant to the primary image",
                            use_container_width=True,
                        ):
                            # if stage == CreativeProcessType.MOTION.value:
                            #     promote_video_variant(shot.uuid, variants[variant_index].uuid)
                            promote_image_variant(timing.uuid, variant_index)
                            refresh_app()
                    elif (
                        variants[variant_index].inference_log.generation_tag
                        != InferenceLogTag.UPSCALED_VIDEO.value
                    ):
                        if variants[variant_index].tag == InternalFileTag.SHORTLISTED_VIDEO.value:
                            video_shortlist_btn(variants[variant_index].uuid, type="rempve_from_shortlist")
                        else:
                            video_shortlist_btn(variants[variant_index].uuid)
                with h3:
                    st.warning(convert_timestamp_to_relative(variants[variant_index].created_on))
                is_upscaled_variant = is_upscaled_video(variants[variant_index])
                if stage == CreativeProcessType.MOTION.value:
                    if variants[variant_index]:
                        individual_video_display_element(variants[variant_index], is_upscaled_variant)
                    else:
                        st.error("No video present")

                    additional_tags = (
                        ["Upscale In Progress"]
                        if str(variants[variant_index].uuid) in upscale_in_progress_arr
                        else []
                    )
                    displayed_tags = video_tag_element(variants[variant_index], additional_tags)

                    variant_inference_detail_element(
                        variants[variant_index],
                        stage,
                        shot_uuid,
                        tag="var_details",
                        open_generaton_details=open_generaton_details,
                    )

                    if gpu_enabled or "Upscaled Video" not in displayed_tags:
                        uspcale_expander_element([variants[variant_index].uuid])
                    create_video_download_button(variants[variant_index].location, ui_key="var_details")

                else:
                    if variants[variant_index]:
                        st.image(variants[variant_index].location, use_column_width=True)
                        image_variant_details(variants[variant_index])
                    else:
                        st.error("No image present")

            cur_col += 1

            # if there's only one item, show a line break
            if len(page_indices) == 1:
                st.markdown("***")
            if cur_col >= num_columns or i == len(page_indices) - 1 or len(page_indices) == i:
                cur_col = 0  # Reset the column counter
                st.markdown("***")  # Add markdown line
                cols = st.columns(num_columns)  # Prepare for the next row
                # Add markdown line if this is not the last variant in page_indices

        return (
            st.session_state[f"{shot_uuid}_selected_variant_log_uuid"]
            if f"{shot_uuid}_selected_variant_log_uuid" in st.session_state
            else None
        )


def get_video_upscale_dict(project_uuid):
    """
    returns a arr [uuid_1, uuid_2] of video uuids, for which upscale is in progress
    """
    data_repo = DataRepo()
    log_filter_data = {
        "project_id": project_uuid,
        "page": 1,
        "data_per_page": 1000,
        "status_list": [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value],
        "model_name_list": ["upscale"],
    }

    log_list, page_count = data_repo.get_all_inference_log_list(**log_filter_data)
    upscale_in_progress_arr = []
    if log_list:
        for log in log_list:
            relation_data = json.loads(log.input_params).get(
                InferenceParamType.FILE_RELATION_DATA.value, None
            )
            if relation_data:
                relation_data = json.loads(relation_data)
                if relation_data[0]["transformation_type"] == "upscale":
                    upscale_in_progress_arr.append(relation_data[0]["id"])

    return list(set(upscale_in_progress_arr))


def video_tag_element(video_file: InternalFileObject, additional_tags=[]):
    displayed_tags = []
    # additional_tags can also be provided, these are mostly generated in runtimes
    if additional_tags and len(additional_tags):
        for tag in additional_tags:
            st.info(tag)
            displayed_tags.append(tag)

    # there are two tags, one on video_file (mainly used for shortlisting/filtering)
    # the other is on the log, used to mark the process of generation (upscale/preview etc..)
    if video_file.inference_log.generation_tag:
        t = " ".join(video_file.inference_log.generation_tag.split("_")).title()
        st.info(t)
        displayed_tags.append(t)

    displayed_tags = list(set(displayed_tags))
    return displayed_tags


def uspcale_expander_element(
    ele_uuid_list,
    heading="Upscale settings",
    btn_text="Upscale",
    ui_key=None,
    default_expanded=False,
):
    ui_key = ui_key or ele_uuid_list[0]
    with st.expander(heading, expanded=default_expanded):
        if not (ele_uuid_list and len(ele_uuid_list)):
            st.info("No videos to upscale")
        else:
            (
                styling_model,
                upscale_factor,
                promote_to_main_variant,
            ) = upscale_settings(ui_key=ui_key)

            if st.button(
                btn_text,
                key=f"upscale_main_variant_{ui_key}",
                help="Upscale",
                use_container_width=True,
            ):
                for ele_uuid in ele_uuid_list:
                    upscale_video(
                        ele_uuid,
                        styling_model,
                        upscale_factor,
                        promote_to_main_variant,
                    )
                refresh_app()


def is_upscaled_video(variant: InternalFileObject):
    log: InferenceLogObject = variant.inference_log
    if log and log.generation_tag and log.generation_tag == InferenceLogTag.UPSCALED_VIDEO.value:
        return True
    return False


def image_variant_details(variant: InternalFileObject):
    with st.expander("Inference Details", expanded=False):
        if variant.inference_params and "query_dict" in variant.inference_params:
            query_dict = (
                json.loads(variant.inference_params["query_dict"])
                if isinstance(variant.inference_params["query_dict"], str)
                else variant.inference_params["query_dict"]
            )
            st.markdown(f"Prompt:  {query_dict['prompt']}", unsafe_allow_html=True)
            st.markdown(f"Negative Prompt: {query_dict['negative_prompt']}", unsafe_allow_html=True)
            if "width" in query_dict:
                st.markdown(
                    f"Dimension: {query_dict['width']}x{query_dict['height']}", unsafe_allow_html=True
                )
            if "guidance_scale" in query_dict:
                st.markdown(f"Guidance scale: {query_dict['guidance_scale']}", unsafe_allow_html=True)
            model_name = variant.inference_log.model_name
            st.markdown(f"Model name: {model_name}", unsafe_allow_html=True)
            if model_name in []:
                st.markdown(f"Low threshold: {query_dict['low_threshold']}", unsafe_allow_html=True)
                st.markdown(f"High threshold: {query_dict['high_threshold']}", unsafe_allow_html=True)
            if model_name in [
                ML_MODEL.sdxl_img2img.display_name(),
                ML_MODEL.sdxl_controlnet.display_name(),
                ML_MODEL.ipadapter_face.display_name(),
                ML_MODEL.ipadapter_plus.display_name(),
            ]:
                s = query_dict["strength"]
                st.markdown(f"Strength: {s if s > 1 and s <= 100 else int(s * 100)}", unsafe_allow_html=True)
            if model_name in [ML_MODEL.ipadapter_face_plus.display_name()]:
                s = query_dict["strength"]
                st.markdown(
                    f"Face Img Strength: {s[0] if s[0] > 1 and s[0] <= 100 else int(s[0] * 100)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Plus Img Strength: {s[1] if s[1] > 1 and s[1] <= 100 else int(s[1] * 100)}",
                    unsafe_allow_html=True,
                )


def variant_inference_detail_element(
    variant: InternalFileObject,
    stage,
    shot_uuid,
    tag="temp",
    open_generaton_details=False,
):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    if stage == CreativeProcessType.MOTION.value:

        with st.expander("Settings", expanded=open_generaton_details):
            shot_meta_data, data_type = get_generation_settings_from_log(variant.inference_log.uuid)
            if shot_meta_data and shot_meta_data.get("main_setting_data", None):
                # ---------- main settings data ------------------
                for k, v in shot_meta_data.get("main_setting_data", {}).items():
                    # Custom title formatting based on the key
                    if k.startswith("strength_of_adherence_value"):
                        title = "**Strength of adherence:**"
                    elif k.startswith("type_of_motion_context_index"):
                        title = "**Type of motion context:**"
                    elif k.startswith("ckpt"):
                        title = "**Model:**"
                    elif k.startswith("stabilise_motion"):
                        title = "**Stablise Motion:**"
                    elif k.startswith("high_detail_mode_val"):
                        continue  # Skip displaying this key
                    else:
                        title = f"**{k.split(str(shot.uuid))[0][:-1]}:**"

                    # Check if the key starts with 'lora_data'
                    if k.startswith("lora_data"):
                        if isinstance(v, list) and len(v) > 0:
                            lora_items = [
                                f"- {item.get('filename', 'No filename')} - {item.get('lora_strength', 'No strength')} strength"
                                for item in v
                            ]
                            lora_data_formatted = "\n".join(lora_items)
                            st.markdown(f"{title} \n{lora_data_formatted}", unsafe_allow_html=True)

                    elif k.startswith("type_of_generation_index"):
                        if v is not None:
                            # Assuming 'v' is the order number you are looking for
                            workflow_name = next(
                                (
                                    workflow["name"]
                                    for workflow in STEERABLE_MOTION_WORKFLOWS
                                    if workflow["order"] == v
                                ),
                                None,
                            )

                            if workflow_name:
                                st.markdown(f"**Workflow:** {workflow_name}", unsafe_allow_html=True)
                            else:
                                st.error("Invalid workflow order")
                    else:
                        if v:  # Check if v is not empty or None
                            st.markdown(f"{title} {v}", unsafe_allow_html=True)
                        else:
                            # Optionally handle empty or None values differently here
                            pass

                # ------------ individual frame settings --------------------
                timing_data = shot_meta_data.get("timing_data", [])
                display_dict = defaultdict(list)
                for idx in range(len(timing_data)):
                    if timing_data and len(timing_data) >= idx + 1:
                        motion_data = timing_data[idx]

                    for k, v in motion_data.items():
                        if v == "":
                            v = "__"
                        display_dict[k].append(v)

                are_all_elements_similar = lambda arr: (
                    True if not arr else all(element == arr[0] for element in arr[1:])
                )
                for k, v in display_dict.items():
                    k = k.replace("_", " ").title()
                    if are_all_elements_similar(v):
                        v = f"{v[0]} (for all frames)"
                    else:
                        if (
                            k.startswith("Distance To Next Frame")
                            or k.startswith("Speed Of Transition")
                            or k.startswith("Freedom Between Frames")
                        ):
                            v = v[:-1]  # removing the last ele in these cases
                        v = ", ".join(str(e) for e in v)
                    st.write(f"**{k}**: {v}")

                st.write(f"**Created On**: ", convert_timestamp_1(variant.created_on))

                btn1, btn2 = st.columns([1, 1])
                with btn1:
                    if st.button(
                        "Load settings",
                        key=f"boot_{tag}_{variant.name}",
                        help="This will load all the settings for this run below. In doing so, it'll remove the current settings and images - though they'll be available for all previous runs.",
                        use_container_width=True,
                        type="primary",
                    ):
                        load_shot_settings(
                            shot_uuid, variant.inference_log.uuid, load_images=False, load_setting_values=True
                        )
                        st.success("Settings Loaded")
                        time.sleep(0.3)
                        refresh_app()

                with btn2:
                    if st.button(
                        "Load images",
                        key=f"load_img_{tag}_{variant.name}",
                        help="This will load all the images for this run below. In doing so, it'll remove the current images and images - though they'll be available for all previous runs.",
                        use_container_width=True,
                    ):
                        load_shot_settings(
                            shot_uuid, variant.inference_log.uuid, load_images=True, load_setting_values=False
                        )
                        st.success("Images Loaded")
                        time.sleep(0.3)
                        refresh_app()

                if "sidebar_variant" not in st.session_state:
                    st.session_state["sidebar_variant"] = []

                if variant in st.session_state["sidebar_variant"]:

                    def remove_from_sidebar_button(variant):
                        st.session_state["sidebar_variant"].remove(variant)

                    if st.button(
                        "Remove from sidebar",
                        key=f"remove_from_sidebar_{variant.uuid}",
                        use_container_width=True,
                        on_click=remove_from_sidebar_button,
                        args=(variant,),
                    ):
                        refresh_app()

                else:

                    def add_to_sidebar_button(variant):
                        st.session_state["sidebar_variant"].append(variant)

                    if st.button(
                        "View in sidebar",
                        key=f"view_in_sidebar_{variant.uuid}",
                        use_container_width=True,
                        on_click=add_to_sidebar_button,
                        args=(variant,),
                    ):
                        refresh_app()

    else:
        h1, h2 = st.columns([1, 1])
        with h1:
            st.markdown(f"Add to shortlist:")
            add_variant_to_shortlist_element(variant, shot.project.uuid)
        with h2:
            add_variant_to_shot_element(variant, shot.project.uuid)


def prepare_values(inf_data, timing_list):
    settings = inf_data  # Map interpolation_type to indices
    interpolation_style_map = {"ease-in-out": 0, "ease-in": 1, "ease-out": 2, "linear": 3}

    values = {
        "type_of_frame_distribution": 1 if settings.get("type_of_frame_distribution") == "dynamic" else 0,
        "linear_frame_distribution_value": settings.get("linear_frame_distribution_value", None),
        "type_of_key_frame_influence": 1 if settings.get("type_of_key_frame_influence") == "dynamic" else 0,
        "length_of_key_frame_influence": (
            float(settings.get("linear_key_frame_influence_value"))
            if settings.get("linear_key_frame_influence_value")
            else None
        ),
        "type_of_cn_strength_distribution": (
            1 if settings.get("type_of_cn_strength_distribution") == "dynamic" else 0
        ),
        "linear_cn_strength_value": (
            tuple(map(float, ast.literal_eval(settings.get("linear_cn_strength_value"))))
            if settings.get("linear_cn_strength_value")
            else None
        ),
        "interpolation_style": (
            interpolation_style_map[settings.get("interpolation_type")]
            if settings.get("interpolation_type", "ease-in-out") in interpolation_style_map
            else None
        ),
        "motion_scale": settings.get("motion_scale", None),
        "negative_prompt_video": settings.get("negative_prompt", None),
        "relative_ipadapter_strength": settings.get("relative_ipadapter_strength", None),
        "relative_ipadapter_influence": settings.get("relative_ipadapter_influence", None),
        "soft_scaled_cn_weights_multiple_video": settings.get("soft_scaled_cn_weights_multiplier", None),
    }

    # Add dynamic values
    dynamic_frame_distribution_values = (
        settings["dynamic_frame_distribution_values"].split(",")
        if settings["dynamic_frame_distribution_values"]
        else []
    )
    dynamic_key_frame_influence_values = (
        settings["dynamic_key_frame_influence_values"].split(",")
        if settings["dynamic_key_frame_influence_values"]
        else []
    )
    dynamic_cn_strength_values = (
        settings["dynamic_cn_strength_values"].split(",") if settings["dynamic_cn_strength_values"] else []
    )

    min_length = len(timing_list) if timing_list else 0

    for idx in range(min_length):

        # Process dynamic_frame_distribution_values
        if dynamic_frame_distribution_values:
            values[f"dynamic_frame_distribution_values_{idx}"] = (
                int(dynamic_frame_distribution_values[idx])
                if dynamic_frame_distribution_values[idx] and dynamic_frame_distribution_values[idx].strip()
                else None
            )
        # Process dynamic_key_frame_influence_values
        if dynamic_key_frame_influence_values:
            values[f"dynamic_key_frame_influence_values_{idx}"] = (
                float(dynamic_key_frame_influence_values[idx])
                if dynamic_key_frame_influence_values[idx] and dynamic_key_frame_influence_values[idx].strip()
                else None
            )

        # Process dynamic_cn_strength_values
        if dynamic_cn_strength_values and idx * 2 <= len(dynamic_cn_strength_values):
            # Since idx starts from 1, we need to adjust the index for zero-based indexing
            adjusted_idx = idx * 2
            # Extract the two elements that form a tuple
            first_value = dynamic_cn_strength_values[adjusted_idx].strip("(")
            second_value = dynamic_cn_strength_values[adjusted_idx + 1].strip(")")
            # Convert both strings to floats and create a tuple
            value_tuple = (float(first_value), float(second_value))
            # Store the tuple in the dictionary with a key indicating its order
            values[f"dynamic_cn_strength_values_{idx}"] = value_tuple

    return values


def upscale_settings(ui_key):
    if gpu_enabled:
        checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
        all_files = os.listdir(checkpoints_dir)
    else:
        all_files = list(SD_MODEL_DICT.keys())

    if len(all_files) == 0:
        st.info("No models found in the checkpoints directory")
        styling_model = "None"
    else:
        # Filter files to only include those with .safetensors and .ckpt extensions
        model_files = [file for file in all_files if file.endswith(".safetensors") or file.endswith(".ckpt")]
        # drop all files that contain xl
        model_files = [file for file in model_files if "xl" not in file and "sd3" not in file]
        # model_files.insert(0, "None")  # Add "None" option at the beginning
        styling_model = st.selectbox("Styling model", model_files, key=f"styling_model_{ui_key}")

    if gpu_enabled:
        upscale_by = st.slider(
            "Upscale by:", min_value=1.25, max_value=3.0, step=0.05, key=f"upscale_by_{ui_key}", value=1.5
        )
    else:
        upscale_by = 1.5

    return styling_model, upscale_by, True


def fetch_inference_data(file: InternalFileObject):
    if not file:
        return

    not_found_msg = "No data available."
    inf_data = None
    # NOTE: generated videos also have other params stored inside origin_data > settings
    if file.inference_log and file.inference_log.input_params:
        inf_data = json.loads(file.inference_log.input_params)
        if "origin_data" in inf_data and inf_data["origin_data"]["inference_type"] == "frame_interpolation":
            inf_data = inf_data["origin_data"]["settings"]
        else:
            for data_type in InferenceParamType.value_list():
                if data_type in inf_data:
                    del inf_data[data_type]

    inf_data = inf_data or not_found_msg

    return inf_data


def add_variant_to_shortlist_element(file: InternalFileObject, project_uuid):

    if st.button(
        "Add to shortlist ➕",
        key=f"shortlist_{file.uuid}",
        use_container_width=True,
        help="Add to shortlist",
    ):
        add_file_to_shortlist(file.uuid, project_uuid)


def add_variant_to_shot_element(file: InternalFileObject, project_uuid):
    data_repo = DataRepo()

    shot_list = data_repo.get_shot_list(project_uuid)
    shot_names = [s.name for s in shot_list]

    shot_name = st.selectbox("Add to shot:", shot_names, key=f"current_shot_variant_{file.uuid}")
    if shot_name:
        if st.button(
            f"Add to shot",
            key=f"add_{file.uuid}",
            help="Promote this variant to the primary image",
            use_container_width=True,
        ):
            shot_number = shot_names.index(shot_name)
            shot_uuid = shot_list[shot_number].uuid

            duplicate_file = create_duplicate_file(file, project_uuid)
            add_key_frame(
                duplicate_file,
                shot_uuid,
                len(data_repo.get_timing_list_from_shot(shot_uuid)),
                refresh_state=False,
                update_cur_frame_idx=False,
            )
            refresh_app()
