import json
import os
import streamlit as st
from ui_components.constants import GalleryImageViewType
from ui_components.methods.animation_style_methods import toggle_generate_inference
from ui_components.methods.common_methods import (
    get_canny_img,
    process_inference_output,
    add_new_shot,
    save_new_image,
)
from ui_components.methods.file_methods import add_file_to_shortlist, zoom_and_crop
from ui_components.widgets.add_key_frame_element import add_key_frame
from ui_components.widgets.inpainting_element import inpainting_image_input
from ui_components.widgets.inspiration_engine import inspiration_engine_element
from ui_components.widgets.model_selector_element import model_selector_element
from utils.constants import MLQueryObject, T2IModel
from utils.state_refresh import refresh_app
from utils.data_repo.data_repo import DataRepo
from shared.constants import (
    COMFY_BASE_PATH,
    GPU_INFERENCE_ENABLED,
    QUEUE_INFERENCE_QUERIES,
    AIModelType,
    InferenceType,
    InternalFileTag,
    InternalFileType,
    SortOrder,
)
from utils import st_memory
import time
from utils.encryption import Encryptor
from utils.enum import ExtendedEnum
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL
import numpy as np
from PIL import Image
from utils import st_memory
from utils.ml_processor.sai.sai import StabilityProcessor


class InputImageStyling(ExtendedEnum):
    TEXT2IMAGE = "Text to Image"
    IMAGE2IMAGE = "Image to Image"
    # CONTROLNET_CANNY = "ControlNet Canny"
    IPADAPTER_COMPOSITION = "IP-Adapter Composition"
    IPADAPTER_FACE = "IP-Adapter Face"
    IPADAPTER_PLUS = "IP-Adapter Plus"
    IPADPTER_FACE_AND_PLUS = "IP-Adapter Face & Plus"
    INPAINTING = "Inpainting"


def explorer_page(project_uuid):
    st.markdown(f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}]")
    st.markdown("***")

    with st.expander("✨ Generate Images", expanded=True):
        inspiration_engine_element(
            position="explorer", project_uuid=project_uuid, timing_uuid=None, shot_uuid=None
        )
    st.markdown("***")

    gallery_image_view(
        project_uuid, False, view=["add_and_remove_from_shortlist", "view_inference_details", "shot_chooser"]
    )


def generate_images_element(position="explorer", project_uuid=None, timing_uuid=None, shot_uuid=None):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)
    help_input = '''This will generate a specific prompt based on your input.\n\n For example, "Sad scene of old Russian man, dreary style" might result in "Boris Karloff, 80 year old man wearing a suit, standing at funeral, dark blue watercolour."'''
    if shot_uuid:
        prompt_key = f"prompt_{shot_uuid}"
        negative_prompt_key = f"negative_prompt_{shot_uuid}"
    else:
        prompt_key = f"explorer_base_prompt"
        negative_prompt_key = f"explorer_base_negative_prompt"

    a1, a2 = st.columns([1, 1])

    with a1:
        prompt = st_memory.text_area(
            "Prompt:", key=prompt_key, help="This exact text will be included for each generation."
        )
    # with a2 if 'switch_prompt_position' not in st.session_state or st.session_state['switch_prompt_position'] == False else a1:
    with a2:
        negative_prompt = st_memory.text_area(
            "Negative prompt:",
            value="",
            key=negative_prompt_key,
            help="These are the things you wish to be excluded from the image",
        )

    type_of_generation = st_memory.radio(
        "Type of generation:",
        options=InputImageStyling.value_list(),
        key="type_of_generation_key",
        help="Evolve Image will evolve the image based on the prompt, while Maintain Structure will keep the structure of the image and change the style.",
        horizontal=True,
    )

    # --------------------- taking image inputs --------------------------------
    if type_of_generation != InputImageStyling.TEXT2IMAGE.value:
        if "input_image_1" not in st.session_state:
            st.session_state["input_image_1"] = None
            st.session_state["input_image_2"] = None

        # these require two images
        if type_of_generation == InputImageStyling.INPAINTING.value:
            inpainting_image_input(project_uuid)

        else:

            def handle_image_input(
                column, type_of_generation, output_value_name, data_repo=None, project_uuid=None
            ):
                with column:
                    if st.session_state.get(output_value_name) is None:
                        top0, top1, top2, top3 = st.columns([0.4, 1, 1, 0.4])
                        with top1:
                            st.info(
                                f"{type_of_generation} input:"
                            )  # Dynamic title based on type_of_generation
                        with top2:
                            source_of_starting_image = st.radio(
                                "Image source:",
                                options=["Upload", "From Shot"],
                                key=f"{output_value_name}_starting_image",
                                help="This will be the base image for the generation.",
                                horizontal=True,
                            )

                        if "uploaded_image" not in st.session_state:
                            st.session_state["uploaded_image"] = None

                        if f"uploaded_image_{output_value_name}" not in st.session_state:
                            st.session_state[f"uploaded_image_{output_value_name}"] = f"0_{output_value_name}"

                        if source_of_starting_image == "Upload":
                            uploaded_image = st.file_uploader(
                                "Upload a starting image",
                                type=["png", "jpg", "jpeg", "webp"],
                                key=st.session_state[f"uploaded_image_{output_value_name}"],
                                help="This will be the base image for the generation.",
                            )
                            if uploaded_image:
                                uploaded_image = (
                                    Image.open(uploaded_image)
                                    if not isinstance(uploaded_image, Image.Image)
                                    else uploaded_image
                                )
                                uploaded_image = zoom_and_crop(
                                    uploaded_image, project_settings.width, project_settings.height
                                )

                            st.session_state["uploaded_image"] = uploaded_image

                        else:
                            # taking image from shots
                            shot_list = data_repo.get_shot_list(project_uuid)
                            selection1, selection2 = st.columns([1, 1])
                            with selection1:
                                shot_name = st.selectbox(
                                    "Shot:",
                                    options=[shot.name for shot in shot_list],
                                    key=f"{output_value_name}_shot_name",
                                    help="This will be the base image for the generation.",
                                )
                                shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                                frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                                list_of_timings = [i + 1 for i in range(len(frame_list))]
                                timing = st.selectbox(
                                    "Frame #:",
                                    options=list_of_timings,
                                    key=f"{output_value_name}_frame_number",
                                    help="This will be the base image for the generation.",
                                )
                                if timing:
                                    st.session_state["uploaded_image"] = frame_list[
                                        timing - 1
                                    ].primary_image.location
                            with selection2:
                                if frame_list and len(frame_list) and timing:
                                    st.image(
                                        frame_list[timing - 1].primary_image.location, use_column_width=True
                                    )

                        # Trigger image processing
                        if st.button(
                            "Upload Image", key=f"{output_value_name}_upload_button", use_container_width=True
                        ):
                            st.session_state[output_value_name] = st.session_state["uploaded_image"]
                            # st.session_state[f"uploaded_image_{output_value_name}"] += 1
                            refresh_app()

                        return None
                    else:
                        # Display current image
                        st.info(f"{type_of_generation} image:")
                        half1, half2 = st.columns([1, 1])
                        with half1:
                            st.image(st.session_state[output_value_name], use_column_width=True)
                        # Slider for image adjustment based on type_of_generation
                        with half2:
                            strength_of_image = st.slider(
                                "Image strength:",
                                min_value=0,
                                max_value=100,
                                value=50,
                                step=1,
                                key=f"{output_value_name}_strength",
                                help="This will be the strength of the image for the generation.",
                            )

                            if st.button("Clear image", key=f"{output_value_name}_clear_button"):
                                st.session_state[output_value_name] = None
                                refresh_app()

                        return strength_of_image

            sub0, sub1, sub2, sub3 = st.columns([0.3, 1, 1, 0.3])
            if type_of_generation != InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                strength_of_image = handle_image_input(
                    sub1, type_of_generation, "input_image_1", data_repo, project_uuid
                )
            else:
                strength_of_image_1 = handle_image_input(
                    sub1, "IP-Adapter Face", "input_image_1", data_repo, project_uuid
                )
                strength_of_image_2 = handle_image_input(
                    sub2, "IP-Adapter Plus", "input_image_2", data_repo, project_uuid
                )
                strength_of_image = (strength_of_image_1, strength_of_image_2)
                # if both images are populated, create a "switch images" button
                with sub2:
                    corner1, corner2 = st.columns([1, 1])
                    with corner2:
                        if st.session_state["input_image_1"] and st.session_state["input_image_2"]:
                            if st.button("Switch images 🔄", key="switch_images", use_container_width=True):
                                st.session_state["input_image_1"], st.session_state["input_image_2"] = (
                                    st.session_state["input_image_2"],
                                    st.session_state["input_image_1"],
                                )
                                refresh_app()

        if type_of_generation != InputImageStyling.IPADAPTER_COMPOSITION.value:
            explorer_gen_model = model_selector_element()
    else:
        t2i_1, t2i_2 = st.columns([1, 1])
        with t2i_1:
            t2i_model = st_memory.radio(
                "Select Model:", options=T2IModel.value_list(), index=0, key="t2i_model", horizontal=True
            )

        if t2i_model == T2IModel.SD3.value:
            if not st.session_state.get("stability_key", None):
                app_secrets = data_repo.get_app_secrets_from_user_uuid()
                if "stability_key" in app_secrets and app_secrets["stability_key"]:
                    st.session_state["stability_key"] = app_secrets["stability_key"]
                else:
                    st.warning(
                        "You need to enter a Stability API key to use SD3 right now - please go to App Settings."
                    )
            else:
                with t2i_2:
                    st.info("Stability API will be used for this generation")

        if t2i_model == T2IModel.SDXL.value:
            explorer_gen_model = model_selector_element()

    if position == "explorer":
        _, d2, d3, _ = st.columns([0.25, 1, 1, 0.25])
    else:
        d2, d3 = st.columns([1, 1])
    with d2:
        number_to_generate = st.slider(
            "Number of images to generate:",
            min_value=1,
            max_value=36,
            value=1,
            step=4,
            key="number_to_generate",
            help="It'll generate 4 from each variation.",
        )

    with d3:
        st.write(" ")
        # ------------------- Generating output -------------------------------------
        if st.session_state.get(position + "_generate_inference"):
            ml_client = get_ml_client()

            for _ in range(number_to_generate):
                log = None
                generation_method = InputImageStyling.value_list()[st.session_state["type_of_generation_key"]]
                if generation_method == InputImageStyling.TEXT2IMAGE.value:
                    if t2i_model == T2IModel.SDXL.value:
                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            guidance_scale=8,
                            seed=-1,
                            num_inference_steps=25,
                            strength=0.5,
                            adapter_type=None,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            data={
                                "shot_uuid": shot_uuid,
                                "sdxl_model": explorer_gen_model,
                            },
                            file_data={},
                        )

                        output, log = ml_client.predict_model_output_standardized(
                            ML_MODEL.sdxl, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                        )
                    else:
                        # query for SD3
                        encryptor = Encryptor()
                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            guidance_scale=8,
                            seed=-1,
                            num_inference_steps=25,
                            strength=0.5,
                            adapter_type=None,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            data={
                                "shot_uuid": shot_uuid,
                                "stability_key": encryptor.encrypt_json(st.session_state["stability_key"]),
                            },
                            file_data={},
                        )

                        sai_client = StabilityProcessor()
                        output, log = sai_client.predict_model_output_standardized(
                            ML_MODEL.sd3, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                        )

                elif generation_method == InputImageStyling.IMAGE2IMAGE.value:
                    input_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        data={
                            "shot_uuid": shot_uuid,
                            "sdxl_model": explorer_gen_model,
                        },
                        file_data={
                            "image_1": {"uuid": input_image_file.uuid, "dest": "input/"},
                        },
                    )

                    output, log = ml_client.predict_model_output_standardized(
                        ML_MODEL.sdxl_img2img, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                    )

                elif generation_method == InputImageStyling.IPADAPTER_COMPOSITION.value:
                    input_img = st.session_state["input_image_1"]
                    input_image_file = save_new_image(input_img, project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image / 100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        data={"condition_scale": 1, "shot_uuid": shot_uuid},
                        file_data={"image_1": {"uuid": input_image_file.uuid, "dest": "input/"}},
                    )

                    output, log = ml_client.predict_model_output_standardized(
                        ML_MODEL.ipadapter_composition, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                    )

                # elif generation_method == InputImageStyling.CONTROLNET_CANNY.value:
                #     edge_pil_img = get_canny_img(st.session_state["input_image_1"], low_threshold=50, high_threshold=150)    # redundant incase of local inference
                #     input_img = edge_pil_img if not GPU_INFERENCE_ENABLED else st.session_state["input_image_1"]
                #     input_image_file = save_new_image(input_img, project_uuid)
                #     query_obj = MLQueryObject(
                #         timing_uuid=None,
                #         model_uuid=None,
                #         image_uuid=input_image_file.uuid,
                #         guidance_scale=5,
                #         seed=-1,
                #         num_inference_steps=30,
                #         strength=strength_of_image/100,
                #         adapter_type=None,
                #         prompt=prompt,
                #         negative_prompt=negative_prompt,
                #         height=project_settings.height,
                #         width=project_settings.width,
                #         data={'condition_scale': 1, "shot_uuid": shot_uuid}
                #     )

                #     output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl_controlnet, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                elif generation_method == InputImageStyling.IPADAPTER_FACE.value:
                    # validation
                    if not (st.session_state["input_image_1"]):
                        st.error("Please upload an image")
                        return

                    input_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image / 100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        data={
                            "shot_uuid": shot_uuid,
                            "sdxl_model": explorer_gen_model,
                        },
                        file_data={
                            "image_1": {"uuid": input_image_file.uuid, "dest": "input/"},
                        },
                    )

                    output, log = ml_client.predict_model_output_standardized(
                        ML_MODEL.ipadapter_face, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                    )

                elif generation_method == InputImageStyling.IPADAPTER_PLUS.value:
                    input_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image / 100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        data={
                            "condition_scale": 1,
                            "shot_uuid": shot_uuid,
                            "sdxl_model": explorer_gen_model,
                        },
                        file_data={
                            "image_1": {"uuid": input_image_file.uuid, "dest": "input/"},
                        },
                    )

                    output, log = ml_client.predict_model_output_standardized(
                        ML_MODEL.ipadapter_plus, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                    )

                elif generation_method == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                    # validation
                    if not (st.session_state["input_image_2"] and st.session_state["input_image_1"]):
                        st.error("Please upload both images")
                        return

                    plus_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    face_image_file = save_new_image(st.session_state["input_image_2"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=(strength_of_image_1 / 100, strength_of_image_2 / 100),  # (face, plus)
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        data={
                            "shot_uuid": shot_uuid,
                            "sdxl_model": explorer_gen_model,
                        },
                        file_data={
                            "image_1": {"uuid": plus_image_file.uuid, "dest": "input/"},
                            "image_2": {"uuid": face_image_file.uuid, "dest": "input/"},
                        },
                    )

                    output, log = ml_client.predict_model_output_standardized(
                        ML_MODEL.ipadapter_face_plus, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                    )

                elif generation_method == InputImageStyling.INPAINTING.value:
                    if not ("mask_to_use" in st.session_state and st.session_state["mask_to_use"]):
                        st.error("Please create and save mask before generation")
                        toggle_generate_inference(position)
                        time.sleep(0.7)
                        return

                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        guidance_scale=6,
                        seed=-1,
                        num_inference_steps=25,
                        strength=0.5,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        data={
                            "shot_uuid": shot_uuid,
                            "mask": st.session_state["mask_to_use"],
                            "input_image": st.session_state["editing_image"],
                            "project_uuid": project_uuid,
                            "sdxl_model": explorer_gen_model,
                        },
                    )

                    output, log = ml_client.predict_model_output_standardized(
                        ML_MODEL.sdxl_inpainting, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES
                    )

                if log:
                    inference_data = {
                        "inference_type": (
                            InferenceType.GALLERY_IMAGE_GENERATION.value
                            if position == "explorer"
                            else InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value
                        ),
                        "output": output,
                        "log_uuid": log.uuid,
                        "project_uuid": project_uuid,
                        "timing_uuid": timing_uuid,
                        "promote_new_generation": False,
                        "shot_uuid": shot_uuid if shot_uuid else "explorer",
                    }

                    process_inference_output(**inference_data)

            st.info("Check the Generation Log to the left for the status.")
            time.sleep(0.5)
            toggle_generate_inference(position)
            refresh_app()

        # ----------- generate btn --------------
        if prompt == "":
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please enter a prompt to generate images",
            )
        elif (
            type_of_generation == InputImageStyling.IMAGE2IMAGE.value
            and st.session_state["input_image_1"] is None
        ):
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please upload an image",
            )
        elif (
            type_of_generation == InputImageStyling.IPADAPTER_COMPOSITION.value
            and st.session_state["input_image_1"] is None
        ):
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please upload an image",
            )
        elif (
            type_of_generation == InputImageStyling.IPADAPTER_FACE.value
            and st.session_state["input_image_1"] is None
        ):
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please upload an image",
            )
        elif (
            type_of_generation == InputImageStyling.IPADAPTER_PLUS.value
            and st.session_state["input_image_1"] is None
        ):
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please upload an image",
            )
        elif type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value and (
            st.session_state["input_image_1"] is None or st.session_state["input_image_2"] is None
        ):
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please upload both images",
            )
        elif type_of_generation == InputImageStyling.INPAINTING.value and not (
            "mask_to_use" in st.session_state and st.session_state["mask_to_use"]
        ):
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Please create and save mask before generation",
            )
        else:
            st.button(
                "Generate images",
                key="generate_images",
                use_container_width=True,
                type="primary",
                on_click=lambda: toggle_generate_inference(position),
            )


def gallery_image_view(project_uuid, shortlist=False, view=["main"], shot=None, sidebar=False):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)
    shot_list = data_repo.get_shot_list(project_uuid)
    shot_name_uuid_map = {s.name: s.uuid for s in shot_list}
    shot_uuid_list = [GalleryImageViewType.EXPLORER_ONLY.value]

    if shortlist is False:

        st.markdown("### 🖼️ Generated images")
        st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")

    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    # by default only showing explorer views
    k1, k2 = st.columns([5, 1])

    if not sidebar:
        # if this is not the shortlist view then allowing people to choose items per page and num cols
        if not shortlist:
            with h2:
                num_columns = st_memory.slider(
                    "Number of columns:", min_value=3, max_value=7, value=4, key="num_columns_explorer"
                )
            with h3:
                num_items_per_page = st_memory.slider(
                    "Items per page:",
                    min_value=8,
                    max_value=256,
                    value=16,
                    key="num_items_per_page_explorer",
                    step=8,
                )

        else:
            num_items_per_page = st_memory.number_input(
                "Items per page:",
                min_value=8,
                max_value=256,
                value=16,
                key="num_items_per_page_explorer_shortlist",
                step=2,
            )

            num_columns = 2

        # selecting specific shot for adding to the filter
        shot_uuid_list = []

        if not shortlist:
            with h1:
                if project_settings.total_gallery_pages > 10:
                    page_number = st_memory.number_input(
                        "Select page:",
                        min_value=1,
                        max_value=max(1, project_settings.total_gallery_pages),
                        value=1,
                        step=1,
                        key="main_gallery",
                    )
                elif project_settings.total_gallery_pages > 1:
                    page_number = st_memory.radio(
                        "Select page:",
                        options=range(1, project_settings.total_gallery_pages + 1),
                        horizontal=True,
                        key="main_gallery",
                    )
                else:
                    page_number = 1

                st.markdown(f"#### Page {page_number} of {max(1, project_settings.total_gallery_pages)}")
            with h4:
                st.write("")
                if "view_inference_details" in view:
                    open_detailed_view_for_all = st.toggle("Open all prompts:", key="main_gallery_toggle")

        else:
            with h1:
                project_setting = data_repo.get_project_setting(project_uuid)
                page_number = st_memory.radio(
                    "Select page",
                    options=range(1, project_setting.total_shortlist_gallery_pages + 1),
                    horizontal=True,
                    key="shortlist_gallery",
                )
                open_detailed_view_for_all = False

    else:
        project_setting = data_repo.get_project_setting(project_uuid)
        page_number = k1.radio(
            "Select page",
            options=range(1, project_setting.total_shortlist_gallery_pages + 1),
            horizontal=True,
            key="shortlist_gallery",
        )
        open_detailed_view_for_all = False
        num_items_per_page = 8
        num_columns = 2

    if shortlist:
        if "shortlist_gallery" not in st.session_state:
            st.session_state["shortlist_gallery"] = 1
        page_key = "shortlist_gallery"
    else:
        if "main_gallery" not in st.session_state:
            st.session_state["main_gallery"] = 1
        page_key = "main_gallery"
    """
    gallery_image_filter_data = {
        "file_type": InternalFileType.IMAGE.value,
        "tag": (
            InternalFileTag.GALLERY_IMAGE.value
            if not shortlist
            else InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value
        ),
        "project_id": project_uuid,
        "page": st.session_state[page_key],  # Use the correct key here
        "data_per_page": num_items_per_page,
        "sort_order": SortOrder.DESCENDING.value,
    }

    """
    gallery_image_filter_data = {
        "file_type": InternalFileType.IMAGE.value,
        "tag": (
            InternalFileTag.GALLERY_IMAGE.value
            if not shortlist
            else InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value
        ),
        "project_id": project_uuid,
        "page": st.session_state[page_key] or 1,
        "data_per_page": num_items_per_page,
        "sort_order": SortOrder.DESCENDING.value,
    }

    if shot_uuid_list and not sidebar:
        gallery_image_filter_data["shot_uuid_list"] = shot_uuid_list

    gallery_image_list, res_payload = data_repo.get_all_file_list(**gallery_image_filter_data)
    total_pages = res_payload.get("total_pages", res_payload.get("total_log_pages", 1))
    total_image_count = res_payload.get("count", 0)

    if not shortlist:
        if project_settings.total_gallery_pages != total_pages:
            project_settings.total_gallery_pages = total_pages
            refresh_app()
    else:
        if project_settings.total_shortlist_gallery_pages != total_pages:
            project_settings.total_shortlist_gallery_pages = total_pages
            refresh_app()

    if gallery_image_list and len(gallery_image_list):
        start_index = 0
        end_index = min(start_index + num_items_per_page, total_image_count)

    if shortlist is False:
        _, fetch2, fetch3, _ = st.columns([0.25, 1, 1, 0.25])
        # st.markdown("***")
        explorer_stats = data_repo.get_explorer_pending_stats(project_uuid=project_uuid)

        if explorer_stats["temp_image_count"]:
            st.markdown("***")

            with fetch2:
                if explorer_stats["pending_image_count"]:
                    st.info(
                        f"###### {explorer_stats['temp_image_count']} new image{'s' if explorer_stats['temp_image_count'] != 1 else ''} generated ({explorer_stats['pending_image_count']} pending)"
                    )
                else:
                    st.info(
                        f"###### {explorer_stats['temp_image_count']} new image{'s' if explorer_stats['temp_image_count'] != 1 else ''} generated"
                    )

            with fetch3:

                def check_for_new_images(project_uuid, explorer_stats):
                    data_repo.update_temp_gallery_images(project_uuid)

                # In the part of the code where you create the button:
                if st.button(
                    f"Pull new images",
                    key=f"check_for_new_images_",
                    use_container_width=True,
                    on_click=lambda: check_for_new_images(project_uuid, explorer_stats),
                    type="primary",
                ):
                    if explorer_stats["temp_image_count"]:
                        st.success("New images fetched")
                        time.sleep(0.3)
                    refresh_app()
        else:

            _, display, _ = st.columns([0.5, 2, 0.5])
            with display:
                if explorer_stats["pending_image_count"]:
                    with display:
                        st.info(f"###### {explorer_stats['pending_image_count']} images pending generation")

            st.markdown("***")

    total_image_count = res_payload["count"]
    if gallery_image_list and len(gallery_image_list):
        start_index = 0
        end_index = min(start_index + num_items_per_page, total_image_count)
        shot_names = [s.name for s in shot_list]

        if not shortlist:

            # Image gallery display
            for i in range(start_index, end_index, num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i + j < len(gallery_image_list):
                        with cols[j]:
                            st.image(gallery_image_list[i + j].location, use_column_width=True)

                            def toggle_image_selection(image_uuid):
                                if image_uuid in st.session_state["selected_images"]:
                                    st.session_state["selected_images"].remove(image_uuid)
                                else:
                                    st.session_state["selected_images"].append(image_uuid)

                            # Select/Deselect button
                            select_label = (
                                "Deselect"
                                if gallery_image_list[i + j].uuid in st.session_state["selected_images"]
                                else "Select"
                            )
                            button_type = "primary" if select_label == "Deselect" else "secondary"
                            if st.button(
                                select_label,
                                key=f"select_{gallery_image_list[i + j].uuid}",
                                use_container_width=True,
                                type=button_type,
                                on_click=toggle_image_selection,
                                args=(gallery_image_list[i + j].uuid,),
                            ):
                                refresh_app()

                            # -------- inference details --------------
                            if gallery_image_list[i + j].inference_log:
                                log = gallery_image_list[
                                    i + j
                                ].inference_log  # data_repo.get_inference_log_from_uuid(gallery_image_list[i + j].inference_log.uuid)
                                if log:
                                    input_params = json.loads(log.input_params)
                                    prompt = input_params.get("prompt", None)
                                    if not prompt:
                                        query_dict = input_params.get("query_dict")
                                        if isinstance(query_dict, dict):
                                            prompt = query_dict.get("prompt", "Prompt not found")
                                        else:
                                            prompt = "Prompt not found"
                                    model = json.loads(log.output_details)["model_name"].split("/")[-1]
                                    if "view_inference_details" in view:
                                        with st.expander("Prompt:", expanded=open_detailed_view_for_all):
                                            st.info(f"'{prompt}'")
                                else:
                                    st.warning("No inference data")
                            else:
                                st.warning("No data found")
        elif shortlist:
            shot_names.append("**Create New Shot**")
            for i in range(start_index, end_index, num_columns):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    if i + j < len(gallery_image_list):
                        with cols[j]:
                            st.image(gallery_image_list[i + j].location, use_column_width=True)
                            # ---------- add to shot btn ---------------

                            if "add_to_this_shot" in view or "add_to_any_shot" in view:
                                if "add_to_this_shot" in view:
                                    shot_name = shot.name
                                else:
                                    if st.session_state["last_shot_number"] >= len(shot_names):
                                        st.session_state["last_shot_number"] = 0

                                    shot_name = st.selectbox(
                                        "Add to shot:",
                                        shot_names,
                                        key=f"current_shot_sidebar_selector_{gallery_image_list[i + j].uuid}",
                                        index=st.session_state["last_shot_number"],
                                    )

                                if shot_name != "":
                                    if shot_name == "**Create New Shot**":
                                        shot_name = st.text_input(
                                            "New shot name:",
                                            max_chars=40,
                                            key=f"shot_name_{gallery_image_list[i+j].uuid}",
                                        )
                                        if st.button(
                                            "Create new shot",
                                            key=f"create_new_{gallery_image_list[i + j].uuid}",
                                            use_container_width=True,
                                        ):
                                            new_shot = add_new_shot(project_uuid, name=shot_name)
                                            add_key_frame(
                                                gallery_image_list[i + j],
                                                new_shot.uuid,
                                                len(data_repo.get_timing_list_from_shot(new_shot.uuid)),
                                                refresh_state=False,
                                            )
                                            # removing this from the gallery view
                                            data_repo.update_file(gallery_image_list[i + j].uuid, tag="")

                                            st.session_state["last_shot_number"] = len(shot_list)
                                            refresh_app()

                                    else:
                                        if st.button(
                                            f"Add to shot",
                                            key=f"add_{gallery_image_list[i + j].uuid}",
                                            use_container_width=True,
                                        ):
                                            shot_number = shot_names.index(shot_name)
                                            st.session_state["last_shot_number"] = shot_number
                                            shot_uuid = shot_list[shot_number].uuid

                                            add_key_frame(
                                                gallery_image_list[i + j],
                                                shot_uuid,
                                                len(data_repo.get_timing_list_from_shot(shot_uuid)),
                                                refresh_state=False,
                                                update_cur_frame_idx=False,
                                                update_local_only=True,
                                            )
                                            # removing this from the gallery view
                                            data_repo.update_file(gallery_image_list[i + j].uuid, tag="")
                                            st.session_state[f"open_frame_changer_{shot_uuid}"] = False
                                            refresh_app()

                            # else:
                            #     st.error("The image is truncated and cannot be displayed.")
                            if "add_and_remove_from_shortlist" in view:
                                if shortlist:
                                    if st.button(
                                        "Remove from shortlist ➖",
                                        key=f"shortlist_{gallery_image_list[i + j].uuid}",
                                        use_container_width=True,
                                    ):
                                        data_repo.update_file(
                                            gallery_image_list[i + j].uuid,
                                            tag=InternalFileTag.GALLERY_IMAGE.value,
                                        )
                                        st.success("Removed From Shortlist")
                                        time.sleep(0.3)
                                        refresh_app()
                                else:
                                    if st.button(
                                        "Add to shortlist ➕",
                                        key=f"shortlist_{gallery_image_list[i + j].uuid}",
                                        use_container_width=True,
                                        help="The shortlist appears in a box on the left.",
                                    ):
                                        add_file_to_shortlist(gallery_image_list[i + j].uuid)

                            # -------- inference details --------------
                            if gallery_image_list[i + j].inference_log:
                                log = gallery_image_list[
                                    i + j
                                ].inference_log  # data_repo.get_inference_log_from_uuid(gallery_image_list[i + j].inference_log.uuid)
                                if log:
                                    input_params = json.loads(log.input_params)
                                    prompt = input_params.get("prompt", None)
                                    if not prompt:
                                        prompt = input_params.get("query_dict", {}).get(
                                            "prompt", "Prompt not found"
                                        )
                                    model = json.loads(log.output_details)["model_name"].split("/")[-1]
                                    if "view_inference_details" in view:
                                        with st.expander("Prompt:", expanded=open_detailed_view_for_all):
                                            st.info(f"'{prompt}'")

                                else:
                                    st.warning("No inference data")
                            else:
                                st.warning("No data found")

            st.markdown("***")
    else:
        if shortlist:
            st.info("No images present. You can add images to the shortlist in the 'Inspiration Engine' tab.")
        else:
            st.info(
                "No images present. You can generate images by clicking in the 'Inspiration Engine' section."
            )
