import os
import json

import replicate.model
import streamlit as st
import replicate
from PIL import Image

from shared.constants import QUEUE_INFERENCE_QUERIES, InferenceType, ProjectMetaData
from ui_components.methods.common_methods import process_inference_output, save_new_image
from ui_components.methods.file_methods import zoom_and_crop
from ui_components.models import InternalProjectObject, InternalSettingObject
from ui_components.widgets.model_selector_element import model_selector_element
from utils.common_utils import acquire_lock, release_lock
from utils.constants import MLQueryObject, T2IModel
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL
from utils.ml_processor.ml_interface import get_ml_client


# NOTE: since running locally is very slow (comfy startup, models loading, other gens in process..)
# rn we are accessing the replicate API directly, will switch to some other local method in the future
def query_llama3(**kwargs):
    os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_KEY", None)
    model = replicate.models.get("meta/meta-llama-3-8b")
    model_version = model.versions.get("9a9e68fc8695f5847ce944a5cecf9967fd7c64d0fb8c8af1d5bdcc71f03c5e47")

    output = model_version.predict(**kwargs)
    return output


def edit_prompts(edit_text, list_of_prompts):
    query_data = {
        "top_p": 0.9,
        "prompt": f"Could you update these based on the following instructions - \"{edit_text}\" - and return only the list of items with NO OTHER TEXT in the EXACT SAME FORMAT as it's in and the SAME number of items - don't introduce the text, just the list:\n\n{list_of_prompts}\n\nCould you update these based on the following instructions - \"{edit_text}\" - and return only the list of items with NO OTHER TEXT in the EXACT SAME FORMAT as it's in and the SAME number of items- don't introduce the text, just the list \n\nHere is the updated just the update list:",
        "temperature": 0.7,
        "length_penalty": 1,
        "system_prompt": f"You are an extremely direct assistant. You only share the response with no introductory text or ANYTHING else - you mostly only edit existing items.",
        "max_new_tokens": 756,
        "prompt_template": "{prompt}",
        "presence_penalty": 1.15,
        "stop_sequences": " ",
    }
    output = query_llama3(**query_data)

    proper_output = ""
    for item in output:
        if isinstance(item, dict) and "output" in item:
            proper_output += item["output"]
        else:
            proper_output += str(item)

    list_of_prompts = proper_output.strip()
    list_of_prompts = list_of_prompts.split("\n")[0]
    return list_of_prompts


def generate_prompts(
    prompt,
    total_unique_prompts,
    temperature=0.8,
    presence_penalty=1.15,
    top_p=0.9,
    length_penalty=1,
):
    query_data = {
        "top_p": top_p,
        "prompt": f"Given an overall prompt, complete a series of sub-prompts each containing a concise story:\n\nOverall prompt:Story about Leonard Cohen's big day at the beach.\nNumber of items: 12\nSub-prompts:Leonard Cohen looking lying in bed|Leonard Cohen brushing teeth|Leonard Cohen smiling happily|Leonard Cohen driving in car, morning|Leonard Cohen going for a swim at beach|Leonard Cohen sitting on beach towel|Leonard Cohen building sandcastle|Leonard Cohen eating sandwich|Leonard Cohen walking along beach|Leonard Cohen getting out of water at seaside|Leonard Cohen driving home, dark outside|Leonard Cohen lying in bed smiling|close of of Leonard Cohen asleep in bed\n---\nOverall prompt: Visualizing the first day of spring\nNumber of items: 24\nSub-prompts:Frost melting off grass|Sun rising over dewy meadow|Sparrows chirping in tree|Puddles drying up|Bees flying out of hive|Flowers blooming in garden bed|Robin landing on branch|Steam rising from cup of coffee|Morning light creeping through curtains|Wind rustling through leaves|Crocus bulbs pushing through soil|Buds swelling on branches|Birds singing in chorus|Sun shining through rain-soaked pavement|Droplets clinging to spider's web|Green shoots bursting forth from roots|Garden hose dripping water|Fog burning off lake|Light filtering through stained glass window|Hummingbird sipping nectar from flower|Warm breeze rustling through wheat field|Birch trees donning new green coat|Solar eclipse casting shadow on path|Birds returning to their nests\n---\nOverall prompt:{prompt}\nNumber of items: {total_unique_prompts}\nSub-prompts:",
        "temperature": temperature,
        "length_penalty": length_penalty,
        "max_new_tokens": 756,
        "prompt_template": "{prompt}",
        "presence_penalty": presence_penalty,
        "stop_sequences": " ",
    }
    output = query_llama3(**query_data)

    proper_output = ""
    for item in output:
        if isinstance(item, dict) and "output" in item:
            proper_output += item["output"]
        else:
            proper_output += str(item)

    list_of_prompts = proper_output.split("\n")[0]
    return list_of_prompts


def inspiration_engine_element(project_uuid, position="explorer", shot_uuid=None, timing_uuid=None):
    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(project_uuid)
    project: InternalProjectObject = data_repo.get_project_from_uuid(uuid=project_uuid)

    default_prompt_list = "Small seedling sprouting from sand|Desert sun beating down on small plant|small plant plant growing in desert|Vines growing in all directions across dessert|huge plant growing over roads|plant taking growing over man's legs|plant growing over man's face|huge plant growing over building|zoomed out view of plant city|plants everywhere|huge plant growing over white house|huge plant growing over eifel tower|plant growing over ocean|huge plant growing over pyradmids|zoomed out view of entirely green surface|zoomed out view of entirely green planet"
    default_generation_text = "Story about about a plant growing out of a bleak desert, expanding to cover everything and every person and taking over the world, final view of green planet from space, inspirational, each should be about the plant"
    generate_mode, edit_mode = "generate_mode", "edit_mode"
    style_reference_list = [
        {
            "name": "Cutsey Baby Blue",
            "images": [
                "https://i.ibb.co/HrkKwGx/IPAdapter-01084-1.png",
                "https://i.ibb.co/d0zcrHS/Comfy-UI-temp-juama-00007-1.png",
                "https://i.ibb.co/FgQqY18/IPAdapter-01075-1.png",
            ],
        },
    ]

    with st.expander("Inspiration engine", expanded=True):
        default_form_values = {
            "generated_images": [],
            "list_of_prompts": default_prompt_list,
            "prompt_generation_mode": generate_mode,
            "insp_text_prompt": default_generation_text,
            "total_unique_prompt": 16,
            "insp_creativity": 8,
            "insp_edit_prompt": 8,
            "insp_additional_desc": "",
            "insp_additional_neg_desc": "",
            "insp_model_idx": 0,
            "insp_type_of_style": 0,
            "insp_style_influence": 0.5,
            "insp_additional_style_text": "",
            "insp_img_per_prompt": 4,
            "insp_test_mode": False,
            "insp_lightening_mode": False,
            "insp_selected_model": None,
        }

        project_meta_data = json.loads(project.meta_data) if project.meta_data else {}
        prev_settings = project_meta_data.get(ProjectMetaData.INSP_VALUES.value, None)

        settings_to_load = prev_settings or default_form_values
        # picking up default values
        for k, v in settings_to_load.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # ---------------- PROMPT GUIDANCE ---------------------
        st.markdown("#### Prompt guidance")
        h2, h1 = st.columns([1, 1])
        with h2:
            list_of_prompts = st.text_area(
                "List of prompts separated by |:",
                value=st.session_state["list_of_prompts"],
                height=300,
                help="This is a list of prompts that will be used to generate images. Each prompt should be separated by a '|'.",
            )

            st.session_state["list_of_prompts"] = list_of_prompts
            number_of_prompts = len(list_of_prompts.split("|"))
            st.caption(f"Total number of prompts: {number_of_prompts}")

        with h1:
            i1, i2 = st.columns([1, 1])
            with i1:
                if st.session_state["prompt_generation_mode"] == generate_mode:
                    st.success("Generate prompts")
                else:
                    st.warning("Edit prompts")
            with i2:
                if st.session_state["prompt_generation_mode"] == generate_mode:
                    if st.button("Switch to edit prompt mode", use_container_width=True):
                        st.session_state["prompt_generation_mode"] = edit_mode
                        st.rerun()
                else:
                    if st.button("Switch to generate prompt mode", use_container_width=True):
                        st.session_state["prompt_generation_mode"] = generate_mode
                        st.rerun()

            if st.session_state["prompt_generation_mode"] == generate_mode:
                generaton_text = st.text_area(
                    "Text to generate prompts:",
                    value=st.session_state["insp_text_prompt"],
                    height=100,
                    help="This will be used to generate prompts and will overwrite the existing prompts.",
                )

                st.session_state["insp_text_prompt"] = generaton_text
                subprompt1, subprompt2 = st.columns([2, 1])
                with subprompt1:
                    total_unique_prompts = st.slider(
                        "Roughly, how many unique prompts:",
                        min_value=4,
                        max_value=32,
                        step=4,
                        value=st.session_state["total_unique_prompt"],
                    )

                    st.session_state["total_unique_prompt"] = total_unique_prompts

                with subprompt2:
                    creativity = st.slider(
                        "Creativity:",
                        min_value=0,
                        max_value=11,
                        step=1,
                        value=st.session_state["insp_creativity"],
                        help="😏",
                    )
                    temperature = creativity / 10

                    st.session_state["insp_creativity"] = creativity

                total_unique_prompts = total_unique_prompts + 5

                if st.button(
                    "Generate prompts",
                    use_container_width=True,
                    help="This will overwrite the existing prompts.",
                ):
                    st.session_state["list_of_prompts"] = generate_prompts(
                        generaton_text,
                        total_unique_prompts,
                        temperature=temperature,
                    )
                    st.rerun()

            else:
                edit_text = st.text_area(
                    "Text to edit prompts:",
                    value=st.session_state["insp_edit_prompt"],
                    height=100,
                )

                st.session_state["insp_edit_prompt"] = edit_text
                if st.button("Edit Prompts", use_container_width=True):
                    st.session_state["list_of_prompts"] = edit_prompts(edit_text, list_of_prompts)
                    st.rerun()

        i1, i2, _ = st.columns([1, 1, 0.5])
        with i1:
            additonal_description_text = st.text_area(
                "Additional description text:",
                value=st.session_state["insp_additional_desc"],
                help="This will be attached to each prompt.",
            )
            st.session_state["insp_additional_desc"] = additonal_description_text

        with i2:
            negative_prompt = st.text_area(
                "Negative prompt:",
                value=st.session_state["insp_additional_neg_desc"],
                help="This is a list of things to avoid in the generated images.",
            )

            st.session_state["insp_additional_neg_desc"] = negative_prompt

        # ----------------- STYLE GUIDANCE ----------------------
        st.markdown("***")
        st.markdown("#### Style guidance")

        type_of_model = st.radio(
            "Type of model:",
            T2IModel.value_list(),
            index=st.session_state["insp_model_idx"],
            help="Select the type of model to use for image generation.",
            horizontal=True,
        )

        if T2IModel.value_list().index(type_of_model) != st.session_state["insp_model_idx"]:
            st.session_state["insp_model_idx"] = T2IModel.value_list().index(type_of_model)
            st.rerun()

        model1, model2, _ = st.columns([1.5, 1, 0.5])
        with model1:
            if type_of_model == T2IModel.SDXL.value:
                model = model_selector_element(
                    type=T2IModel.SDXL.value,
                    position=position,
                    selected_model=st.session_state["insp_selected_model"],
                )

            elif type_of_model == T2IModel.SD3.value:
                model = model_selector_element(
                    type=T2IModel.SD3.value,
                    position=position,
                    selected_model=st.session_state["insp_selected_model"],
                )

            if model != st.session_state["insp_selected_model"]:
                st.session_state["insp_selected_model"] = model

        if type_of_model == T2IModel.SD3.value:
            with model1:
                st.info("Style references aren't yet supported for Stable Diffusion 3.")
            style_influence = 4.5  # this will actually go into cfg
            list_of_style_references = []

        else:
            input_type_list = [
                "Choose From List",
                "Upload Images",
                "None",
            ]
            type_of_style_input = st.radio(
                "Type of style references:",
                input_type_list,
                index=st.session_state["insp_type_of_style"],
                help="Select the type of style input to use for image generation.",
                horizontal=True,
            )

            list_of_style_references = []
            if input_type_list.index(type_of_style_input) != st.session_state["insp_type_of_style"]:
                st.session_state["insp_type_of_style"] = input_type_list.index(type_of_style_input)
                st.rerun()

            with model2:
                st.write("")
                lightening = st.checkbox(
                    "Lightening Model",
                    help="Generate images faster with less quality.",
                    value=st.session_state["insp_lightening_mode"],
                )

                st.session_state["insp_lightening_mode"] = lightening

            if type_of_style_input == "Upload Images":
                columns = st.columns(3)
                for i, col in enumerate(columns):
                    with col:
                        uploaded_img = st.file_uploader(
                            f"Upload style reference {i+1}:", type=["jpg", "jpeg", "png", "webp"]
                        )
                        if uploaded_img:
                            uploaded_img = (
                                Image.open(uploaded_img)
                                if not isinstance(uploaded_img, Image.Image)
                                else uploaded_img
                            )
                            uploaded_img = zoom_and_crop(
                                uploaded_img, project_settings.width, project_settings.height
                            )
                            list_of_style_references.append(uploaded_img)
                            st.image(uploaded_img)

            elif type_of_style_input == "Choose From List":
                items = style_reference_list

                preset_style, preset_images = None, None
                preset1, preset2 = st.columns([0.5, 1])
                with preset1:
                    preset_style = st.selectbox(
                        "Choose a style preset:",
                        [item["name"] for item in items],
                        help="Select a style preset to use for image generation.",
                    )
                with preset2:
                    preset_images = items[[item["name"] for item in items].index(preset_style)]["images"]
                    cols = st.columns(len(preset_images))
                    for i, col in enumerate(cols):
                        with col:
                            st.image(preset_images[i])
                    list_of_style_references = preset_images

            inf1, inf2 = st.columns([1, 2])
            with inf1:
                style_influence = st.slider(
                    "Style influence:",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    value=st.session_state["insp_style_influence"],
                    help="This is the influence of the style on the generated images.",
                )

                st.session_state["insp_style_influence"] = style_influence

        text1, _ = st.columns([1, 1])
        with text1:
            additional_style_text = st.text_area(
                "Additional style guidance text:",
                value=st.session_state["insp_additional_style_text"],
                help="This is additonal text that will be used to guide the style.",
            )

            st.session_state["insp_additional_style_text"] = additional_style_text

        # ---------------------- GENERATION SETTINGS --------------------------
        st.markdown("***")
        st.markdown("#### Generation settings")

        prompt1, prompt2 = st.columns([1.25, 1])
        with prompt1:
            images_per_prompt = st.slider(
                "Images per prompt:",
                min_value=1,
                max_value=64,
                step=4,
                value=st.session_state["insp_img_per_prompt"],
            )

            st.session_state["insp_img_per_prompt"] = images_per_prompt

        with prompt2:
            if number_of_prompts == 1:
                st.info(
                    f"{number_of_prompts} prompt for {images_per_prompt} images per prompt makes a total of **{number_of_prompts*images_per_prompt} images**."
                )
            else:
                st.info(
                    f"{number_of_prompts} prompts for {images_per_prompt} images per prompt makes a total of **{number_of_prompts*images_per_prompt} images**."
                )

        test_first_prompt = st.toggle(
            "Test only first prompt",
            value=st.session_state["insp_test_mode"],
            help="This will only generate images for the first prompt.",
        )

        st.session_state["insp_test_mode"] = test_first_prompt

        if test_first_prompt:
            list_of_prompts = list_of_prompts.split("|")[0]

        # ------------------ GENERATE --------------------------
        if st.button("Generate images"):
            ml_client = get_ml_client()

            input_image_file_list = []
            atleast_one_log_created = False
            for img in list_of_style_references:
                input_image_file = save_new_image(img, project_uuid)
                input_image_file_list.append(input_image_file)

            prompts_to_be_processed = [item for item in list_of_prompts.split("|") if item]
            for _, image_prompt in enumerate(prompts_to_be_processed):
                for _ in range(images_per_prompt):

                    if type_of_model == T2IModel.SDXL.value:
                        data = {
                            "shot_uuid": shot_uuid,
                            "img_uuid_list": json.dumps([f.uuid for f in input_image_file_list]),
                            "additonal_description_text": additonal_description_text,
                            "additional_style_text": additional_style_text,
                            "sdxl_model": model,
                            "lightening": lightening,
                            "width": project_settings.width,
                            "height": project_settings.height,
                        }

                        for idx, f in enumerate(input_image_file_list):
                            data[f"file_uuid_{idx}"] = f.uuid

                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            model_uuid=None,
                            image_uuid=None,
                            guidance_scale=5,
                            seed=-1,
                            num_inference_steps=30,
                            strength=style_influence,
                            adapter_type=None,
                            prompt=image_prompt,
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            data=data,
                        )

                        output, log = ml_client.predict_model_output_standardized(
                            ML_MODEL.sd3_local,
                            query_obj,
                            queue_inference=QUEUE_INFERENCE_QUERIES,
                        )

                    # for sd3 model
                    else:
                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            model_uuid=None,
                            image_uuid=None,
                            guidance_scale=5,
                            seed=-1,
                            num_inference_steps=30,
                            strength=style_influence,
                            adapter_type=None,
                            prompt=f"{image_prompt}, {additonal_description_text}, {additional_style_text}",
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            data={"shift": 3.0, "model": model},  # default value
                        )

                        output, log = ml_client.predict_model_output_standardized(
                            ML_MODEL.sd3_local,
                            query_obj,
                            queue_inference=QUEUE_INFERENCE_QUERIES,
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

                        # saving state here (this ensures that generation was successfully created with the current settings)
                        if not atleast_one_log_created:
                            atleast_one_log_created = True
                            data_dict = {}
                            for k, _ in default_form_values.items():
                                if k in st.session_state:
                                    data_dict[k] = st.session_state[k]

                            key = project_uuid
                            # TODO: make storing and retrieving meta data into a single method
                            if acquire_lock(key):
                                project: InternalProjectObject = data_repo.get_project_from_uuid(uuid=key)
                                if project:
                                    meta_data = json.loads(project.meta_data) if project.meta_data else {}
                                    meta_data[ProjectMetaData.INSP_VALUES.value] = data_dict
                                    data_repo.update_project(
                                        uuid=project.uuid, meta_data=json.dumps(meta_data)
                                    )
                                release_lock(key)
