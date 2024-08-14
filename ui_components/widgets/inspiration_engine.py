import os
import json
import time
import math
from django.db import connection
import replicate.model
import streamlit as st
import replicate
from PIL import Image
import requests
from io import BytesIO

from shared.constants import QUEUE_INFERENCE_QUERIES, InferenceType, ProjectMetaData
from ui_components.methods.common_methods import process_inference_output, save_new_image
from ui_components.methods.file_methods import generate_pil_image, zoom_and_crop
from ui_components.models import InternalProjectObject, InternalSettingObject
from ui_components.widgets.model_selector_element import model_selector_element
from utils.common_utils import sqlite_atomic_transaction
from utils.constants import MLQueryObject, T2IModel
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL
from utils.ml_processor.ml_interface import get_ml_client
from utils import st_memory
from utils.state_refresh import refresh_app

# NOTE: since running locally is very slow (comfy startup, models loading, other gens in process..)
# rn we are accessing the replicate API directly, will switch to some other local method in the future


def check_replicate_key():
    data_repo = DataRepo()
    app_secrets = data_repo.get_app_secrets_from_user_uuid()
    if "replicate_key" in app_secrets and app_secrets["replicate_key"]:
        if app_secrets["replicate_key"] == "xyz":
            return False
        st.session_state["replicate_key"] = app_secrets["replicate_key"]
        os.environ["REPLICATE_API_TOKEN"] = st.session_state["replicate_key"]
    else:
        return False

    return True


def edit_prompts(edit_text, list_of_prompts):
    # Convert list of prompts to a single string separated by '|'
    prompts_string = "|".join(list_of_prompts)

    query_data = {
        "top_p": 0.9,
        "prompt": f"Could you update these based on the following instructions - \"{edit_text}\" - and return only the list of items with NO OTHER TEXT in the EXACT SAME FORMAT as it's in and the SAME number of items - don't introduce the text, just the list:\n\n{prompts_string}\n\nCould you update these based on the following instructions - \"{edit_text}\" - and return only the list of items with NO OTHER TEXT in the EXACT SAME FORMAT as it's in and the SAME number of items- don't introduce the text, just the list \n\nHere is the updated just the update list:",
        "temperature": 0.7,
        "length_penalty": 1,
        "system_prompt": f"You are an extremely direct assistant. You only share the response with no introductory text or ANYTHING else - you mostly only edit existing items.",
        "max_new_tokens": 756,
        "prompt_template": "{prompt}",
        "presence_penalty": 1.15,
        "stop_sequences": " ",
    }

    output = replicate.run(
        "meta/meta-llama-3-70b",
        input=query_data,
    )

    proper_output = ""  # Initialize an empty string to accumulate outputs
    for item in output:
        if isinstance(item, dict) and "output" in item:
            proper_output += item["output"]  # Concatenate each output to the proper_output string
        else:
            proper_output += str(item)  # Handle cases where item is not a dictionary

    list_of_prompts = proper_output.strip()
    list_of_prompts = list_of_prompts.lstrip("\n")
    # Proper output should be only before the first \n
    list_of_prompts = list_of_prompts.split("\n")[0]

    return list_of_prompts


def generate_prompts(
    overall_request,
    initial_examples,
    total_unique_prompts,
    temperature=0.8,
    presence_penalty=1.15,
    top_p=0.9,
    length_penalty=1,
):
    # Combine different types of examples with varying lengths
    mixed_examples = f"""
    Overall prompt: Story about Leonard Cohen's big day at the beach
    Number of items: 8
    Responses: Leonard Cohen waking up early|Leonard Cohen packing beach essentials|Leonard Cohen driving to the beach|Leonard Cohen setting up beach umbrella|Leonard Cohen swimming in the ocean|Leonard Cohen writing lyrics in the sand|Leonard Cohen watching the sunset|Leonard Cohen falling asleep under the stars
    ---
    Overall prompt: Moody and atmospheric styles
    Number of items: 15
    Responses: Film noir alley with flickering streetlight|Misty lake at dawn|Abandoned amusement park at twilight|Candlelit cathedral interior|Foggy London street with gas lamps|Rainy cafe window with blurred figures|Moonlit graveyard with ancient tombstones|Dimly lit jazz club with smoky air|Stormy sea with lighthouse in distance|Shadowy forest with twisted trees|Desolate desert highway at night|Crumbling castle on a misty hilltop|Neon-lit Chinatown in the rain|Dusty attic with cobweb-covered relics|Snow-covered mountain peak at dusk
    ---
    Overall prompt: Evolution of communication
    Number of items: 10
    Responses: Cave paintings depicting hunt scenes|Ancient Egyptian hieroglyphs on papyrus|Medieval monk transcribing manuscript|Gutenberg printing press in action|Telegraph operator sending Morse code|Rotary phone on vintage desk|First television broadcast|Early computer with punch cards|Smartphone displaying social media feed|Futuristic holographic video call
    ---
    Overall prompt: Bohemian Rhapsody by Queen
    Number of items: 12
    Responses: Silhouette of a man confessing to his mother|Dramatic courtroom scene with figure pleading case|Tears falling on a letter of apology|Person staring at reflection, questioning reality|Thunderbolts and lightning illuminating a dark sky|Devils and angels fighting over a soul|Operatic figures dramatically posed on a stage|Wind blowing through an empty room, symbolizing goodbye|Galileo's telescope pointed at the night sky|Crowd headbanging to rock music|Broken shards of mirror reflecting fractured identity|Lone figure walking away, leaving the past behind
    ---
    Overall prompt: {overall_request}
    Number of items: {total_unique_prompts}
    Responses: {initial_examples}"""

    prompt = mixed_examples

    query_data = {
        "top_p": top_p,
        "prompt": prompt,
        "temperature": temperature,
        "length_penalty": length_penalty,
        "max_new_tokens": 756,
        "prompt_template": "{prompt}",
        "presence_penalty": presence_penalty,
        "stop_sequences": " ",
    }
    output = replicate.run(
        "meta/meta-llama-3-70b",
        input=query_data,
    )

    proper_output = ""
    for item in output:
        if isinstance(item, dict) and "output" in item:
            proper_output += item["output"]
        else:
            proper_output += str(item)
    
    list_of_prompts = proper_output.strip()
    list_of_prompts = list_of_prompts.lstrip("\n")
    list_of_prompts = list_of_prompts.split("\n")[0]
    
    # Combine initial examples with generated prompts
    combined_prompts = initial_examples + "|" + list_of_prompts
    
    # Remove any double bars that might have been created
    combined_prompts = combined_prompts.replace("||", "|")

    st.session_state["prompts_to_display_separately"] = combined_prompts
    
    return combined_prompts


def inspiration_engine_element(project_uuid, position="explorer", shot_uuid=None, timing_uuid=None):
    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(project_uuid)
    project: InternalProjectObject = data_repo.get_project_from_uuid(uuid=project_uuid)

    default_prompt_list = [
        "Small seedling sprouting from sand",
        "Desert sun beating down on small plant",
        "small plant growing in desert",
        "Vines growing in all directions across desert",
        "huge plant growing over roads",
        "plant taking growing over man's legs",
        "plants everywhere",
        "huge plant growing over pyramids",
        "zoomed out view of entirely green planet",
    ]
    default_generation_text = "Story about about a plant growing out of a bleak desert, expanding to cover everything and every person and taking over the world, final view of green planet from space, inspirational, each should be about the plant"
    generate_mode, edit_mode = "generate_mode", "edit_mode"
    style_reference_list = [
        {
            "name": "The Strangest Dream",
            "images": [
                "https://banodoco-data-bucket-public.s3.ap-south-1.amazonaws.com/general_pics/strangest_dream_2.png",
            ],
            "description": "Bold, dark colors, Dark Blue & red dominant, dream like, surreal.",
            "models_works_best_with": "Dreamshaper, Deliberate",
            "workflows_works_best_with": "Rad Attack, Slushy Realistiche, Smooth n’ Steady",
            "example_video": "https://banodoco.s3.amazonaws.com/plan/5c2cfcb1-04c3-432b-9585-a39a0b55686e+(1).mp4",
            "created_by": "Hannah Submarine",
        },
        {
            "name": "Green Hard Funk",
            "images": [
                "https://banodoco-data-bucket-public.s3.ap-south-1.amazonaws.com/general_pics/green_hard_funk_1.png",
            ],
            "description": "Dominant green tones, red accents, high contrast, geometrical lining.",
            "models_works_best_with": "Dreamshaper, Deliberate",
            "workflows_works_best_with": "Rad attack, Chucky realistiche",
            "example_video": "https://banodoco.s3.amazonaws.com/plan/6a974b2d-bc0e-4c7c-87b5-99ca5594162e.mp4",
            "created_by": "Hannah Submarine",
        },
        {
            "name": "Nordic Pale Blue",
            "images": [
                "https://banodoco-data-bucket-public.s3.ap-south-1.amazonaws.com/general_pics/nordic_pale_blue_1.png",
            ],
            "description": "Pale blue, grey, pastel colors, ornamental and decorative details.",
            "models_works_best_with": "Dreamshaper, Deliberate",
            "workflows_works_best_with": "Rad Attack, Liquid Loop",
            "example_video": "https://banodoco.s3.amazonaws.com/plan/454e2dff-97c5-49c7-91dc-a66c5216cd44+(1).mp4",
            "created_by": "Hannah Submarine",
        },
        {
            "name": "Insane Animane",
            "images": [
                "https://banodoco-data-bucket-public.s3.ap-south-1.amazonaws.com/general_pics/insane_animane_1.png",
            ],
            "description": "Detailed animation style, Blue, orange, white dominant, intense, expressive",
            "models_works_best_with": "Dreamshaper, Deliberate",
            "workflows_works_best_with": "Rad Attack, Slushy Realistiche",
            "example_video": "https://banodoco.s3.amazonaws.com/plan/3b2a0c0f-e100-4f44-b386-9e3c7781854e.mp4",
            "created_by": "Hannah Submarine",
        },
        {
            "name": "Delicate Pink Glimmer",
            "images": [
                "https://banodoco-data-bucket-public.s3.ap-south-1.amazonaws.com/general_pics/delicate_pink_2.png",
            ],
            "description": "Delicate lines, soft colors, pink-blue & green dominant, misty/glimmer shine.",
            "models_works_best_with": "Deliberate, Realistic Vision",
            "workflows_works_best_with": "Smooth n' Steady, Rad Attack",
            "example_video": "https://banodoco.s3.amazonaws.com/plan/8d689176-9833-47ed-970f-2c14ff0440a6.mp4",
            "created_by": "Hannah Submarine",
        },
    ]

    with st.expander("Inspiration engine", expanded=True):
        if st_memory.toggle(
            "Open", value=True, key="inspiration_engine_toggle", help="Close it to speed up the page."
        ):
            default_form_values = {
                "generated_images": [],
                "list_of_prompts": default_prompt_list,
                "prompt_generation_mode": generate_mode,
                "insp_text_prompt": default_generation_text,
                "insp_examples": "empty barren desert|tiny seeding visible in growing out of desert sand",
                "total_unique_prompt": 16,
                "insp_creativity": 8,
                "insp_how_to_display": 0,
                "insp_edit_prompt": 8,
                "insp_additional_desc": "",
                "insp_additional_neg_desc": "",
                "insp_model_idx": 0,
                "insp_type_of_style": 0,
                "insp_style_influence": [0.7, 0.7, 0.7],
                "insp_composition_influence": [0.0, 0.0, 0.0],
                "insp_vibe_influence": [0.0, 0.0, 0.0],
                "insp_additional_style_text": "",
                "insp_img_per_prompt": 4,
                "insp_test_mode": True,
                "insp_lightning_mode": False,
                "insp_selected_model": None,
            }

            project_meta_data = json.loads(project.meta_data) if project.meta_data else {}
            prev_settings = project_meta_data.get(ProjectMetaData.INSP_VALUES.value, {})

            # Ensure that settings_to_load prioritizes prev_settings and falls back to default_form_values if a key is missing
            settings_to_load = {**default_form_values, **prev_settings}

            # Update settings_to_load to prioritize prev_settings over default_form_values
            for key in default_form_values:
                if key not in settings_to_load:
                    settings_to_load[key] = default_form_values[key]

            # Picking up default values
            for k, v in settings_to_load.items():
                if k not in st.session_state:
                    st.session_state[k] = v

            # ---------------- PROMPT GUIDANCE ---------------------
            h1, h2, _ = st.columns([1, 1, 0.25])
            with h1:
                st.markdown("#### Prompt guidance")

            # list_of_prompts = st.session_state["list_of_prompts"]
            with h2:
                if st.button("Remove all prompts"):
                    st.session_state["list_of_prompts"] = [""]
                    refresh_app()

            h2_a, h2_b, h2_c, h1 = st.columns([0.5, 0.5, 0.5, 0.75])
            if isinstance(st.session_state["list_of_prompts"], str):
                st.session_state["list_of_prompts"] = st.session_state["list_of_prompts"].split("|")

            column_handlers = [h2_a, h2_b, h2_c] * ((len(st.session_state["list_of_prompts"]) + 2) // 3)

            for index, prompt in enumerate(st.session_state["list_of_prompts"]):
                with column_handlers[index]:
                    # Create a text area for each prompt
                    user_input = st.text_area(
                        f"Prompt {index + 1}:", value=prompt, height=50, key=f"prompt_{index}"
                    )
                    # Update the prompt in session state if it changes
                    if user_input != prompt:
                        st.session_state["list_of_prompts"][index] = user_input

                    col1, col2 = st.columns(2)
                    with col1:
                        # Add a new prompt below the current one when the "+" button is pressed
                        if st.button("➕", key=f"add_{index}", use_container_width=True):
                            st.session_state["list_of_prompts"].insert(index + 1, user_input)
                            refresh_app()
                    with col2:
                        # Delete the current prompt when the "🗑️" button is pressed
                        if len(st.session_state["list_of_prompts"]) > 1:
                            if st.button("🗑️", key=f"delete_{index}", use_container_width=True):
                                st.session_state["list_of_prompts"].pop(index)
                                refresh_app()
                        else:
                            st.button(
                                "🗑️",
                                key=f"delete_{index}",
                                use_container_width=True,
                                disabled=True,
                                help="You can't delete the last prompt.",
                            )
            """
            if st.session_state["list_of_prompts"] != list_of_prompts:
                st.session_state["list_of_prompts"] = list_of_prompts
                refresh_app()
            """
            number_of_prompts = len(st.session_state["list_of_prompts"])

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
                            refresh_app()
                    else:
                        if st.button("Switch to generate prompt mode", use_container_width=True):
                            st.session_state["prompt_generation_mode"] = generate_mode
                            refresh_app()

                app_secrets = data_repo.get_app_secrets_from_user_uuid()
                if "replicate_key" in app_secrets and app_secrets["replicate_key"]:
                    st.session_state["replicate_key"] = app_secrets["replicate_key"]
                else:
                    st.session_state["replicate_key"] = ""

                replicate_warning_message = "We currently use Replicate for LLM queries for simplicity. This costs $0.00025/run. You can add a key in App Settings."
                if st.session_state["prompt_generation_mode"] == generate_mode:

                    generation_text = st_memory.text_area(
                        "Text to generate prompts:",                        
                        height=100,
                        help="This will be used to generate prompts and will overwrite the existing prompts.",
                        key="insp_text_prompt",
                    )

                    generation_examples = st_memory.text_area(
                        "Examples separated by |:",                        
                        height=100,
                        help="This will be used to generate prompts and will overwrite the existing prompts.",
                        key="insp_examples",
                    )

                    if st.session_state["insp_text_prompt"] != generation_text:
                        st.session_state["insp_text_prompt"] = generation_text
                        refresh_app()

                    subprompt1, subprompt2 = st.columns([2, 1])
                    with subprompt1:
                        total_unique_prompts = st.slider(
                            "Roughly, how many unique prompts:",
                            min_value=4,
                            max_value=32,
                            step=4,
                            value=st.session_state["total_unique_prompt"],
                        )

                        if st.session_state["total_unique_prompt"] != total_unique_prompts:
                            st.session_state["total_unique_prompt"] = total_unique_prompts
                            refresh_app()

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

                        if st.session_state["insp_creativity"] != creativity:
                            st.session_state["insp_creativity"] = creativity
                            refresh_app()

                    total_unique_prompts = total_unique_prompts + 5

                    if not check_replicate_key():
                        st.info(replicate_warning_message)
                    else:

    

                        if st.button(
                            "Generate prompts",
                            use_container_width=True,
                            help="This will overwrite the existing prompts.",
                            on_click=generate_prompts(
                                generation_text,
                                generation_examples,
                                total_unique_prompts,
                                temperature=temperature,                                
                            )
                        ):                                             
                            refresh_app()

                    if "prompts_to_display_separately" not in st.session_state:
                        st.session_state["prompts_to_display_separately"] = ""

                    if st.session_state["prompts_to_display_separately"] != "":
                        height = math.ceil(len(st.session_state["prompts_to_display_separately"])/2)
                        st.text_area(
                            "Generated prompts:",
                            value=st.session_state["prompts_to_display_separately"],
                            height=height                            
                        )
                        bottom1, bottom2, bottom3 = st.columns([1, 1, 1])
                        with bottom1:
                            if st.button("Remove"):
                                st.session_state["prompts_to_display_separately"] = ""
                                refresh_app()
                        with bottom2:
                            if st.button("Replace existing prompts"):
                                st.session_state["list_of_prompts"] = st.session_state[
                                    "prompts_to_display_separately"
                                ].split("|")
                                st.session_state["prompts_to_display_separately"] = ""
                                refresh_app()
                        with bottom3:
                            if st.button("Add to existing prompts"):
                                st.session_state["list_of_prompts"] = st.session_state[
                                    "list_of_prompts"
                                ] + st.session_state["prompts_to_display_separately"].split("|")
                                st.session_state["prompts_to_display_separately"] = ""
                                refresh_app()

                else:
                    edit_text = st.text_area(
                        "Text to edit prompts:",
                        value=st.session_state["insp_edit_prompt"],
                        height=100,
                    )

                    if st.session_state["insp_edit_prompt"] != edit_text:
                        st.session_state["insp_edit_prompt"] = edit_text
                        refresh_app()

                    if not check_replicate_key():
                        st.info(replicate_warning_message)
                    else:
                        if st.button("Edit Prompts", use_container_width=True):
                            generated_prompts = edit_prompts(edit_text, st.session_state["list_of_prompts"])
                            st.session_state["list_of_prompts"] = generated_prompts.split("|")
                            refresh_app()

            i1, i2, _ = st.columns([1, 1, 0.5])
            with i1:
                additional_description_text = st.text_area(
                    "Additional description text:",
                    value=st.session_state["insp_additional_desc"],
                    help="This will be attached to each prompt.",
                )
                if st.session_state["insp_additional_desc"] != additional_description_text:
                    st.session_state["insp_additional_desc"] = additional_description_text
                    refresh_app()

            with i2:
                negative_prompt = st.text_area(
                    "Negative prompt:",
                    value=st.session_state["insp_additional_neg_desc"],
                    help="This is a list of things to avoid in the generated images.",
                )

                if st.session_state["insp_additional_neg_desc"] != negative_prompt:
                    st.session_state["insp_additional_neg_desc"] = negative_prompt
                    refresh_app()

            # ----------------- STYLE GUIDANCE ----------------------
            st.markdown("***")
            st.markdown("#### Model selection")

            type_of_model = st.radio(
                "Type of model:",
                T2IModel.value_list(),
                index=st.session_state["insp_model_idx"],
                help="Select the type of model to use for image generation. We strongly recommend SDXL because it allows you to guide the style with more precision.",
                horizontal=True,
            )

            if T2IModel.value_list().index(type_of_model) != st.session_state["insp_model_idx"]:
                st.session_state["insp_model_idx"] = T2IModel.value_list().index(type_of_model)
                refresh_app()

            lightning = False
            list_of_strengths = []
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

                elif type_of_model == T2IModel.FLUX.value:
                    model = "flux1-schnell-fp8.safetensors"

                    st.info("Flux Schell FP8 will be selected by default. It requires atleast 17GB VRAM.")

                """
                model - {url, filename, desc}
                """
                if model != st.session_state["insp_selected_model"]:
                    st.session_state["insp_selected_model"] = model
                    refresh_app()

            st.markdown("***")
            st.markdown("#### Style guidance")

            type_of_style_input = None
            if type_of_model == T2IModel.SD3.value:
                sd3, _ = st.columns([1, 1])
                with sd3:
                    st.info("Style references aren't yet supported for SD3.")
                style_influence = 4.5  # this will actually go into cfg

            elif type_of_model == T2IModel.SDXL.value:
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

                if input_type_list.index(type_of_style_input) != st.session_state["insp_type_of_style"]:
                    st.session_state["insp_type_of_style"] = input_type_list.index(type_of_style_input)
                    refresh_app()

                with model1:
                    st.write("")
                    lightning = st.checkbox(
                        "Lightning Model",
                        help="Generate images faster with less quality.",
                        value=st.session_state["insp_lightning_mode"],
                    )

                    if st.session_state["insp_lightning_mode"] != lightning:
                        st.session_state["insp_lightning_mode"] = lightning
                        refresh_app()

                if "list_of_style_references" not in st.session_state:
                    st.session_state["list_of_style_references"] = []

                if type_of_style_input == "Upload Images":
                    preview_1, preview_2, preview_3 = st.columns([1, 1, 1])

                    if len(st.session_state["list_of_style_references"]) < 3:
                        h1, h2 = st.columns([1, 1.5])
                        if len(st.session_state["list_of_style_references"]) == 0:
                            column = preview_1
                            text = "Upload up to 3 style references."
                        elif len(st.session_state["list_of_style_references"]) == 1:
                            column = preview_2
                            text = "Upload up to 2 more style references."
                        elif len(st.session_state["list_of_style_references"]) == 2:
                            column = preview_3
                            text = "Upload 1 more style reference."
                        else:
                            column = h1
                        with column:
                            uploaded_images = st.file_uploader(
                                text,
                                type=["jpg", "jpeg", "png", "webp"],
                                accept_multiple_files=True,
                            )
                            
                            if len(uploaded_images) > 1:
                                text = "Add reference images"
                            else:
                                text = "Add reference image"

                            def add_image_reference():
        

                                # Check if there are less than 3 images already in the list
                                while (
                                    len(st.session_state["list_of_style_references"]) < 3
                                    and uploaded_images
                                ):
                                    st.session_state["list_of_style_references"].append(
                                        uploaded_images.pop(0)
                                    )

                                if uploaded_images:  # If there are still images left, show a warning
                                    st.warning("You can only upload 3 style references.")
                                
                            if st.button("Add image reference", use_container_width=True, on_click=add_image_reference):
                                refresh_app()
                 

                elif type_of_style_input == "Choose From List":

                    if st.session_state["list_of_style_references"]:
                        st.session_state["list_of_style_references"] = [
                            items["images"] for items in style_reference_list
                        ]

                    items = style_reference_list

                    preset_style, preset_images = None, None
                    preset1, _ = st.columns([0.5, 1.0])
                    with preset1:
                        preset_style = st_memory.selectbox(
                            "Choose a style preset:",
                            [item["name"] for item in items],
                            key="insp_style",
                            help="Select a style preset to use for image generation.",
                        )

                    preview_1, preview_2, preview_3 = st.columns([1, 1, 1])

                    if (
                        st.session_state["list_of_style_references"]
                        != items[[item["name"] for item in items].index(preset_style)]["images"]
                    ):
                        st.session_state["list_of_style_references"] = items[
                            [item["name"] for item in items].index(preset_style)
                        ]["images"]
                        # st.image(items[[item["name"] for item in items].index(preset_style)]["images"])
                    with preview_2:
                        st.info(
                            f"""**Recommended animation styling models:** {items[[item["name"] for item in items].index(preset_style)]["models_works_best_with"]}
                        \n**Recommended workflow:** {items[[item["name"] for item in items].index(preset_style)]["workflows_works_best_with"]}
                        \n**Created by:** {items[[item["name"] for item in items].index(preset_style)]["created_by"]}
                        \n**Description:** {items[[item["name"] for item in items].index(preset_style)]["description"]}
"""
                        )

                    with preview_3:
                        st.video(items[[item["name"] for item in items].index(preset_style)]["example_video"])

                else:
                    st.session_state["list_of_style_references"] = []

                for key in ["insp_style_influence", "insp_composition_influence", "insp_vibe_influence"]:
                    # if it's a float, make it into a list with 3 items
                    if not isinstance(st.session_state[key], list):
                        st.session_state[key] = [st.session_state[key]] * 3

                # Determine if we should use the first value for all sliders
                if type_of_style_input != "None":
                    use_first_value = len(st.session_state["list_of_style_references"]) == 1

                    total_influences = 0

                    for i, col in enumerate([preview_1, preview_2, preview_3]):
                        with col:
                            if i < len(st.session_state["list_of_style_references"]):
                                uploaded_file = st.session_state["list_of_style_references"][i]
                                if uploaded_file is not None:
                                    if isinstance(uploaded_file, str) and uploaded_file.startswith("http"):
                                        st.image(uploaded_file)
                                    else:
                                        display_img = Image.open(uploaded_file)
                                        st.image(display_img)

                                    # Determine which index to use for the sliders
                                    slider_index = 0 if use_first_value else i

                                    # Style influence slider
                                    # Ensure the session state lists have enough elements
                                    for key in ["insp_style_influence", "insp_composition_influence", "insp_vibe_influence"]:
                                        if slider_index >= len(st.session_state[key]):
                                            st.session_state[key].extend([0.5] * (slider_index + 1 - len(st.session_state[key])))

                                    style_influence = st.slider(
                                        f"Style influence:",
                                        min_value=0.0,
                                        max_value=1.0,
                                        step=0.1,
                                        value=st.session_state["insp_style_influence"][slider_index],
                                        key=f"style_influence_{i}",
                                        help=f"How much the style of the image should influence the generated image.",
                                    )
                                    if st.session_state["insp_style_influence"][slider_index] != style_influence:
                                        st.session_state["insp_style_influence"][slider_index] = style_influence
                                        refresh_app()

                                    # Composition influence slider
                                    composition_influence = st.slider(
                                        f"Composition influence:",
                                        min_value=0.0,
                                        max_value=1.0,
                                        step=0.1,
                                        value=st.session_state["insp_composition_influence"][slider_index],
                                        key=f"composition_influence_{i}",
                                        help=f"How much the composition of the image should influence the generated image.",
                                    )
                                    if st.session_state["insp_composition_influence"][slider_index] != composition_influence:
                                        st.session_state["insp_composition_influence"][slider_index] = composition_influence
                                        refresh_app()

                                    # Vibe influence slider
                                    vibe_influence = st.slider(
                                        f"Vibe influence:",
                                        min_value=0.0,
                                        max_value=1.0,
                                        step=0.1,
                                        value=st.session_state["insp_vibe_influence"][slider_index],
                                        key=f"vibe_influence_{i}",
                                        help=f"How much the vibe of the image should influence the generated image.",
                                    )
                                    if st.session_state["insp_vibe_influence"][slider_index] != vibe_influence:
                                        st.session_state["insp_vibe_influence"][slider_index] = vibe_influence
                                        refresh_app()

                                    if type_of_style_input == "Upload Images":

                                        def remove_image_reference(i):
                                            # Check if the index is valid before removing
                                            if i < len(st.session_state["list_of_style_references"]):
                                                # Remove the image from the list
                                                st.session_state["list_of_style_references"].pop(i)

                                                for key in ["insp_style_influence", "insp_composition_influence", "insp_vibe_influence"]:
                                                    if i < len(st.session_state[key]):
                                                        st.session_state[key].pop(i)
                                        if st.button(
                                            f"Remove image reference",
                                            use_container_width=True,
                                            key=f"remove_{i}",
                                            on_click=remove_image_reference(i),
                                        ):                                   
                                            refresh_app()

                                    # add up the 3 values and if together they're over 1.5, show a warning
                                    item_strengths = (style_influence, composition_influence, vibe_influence)
                                    list_of_strengths.append(item_strengths)
                                    total_influences += sum(1 for strength in item_strengths if strength > 0.0)

                                else:
                                    st.error("Uploaded file does not support file-like operations.")

                    warning, _ = st.columns([1, 1.5])
                    with warning:
                        if total_influences > 6:
                            st.warning("Any more than 6 total influences will take more than 24GB of VRAM.")

            text1, _ = st.columns([1, 1])
            with text1:
                additional_style_text = st.text_area(
                    "Additional style guidance text:",
                    value=st.session_state["insp_additional_style_text"],
                    help="This is additonal text that will be used to guide the style.",
                )

                if st.session_state["insp_additional_style_text"] != additional_style_text:
                    st.session_state["insp_additional_style_text"] = additional_style_text
                    refresh_app()

            # ---------------------- GENERATION SETTINGS --------------------------
            st.markdown("***")
            st.markdown("#### Generation settings")
            h1, h2, _ = st.columns([0.35, 1, 0.6])
            with h1:
                run_all_prompts = st_memory.toggle(
                    "Run all prompts:",
                    value=st.session_state["insp_test_mode"],
                    help="This will only generate images for the first prompt.",
                )
            if not run_all_prompts:
                # range of prompts to test
                with h2:
                    # if there's more than 1 prompt, show a slider
                    if len(st.session_state["list_of_prompts"]) > 1:
                        prompts_to_test = st.multiselect(
                            "Prompts to run:",
                            options=list(range(1, len(st.session_state["list_of_prompts"]) + 1)),
                            default=[1, 2],
                            format_func=lambda x: f"{st.session_state['list_of_prompts'][x-1]}",
                        )

                        # Convert selected indices to actual prompts
                        prompts_to_test = [
                            st.session_state["list_of_prompts"][i - 1] for i in prompts_to_test
                        ]

                        # Number of prompts selected
                        number_of_prompts = len(prompts_to_test)
                    else:
                        prompts_to_test = [st.session_state["list_of_prompts"][0]]  # Wrap in a list
                        number_of_prompts = 1

            prompt1, prompt2, prompt3 = st.columns([1.25, 1, 1])
            with prompt1:
                images_per_prompt = st.slider(
                    "Images per prompt:",
                    min_value=1,
                    max_value=64,
                    step=1,
                    value=st.session_state["insp_img_per_prompt"],
                )

                if st.session_state["insp_img_per_prompt"] != images_per_prompt:
                    st.session_state["insp_img_per_prompt"] = images_per_prompt
                    refresh_app()

            with prompt2:
                if number_of_prompts == 1:
                    st.info(
                        f"{number_of_prompts} prompt for {images_per_prompt} images per prompt makes a total of **{number_of_prompts*images_per_prompt} images**."
                    )
                else:
                    st.info(
                        f"{number_of_prompts} prompts for {images_per_prompt} images per prompt makes a total of **{number_of_prompts*images_per_prompt} images**."
                    )

            st.session_state["insp_test_mode"] = run_all_prompts

            if not run_all_prompts:
                prompts_to_be_processed = prompts_to_test
            else:
                prompts_to_be_processed = st.session_state["list_of_prompts"]

            # ------------------ GENERATE --------------------------
            if type_of_style_input == "Upload Images" and not st.session_state["list_of_style_references"]:
                button_status = True
                help = "You need to upload at least one style reference."
            else:
                button_status = False
                help = ""

            st.markdown("***")

            def generate_images(
                ml_client,
                project_uuid,
                shot_uuid,
                timing_uuid,
                position,
                prompts_to_be_processed,
                images_per_prompt,
                type_of_model,
                model,
                lightning,
                project_settings,
                additional_description_text,
                additional_style_text,
                negative_prompt,
                list_of_strengths,
                default_form_values,
            ):
                input_image_file_list = []
                atleast_one_log_created = False
                for img in st.session_state.get("list_of_style_references", []):
                    input_image_file = save_new_image(img, project_uuid)
                    input_image_file_list.append(input_image_file)

                for _, image_prompt in enumerate(prompts_to_be_processed):
                    for _ in range(images_per_prompt):
                        if type_of_model == T2IModel.SDXL.value:
                            data = {
                                "shot_uuid": shot_uuid,
                                "additional_description_text": additional_description_text,
                                "additional_style_text": additional_style_text,
                                "sdxl_model": model,
                                "lightning": lightning,
                                "width": project_settings.width,
                                "height": project_settings.height,
                            }

                            if input_image_file_list and len(input_image_file_list):
                                data["img_uuid_list"] = json.dumps([f.uuid for f in input_image_file_list])

                            file_data = {}
                            for idx, f in enumerate(input_image_file_list):
                                file_data[f"image_file_{idx}"] = {"uuid": f.uuid, "dest": "input/"}

                            query_obj = MLQueryObject(
                                timing_uuid=None,
                                image_uuid=None,
                                guidance_scale=5,
                                seed=-1,
                                num_inference_steps=30,
                                strength=list_of_strengths,
                                adapter_type=None,
                                prompt=f"{image_prompt}, {additional_description_text}, {additional_style_text}",
                                negative_prompt=negative_prompt,
                                height=project_settings.height,
                                width=project_settings.width,
                                data=data,
                                file_data=file_data,
                            )

                            output, log = ml_client.predict_model_output_standardized(
                                ML_MODEL.creative_image_gen,
                                query_obj,
                                queue_inference=QUEUE_INFERENCE_QUERIES,
                            )

                        elif type_of_model == T2IModel.SD3.value:
                            query_obj = MLQueryObject(
                                timing_uuid=None,
                                model_uuid=None,
                                image_uuid=None,
                                guidance_scale=5,
                                seed=-1,
                                num_inference_steps=30,
                                strength=5.0,
                                adapter_type=None,
                                prompt=f"{image_prompt}, {additional_description_text}, {additional_style_text}",
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

                        elif type_of_model == T2IModel.FLUX.value:
                            query_obj = MLQueryObject(
                                timing_uuid=None,
                                model_uuid=None,
                                image_uuid=None,
                                guidance_scale=5,
                                seed=-1,
                                num_inference_steps=30,
                                strength=5.0,
                                adapter_type=None,
                                prompt=f"{image_prompt}, {additional_description_text}, {additional_style_text}",
                                negative_prompt=negative_prompt,
                                height=project_settings.height,
                                width=project_settings.width,
                                data={"shift": 3.0, "model": model},  # default value
                            )

                            output, log = ml_client.predict_model_output_standardized(
                                ML_MODEL.flux,
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

                            if not atleast_one_log_created:
                                atleast_one_log_created = True
                                data_dict = {}
                                for k, _ in default_form_values.items():
                                    if k in st.session_state:
                                        data_dict[k] = st.session_state[k]

                                key = project_uuid
                                with sqlite_atomic_transaction():
                                    project: InternalProjectObject = data_repo.get_project_from_uuid(uuid=key)
                                    if project:
                                        meta_data = json.loads(project.meta_data) if project.meta_data else {}
                                        meta_data[ProjectMetaData.INSP_VALUES.value] = data_dict
                                        data_repo.update_project(
                                            uuid=project.uuid, meta_data=json.dumps(meta_data)
                                        )

            # In the part of the code where you create the button:
            if st.button(
                "Generate images",
                type="primary",
                disabled=button_status,
                help=help,
                on_click=generate_images,
                args=(
                    get_ml_client(),
                    project_uuid,
                    shot_uuid,
                    timing_uuid,
                    position,
                    prompts_to_be_processed,
                    images_per_prompt,
                    type_of_model,
                    model,
                    lightning,
                    project_settings,
                    additional_description_text,
                    additional_style_text,
                    negative_prompt,
                    list_of_strengths,
                    default_form_values,
                ),
            ):
                st.success("Images added to queue for processing!")
                refresh_app()

        st.write("")
