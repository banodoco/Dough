import json
import streamlit as st
import uuid

from typing import List
from shared.constants import AIModelCategory, AIModelType, ViewType
from ui_components.methods.ml_methods import trigger_restyling_process
from ui_components.models import InternalAIModelObject, InternalFrameTimingObject, InternalSettingObject
from utils.constants import ImageStage
from utils.data_repo.data_repo import DataRepo


def styling_element(timing_uuid, view_type=ViewType.SINGLE.value):
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(timing.project.uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(timing.project.uuid)

    # -------------------- Transfomation Stage -------------------- #
    stages = ImageStage.value_list()
    if view_type == ViewType.SINGLE.value:
        append_to_item_name = f"{timing_uuid}"
    elif view_type == ViewType.LIST.value:
        append_to_item_name = str(timing.project.uuid)
        st.markdown("## Batch queries")

    if project_settings.default_stage:
        if f'index_of_which_stage_to_run_on_{append_to_item_name}' not in st.session_state:
            st.session_state["transformation_stage"] = project_settings.default_stage
            st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(
                st.session_state["transformation_stage"])
    else:
        st.session_state["transformation_stage"] = ImageStage.SOURCE_IMAGE.value
        st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = 0


    stages1, stages2 = st.columns([1, 1])
    with stages1:
        st.session_state["transformation_stage"] = st.radio(
            "What stage of images would you like to run styling on?", 
            options=stages, 
            horizontal=True, 
            key=f"image_stage_selector_{append_to_item_name}",
            index=st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'], 
            help="Extracted frames means the original frames from the video."
        )

    with stages2:
        image = None
        if st.session_state["transformation_stage"] == ImageStage.SOURCE_IMAGE.value:
            source_img = timing_details[st.session_state['current_frame_index'] - 1].source_image
            image = source_img.location if source_img else ""
        elif st.session_state["transformation_stage"] == ImageStage.MAIN_VARIANT.value:
            image = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
        
        if image:
            st.image(image, use_column_width=True, caption=f"Image {st.session_state['current_frame_index']}")
        elif not image and st.session_state["transformation_stage"] in [ImageStage.SOURCE_IMAGE.value, ImageStage.MAIN_VARIANT.value]:
            st.error(f"No {st.session_state['transformation_stage']} image found for this variant")

    if stages.index(st.session_state["transformation_stage"]) != st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}']:
        st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(st.session_state["transformation_stage"])
        st.rerun()

    
    # -------------------- Model Selection -------------------- #
    if st.session_state["transformation_stage"] != ImageStage.NONE.value:
        st.session_state['add_image_in_params'] = True
        model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.IMG2IMG.value], custom_trained=False)
    else:
        # when NONE is selected then removing input image from the session state
        st.session_state['add_image_in_params'] = False
        model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)

    model_name_list = [m.name for m in model_list]

    if not ('index_of_default_model' in st.session_state and st.session_state['index_of_default_model']):
        if timing.model:
            st.session_state['model'] = timing.model.uuid
            st.session_state['index_of_default_model'] = next((i for i, obj in enumerate(
                model_list) if getattr(obj, 'uuid') == timing.model.uuid), 0)
            # st.write(
            #     f"Index of last model: {st.session_state['index_of_default_model']}")
        else:
            st.session_state['index_of_default_model'] = 0

    # resetting index on list change
    if st.session_state['index_of_default_model'] >= len(model_list):
        st.session_state['index_of_default_model'] = 0

    selected_model_name = st.selectbox(
        f"Which model would you like to use?", model_name_list, index=st.session_state['index_of_default_model'])
    st.session_state['model'] = next((obj.uuid for i, obj in enumerate(
        model_list) if getattr(obj, 'name') == selected_model_name), None)

    selected_model_index = next((i for i, obj in enumerate(
        model_list) if getattr(obj, 'name') == selected_model_name), None)
    if st.session_state['index_of_default_model'] != selected_model_index:
        st.session_state['index_of_default_model'] = selected_model_index
        st.rerun()

    current_model_name = data_repo.get_ai_model_from_uuid(
        st.session_state['model']).name

    # -------------------- Model Params (e.g. adapter type for controlnet) -------------------- #
    if current_model_name == AIModelCategory.CONTROLNET.value:
        controlnet_adapter_types = [
            "scribble", "normal", "canny", "hed", "seg", "hough", "depth2img", "pose"]
        if 'index_of_controlnet_adapter_type' not in st.session_state:
            if timing.adapter_type:
                st.session_state['index_of_controlnet_adapter_type'] = controlnet_adapter_types.index(
                    timing.adapter_type)
            else:
                st.session_state['index_of_controlnet_adapter_type'] = 0

        st.session_state['adapter_type'] = st.selectbox(
            f"Adapter Type", controlnet_adapter_types, index=st.session_state['index_of_controlnet_adapter_type'])

        if st.session_state['index_of_controlnet_adapter_type'] != controlnet_adapter_types.index(st.session_state['adapter_type']):
            st.session_state['index_of_controlnet_adapter_type'] = controlnet_adapter_types.index(
                st.session_state['adapter_type'])
            st.rerun()
        st.session_state['custom_models'] = []

        if not ( 'adapter_type' in st.session_state and st.session_state['adapter_type']):
            st.session_state['adapter_type'] = 'N'

        # setting default values of low and high threshold
        if st.session_state['adapter_type'] in ["canny", "pose"]:
            canny1, canny2 = st.columns(2)
            if view_type == ViewType.LIST.value:
                if project_settings.default_low_threshold != "":
                    low_threshold_value = project_settings.default_low_threshold
                else:
                    low_threshold_value = 50

                if project_settings.default_high_threshold != "":
                    high_threshold_value = project_settings.default_high_threshold
                else:
                    high_threshold_value = 150

            elif view_type == ViewType.SINGLE.value:
                if timing.low_threshold != "":
                    low_threshold_value = timing.low_threshold
                else:
                    low_threshold_value = 50

                if timing.high_threshold != "":
                    high_threshold_value = timing.high_threshold
                else:
                    high_threshold_value = 150

            with canny1:
                st.session_state['low_threshold'] = st.slider(
                    'Low Threshold', 0, 255, value=int(low_threshold_value))
            with canny2:
                st.session_state['high_threshold'] = st.slider(
                    'High Threshold', 0, 255, value=int(high_threshold_value))
        else:
            st.session_state['low_threshold'] = 0
            st.session_state['high_threshold'] = 0

    elif current_model_name == AIModelCategory.LORA.value:
        if not ('index_of_lora_model_1' in st.session_state and st.session_state['index_of_lora_model_1']):
            st.session_state['index_of_lora_model_1'] = 0
            st.session_state['index_of_lora_model_2'] = 0
            st.session_state['index_of_lora_model_3'] = 0

        lora_model_list = data_repo.get_all_ai_model_list(
            model_category_list=[AIModelCategory.LORA.value], custom_trained=True)
        null_model = InternalAIModelObject(
            None, "", None, None, None, None, None, None, None, None, None, None)
        lora_model_list.insert(0, null_model)
        lora_model_name_list = [m.name for m in lora_model_list]

        # TODO: remove this array from db table
        custom_models = []
        selected_lora_1_name = st.selectbox(
            f"LoRA Model 1", lora_model_name_list, index=st.session_state['index_of_lora_model_1'], key="lora_1")
        st.session_state['lora_model_1'] = next((obj.uuid for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_1_name), "")
        st.session_state['lora_model_1_url'] = next((obj.replicate_url for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_1_name), "")
        selected_lora_1_index = next((i for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_1_name), 0)

        if st.session_state['index_of_lora_model_1'] != selected_lora_1_index:
            st.session_state['index_of_lora_model_1'] = selected_lora_1_index

        if st.session_state['index_of_lora_model_1'] != 0:
            custom_models.append(st.session_state['lora_model_1'])

        selected_lora_2_name = st.selectbox(
            f"LoRA Model 2", lora_model_name_list, index=st.session_state['index_of_lora_model_2'], key="lora_2")
        st.session_state['lora_model_2'] = next((obj.uuid for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_2_name), "")
        st.session_state['lora_model_2_url'] = next((obj.replicate_url for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_2_name), "")
        selected_lora_2_index = next((i for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_2_name), 0)

        if st.session_state['index_of_lora_model_2'] != selected_lora_2_index:
            st.session_state['index_of_lora_model_2'] = selected_lora_2_index

        if st.session_state['index_of_lora_model_2'] != 0:
            custom_models.append(st.session_state['lora_model_2'])

        selected_lora_3_name = st.selectbox(
            f"LoRA Model 3", lora_model_name_list, index=st.session_state['index_of_lora_model_3'], key="lora_3")
        st.session_state['lora_model_3'] = next((obj.uuid for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_3_name), "")
        st.session_state['lora_model_3_url'] = next((obj.replicate_url for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_3_name), "")
        selected_lora_3_index = next((i for i, obj in enumerate(
            lora_model_list) if getattr(obj, 'name') == selected_lora_3_name), 0)

        if st.session_state['index_of_lora_model_3'] != selected_lora_3_index:
            st.session_state['index_of_lora_model_3'] = selected_lora_3_index

        if st.session_state['index_of_lora_model_3'] != 0:
            custom_models.append(st.session_state['lora_model_3'])

        st.session_state['custom_models'] = json.dumps(custom_models)

        st.info("You can reference each model in your prompt using the following keywords: <1>, <2>, <3> - for example '<1> in the style of <2>.")

        lora_adapter_types = ['sketch', 'seg', 'keypose', 'depth', None]
        if f"index_of_lora_adapter_type_{append_to_item_name}" not in st.session_state:
            st.session_state['index_of_lora_adapter_type'] = 0

        st.session_state['adapter_type'] = st.selectbox(
            f"Adapter Type:", lora_adapter_types, help="This is the method through the model will infer the shape of the object. ", index=st.session_state['index_of_lora_adapter_type'])

        if st.session_state['index_of_lora_adapter_type'] != lora_adapter_types.index(st.session_state['adapter_type']):
            st.session_state['index_of_lora_adapter_type'] = lora_adapter_types.index(
                st.session_state['adapter_type'])

    elif current_model_name == AIModelCategory.DREAMBOOTH.value:
        # df = pd.read_csv('models.csv')
        # filtered_df = df[df.iloc[:, 5] == 'Dreambooth']
        # dreambooth_model_list = filtered_df.iloc[:, 0].tolist()

        dreambooth_model_list = data_repo.get_all_ai_model_list(
            model_category_list=[AIModelCategory.DREAMBOOTH.value], custom_trained=True)
        dreambooth_model_name_list = [m.name for m in dreambooth_model_list]

        if not ('index_of_dreambooth_model' in st.session_state and st.session_state['index_of_dreambooth_model']):
            st.session_state['index_of_dreambooth_model'] = 0

        selected_dreambooth_model_name = st.selectbox(
            f"Dreambooth Model", dreambooth_model_name_list, index=st.session_state['index_of_dreambooth_model'])
        # st.session_state['custom_models'] = next((obj.uuid for i, obj in enumerate(
        #     dreambooth_model_list) if getattr(obj, 'name') == selected_dreambooth_model_name), "")
        selected_dreambooth_model_index = next((i for i, obj in enumerate(
            dreambooth_model_list) if getattr(obj, 'name') == selected_dreambooth_model_name), 0)
        if st.session_state['index_of_dreambooth_model'] != selected_dreambooth_model_index:
            st.session_state['index_of_dreambooth_model'] = selected_dreambooth_model_index

        if len(dreambooth_model_list):
            st.session_state['dreambooth_model_uuid'] = dreambooth_model_list[st.session_state['index_of_dreambooth_model']].uuid
    else:
        st.session_state['custom_models'] = []
        st.session_state['adapter_type'] = "N"

    # -------------------- Prompt -------------------- #
    if st.session_state['model'] == "StyleGAN-NADA":
        # only certain words are available in case of stylegan-nada
        st.warning("StyleGAN-NADA is a custom model that uses StyleGAN to generate a consistent character and style transformation. It only works for square images.")
        st.session_state['prompt'] = st.selectbox("What style would you like to apply to the character?", ['base', 'mona_lisa', 'modigliani', 'cubism', 'elf', 'sketch_hq', 'thomas', 'thanos', 'simpson', 'witcher',
                                                  'edvard_munch', 'ukiyoe', 'botero', 'shrek', 'joker', 'pixar', 'zombie', 'werewolf', 'groot', 'ssj', 'rick_morty_cartoon', 'anime', 'white_walker', 'zuckerberg', 'disney_princess', 'all', 'list'])
        st.session_state['strength'] = 0.5
        st.session_state['guidance_scale'] = 7.5
        st.session_state['seed'] = int(0)
        st.session_state['num_inference_steps'] = int(50)

    else:
        if not (f'prompt_value_{append_to_item_name}' in st.session_state and st.session_state[f'prompt_value_{append_to_item_name}']):
            if timing.prompt:
                st.session_state[f'prompt_value_{append_to_item_name}'] = timing.prompt
            else:
                st.session_state[f'prompt_value_{append_to_item_name}'] = ""

        st.session_state['prompt'] = st.text_area(f"Prompt", label_visibility="visible", 
                                                  value=st.session_state[f'prompt_value_{append_to_item_name}'], height=150)
        
        if st.session_state['prompt'] != st.session_state[f'prompt_value_{append_to_item_name}']:
            st.session_state[f'prompt_value_{append_to_item_name}'] = st.session_state['prompt']
            st.rerun()

        if view_type == ViewType.LIST.value:
            st.info(
                "You can include the following tags in the prompt to vary the prompt dynamically: [expression], [location], [mouth], and [looking]")
        
        if st.session_state['model'] == AIModelCategory.DREAMBOOTH.value:
            model_details: InternalAIModelObject = data_repo.get_ai_model_from_uuid(
                st.session_state['dreambooth_model_uuid'])
            st.info(
                f"Must include '{model_details.keyword}' to run this model")
            # TODO: CORRECT-CODE add controller_type to ai_model
            if model_details.controller_type != "":
                st.session_state['adapter_type'] = st.selectbox(
                    f"Would you like to use the {model_details.controller_type} controller?", ['Yes', 'No'])
            else:
                st.session_state['adapter_type'] = "No"

        else:
            if st.session_state['model'] == AIModelCategory.PIX_2_PIX.value:
                st.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")

        if view_type == ViewType.LIST.value:
            if project_settings.default_strength:
                st.session_state['strength'] = project_settings.default_strength
            else:
                st.session_state['strength'] = 0.5

        elif view_type == ViewType.SINGLE.value:
            if timing.strength:
                st.session_state['strength'] = timing.strength
            else:
                st.session_state['strength'] = 0.5

        st.session_state['strength'] = st.slider(f"Strength", value=float(
            st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)

        if view_type == ViewType.LIST.value:
            if project_settings.default_guidance_scale:
                st.session_state['guidance_scale'] = project_settings.default_guidance_scale
            else:
                st.session_state['guidance_scale'] = 7.5
        elif view_type == ViewType.SINGLE.value:
            if timing.guidance_scale != "":
                st.session_state['guidance_scale'] = timing.guidance_scale
            else:
                st.session_state['guidance_scale'] = 7.5

        if not ('negative_prompt_value' in st.session_state and st.session_state['negative_prompt_value']) and timing.negative_prompt:
            st.session_state['negative_prompt_value'] = timing.negative_prompt

        st.session_state['negative_prompt'] = st.text_area(
            f"Negative prompt", value=st.session_state['negative_prompt_value'], label_visibility="visible")
        
        if st.session_state['negative_prompt'] != st.session_state['negative_prompt_value']:
            st.session_state['negative_prompt_value'] = st.session_state['negative_prompt']
            st.rerun()
        
        st.session_state['guidance_scale'] = st.number_input(
            f"Guidance scale", value=float(st.session_state['guidance_scale']))
        
        if view_type == ViewType.LIST.value:
            if project_settings.default_seed != "":
                st.session_state['seed'] = project_settings.default_seed
            else:
                st.session_state['seed'] = 0

        elif view_type == ViewType.SINGLE.value:
            if timing.seed != "":
                st.session_state['seed'] = timing.seed
            else:
                st.session_state['seed'] = 0

        st.session_state['seed'] = st.number_input(
            f"Seed", value=int(st.session_state['seed']))
        
        if view_type == ViewType.LIST.value:
            if project_settings.default_num_inference_steps:
                st.session_state['num_inference_steps'] = project_settings.default_num_inference_steps
            else:
                st.session_state['num_inference_steps'] = 50
        elif view_type == ViewType.SINGLE.value:
            if timing.num_inference_steps:
                st.session_state['num_inference_steps'] = timing.num_inference_steps
            else:
                st.session_state['num_inference_steps'] = 50
        st.session_state['num_inference_steps'] = st.number_input(
            f"Inference steps", value=int(st.session_state['num_inference_steps']))

    st.session_state["use_new_settings"] = True

    st.session_state["promote_new_generation"] = st.checkbox(
            "Promote new generation to main variant", key="promote_new_generation_to_main_variant")

    if view_type == ViewType.LIST.value:
        batch_run_range = st.slider(
            "Select range:", 1, 1, (1, len(timing_details)))
        first_batch_run_value = batch_run_range[0] - 1
        last_batch_run_value = batch_run_range[1] - 1

        st.write(batch_run_range)
        
        st.session_state["use_new_settings"] = st.checkbox(
            "Use new settings for batch query", key="keep_existing_settings", help="If unchecked, the new settings will be applied to the existing variants.")

        if 'restyle_button' not in st.session_state:
            st.session_state['restyle_button'] = ''
            st.session_state['item_to_restyle'] = ''

        btn1, btn2 = st.columns(2)

        with btn1:

            batch_number_of_variants = st.number_input(
                "How many variants?", value=1, min_value=1, max_value=10, step=1, key="number_of_variants")

        with btn2:

            st.write("")
            st.write("")
            if st.button(f'Batch restyle') or st.session_state['restyle_button'] == 'yes':
                if st.session_state['restyle_button'] == 'yes':
                    range_start = int(st.session_state['item_to_restyle'])
                    range_end = range_start + 1
                    st.session_state['restyle_button'] = ''
                    st.session_state['item_to_restyle'] = ''

                for i in range(first_batch_run_value, last_batch_run_value+1):
                    for _ in range(0, batch_number_of_variants):
                        trigger_restyling_process(
                            timing_uuid=timing_details[i].uuid, 
                            model_uuid=st.session_state['model'],
                            prompt=st.session_state['prompt'], 
                            strength=st.session_state['strength'], 
                            negative_prompt=st.session_state['negative_prompt'], 
                            guidance_scale=st.session_state['guidance_scale'], 
                            seed=st.session_state['seed'], 
                            num_inference_steps=st.session_state['num_inference_steps'], 
                            transformation_stage=st.session_state['transformation_stage'], 
                            promote_new_generation=st.session_state["promote_new_generation"], 
                            custom_models=st.session_state['custom_models'], 
                            adapter_type=st.session_state['adapter_type'], 
                            update_inference_settings=True, 
                            low_threshold=st.session_state['low_threshold'], 
                            high_threshold=st.session_state['high_threshold'],
                            canny_image=st.session_state['canny_image'] if 'canny_image' in st.session_state else None,
                            add_image_in_params=st.session_state['add_image_in_params'],
                        )
                        
                st.rerun()

