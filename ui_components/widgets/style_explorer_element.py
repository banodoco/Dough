import json
import streamlit as st
from ui_components.methods.common_methods import process_inference_output
from ui_components.methods.file_methods import generate_pil_image
from ui_components.methods.ml_methods import query_llama2
from ui_components.widgets.add_key_frame_element import add_key_frame
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from shared.constants import AIModelType, InferenceType, InternalFileTag, InternalFileType, SortOrder

from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import REPLICATE_MODEL


def style_explorer_element(project_uuid):
    st.markdown("***")
    data_repo = DataRepo() 
    project_settings = data_repo.get_project_setting(project_uuid)

    _, a2, a3,_= st.columns([0.5, 1, 0.5,0.5])    
    prompt = a2.text_area("What's your base prompt?", key="prompt", help="This will be included at the beginning of each prompt")
    base_prompt_position = a3.radio("Where would you like to place the base prompt?", options=["Beginning", "End"], key="base_prompt_position", help="This will be included at the beginning of each prompt")

    _, b2, b3, b4, b5, _ = st.columns([0.5, 1, 1, 1, 1, 0.5])    
    character_instructions = create_variate_option(b2, "character")  
    styling_instructions = create_variate_option(b3, "styling")          
    action_instructions = create_variate_option(b4, "action")    
    scene_instructions = create_variate_option(b5, "scene")

    model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)
    model_dict = {}
    for m in model_list:
        model_dict[m.name] = m

    model_name_list = list(model_dict.keys())
    
    _, c2, _ = st.columns([0.25, 1, 0.25])
    with c2:
        models_to_use = st.multiselect("Which models would you like to use?", model_name_list, key="models_to_use", default=model_name_list, help="It'll rotate through the models you select.")

    _, d2, _ = st.columns([0.75, 1, 0.75])
    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=1, max_value=100, value=10, step=1, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    _, e2, _ = st.columns([0.5, 1, 0.5])
    if e2.button("Generate images", key="generate_images", use_container_width=True, type="primary"):
        ml_client = get_ml_client()
        counter = 0
        num_models = len(models_to_use)
        num_images_per_model = number_to_generate // num_models
        varied_text = ""
        for _ in range(num_images_per_model):
            for model_name in models_to_use:
                if counter % 4 == 0 and (styling_instructions or character_instructions or action_instructions or scene_instructions):
                    varied_prompt = create_prompt(
                        styling_instructions=styling_instructions, 
                        character_instructions=character_instructions, 
                        action_instructions=action_instructions, 
                        scene_instructions=scene_instructions
                    )
                    varied_text = varied_prompt
                if base_prompt_position == "Beginning":
                    prompt_with_variations = f"{prompt}, {varied_text}" if prompt else varied_text
                else:  # base_prompt_position is "End"
                    prompt_with_variations = f"{varied_text}, {prompt}" if prompt else varied_text
                # st.write(f"Prompt: '{prompt_with_variations}'")
                # st.write(f"Model: {model_name}")
                counter += 1

                query_obj = MLQueryObject(
                    timing_uuid=None,
                    model_uuid=None,
                    guidance_scale=5,
                    seed=-1,
                    num_inference_steps=30,            
                    strength=1,
                    adapter_type=None,
                    prompt=prompt_with_variations,
                    negative_prompt="bad image, worst image, bad anatomy, washed out colors",
                    height=project_settings.height,
                    width=project_settings.width,
                )

                replicate_model = REPLICATE_MODEL.get_model_by_db_obj(model_dict[model_name])
                output, log = ml_client.predict_model_output_standardized(replicate_model, query_obj, queue_inference=True)

                inference_data = {
                    "inference_type": InferenceType.GALLERY_IMAGE_GENERATION.value,
                    "output": output,
                    "log_uuid": log.uuid,
                    "project_uuid": project_uuid
                }
                process_inference_output(**inference_data)
    
    project_setting = data_repo.get_project_setting(project_uuid)
    page_number = st.radio("Select page", options=range(1, project_setting.total_gallery_pages + 1), horizontal=True)
    num_items_per_page = 10

    gallery_image_list, res_payload = data_repo.get_all_file_list(
        file_type=InternalFileType.IMAGE.value, 
        tag=InternalFileTag.GALLERY_IMAGE.value, 
        project_id=project_uuid,
        page=page_number,
        data_per_page=20,
        sort_order=SortOrder.DESCENDING.value     # newly created images appear first
    )

    if project_setting.total_gallery_pages != res_payload['total_pages']:
        project_setting.total_gallery_pages = res_payload['total_pages']
        st.rerun()
    
    total_image_count = res_payload['count']

    if gallery_image_list and len(gallery_image_list):
        st.markdown("***")
        num_columns = st.slider('Number of columns', min_value=1, max_value=10, value=4)
        start_index = 0
        end_index = min(start_index + num_items_per_page, total_image_count)

        for i in range(start_index, end_index, num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(gallery_image_list):
                    with cols[j]:                        
                        st.image(gallery_image_list[i + j].location, use_column_width=True)
                        with st.expander(f'Variant #{(page_number - 1) * num_items_per_page + i + j + 1}', False):
                            if gallery_image_list[i + j].inference_log:
                                log = data_repo.get_inference_log_from_uuid(gallery_image_list[i + j].inference_log.uuid)
                                if log:
                                    input_params = json.loads(log.input_params)
                                    prompt = input_params.get('prompt', 'No prompt found')
                                    model = json.loads(log.output_details)['model_name'].split('/')[-1]
                                    st.info(f"Prompt: {prompt}")
                                    st.info(f"Model: {model}")
                                else:
                                    st.warning("No data found")
                            else:
                                st.warning("No data found")
                                    
                        if st.button(f"Add to timeline", key=f"Promote Variant #{(page_number - 1) * num_items_per_page + i + j + 1} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image", use_container_width=True):
                            pil_image = generate_pil_image(gallery_image_list[i + j].location)
                            add_key_frame(pil_image, False, 2.5, len(data_repo.get_timing_list_from_project(project_uuid)))
                            st.rerun()
            st.markdown("***")
    else:
        st.warning("No images present")


def create_variate_option(column, key):
    label = key.replace('_', ' ').capitalize()
    variate_option = column.checkbox(f"Vary {label.lower()}", key=f"{key}_checkbox")
    if variate_option:
        instructions = column.text_area(f"How would you like to vary the {label.lower()}?", key=f"{key}_textarea", help=f"It'll write a custom {label.lower()} prompt based on your instructions.")
    else:
        instructions = ""
    return instructions

def create_prompt(**kwargs):
        text_list = []
        order = ["character_instructions", "styling_instructions", "action_instructions", "scene_instructions"]


        system_instruction_template_list = {
            "character_instructions": "Input|Character Descriptions:\nSickly old man|Francois Leger,old Russian man, beaten-down look, wearing suit\nPretty young woman|Jules van Cohen,beautiful young woman, floral dress,vibrant\nIrish boy|James McCarthy,10 year old Irish boy,red hair,pink shirt,wheezing in a small voice\nYoung thug|Hughie Banks,23 y/o English football hooligan with skinned head",
            "styling_instructions": "Input|Style Description:\nmoody and emotion|watercolour style, dark colours and pastel tones.\nchildren's adventure|simple children's book illustration style with light colours\ngritty and realistic|Sin City style,black and white,realistic,strong lines.\nhighly abstract|abstract art style, vibrant colours and thick linework.",
            "action_instructions": "Input|Action Description:\ngoing on an adventure|exploring old ruins,with a flashlight\nbusy day in the city|walking through downtown at rushour\nfamily time|making dinner with the family\nbeing creepy|hiding in bushes,looking in window\nworking hard|finishing homework,late at night",
            "scene_instructions": "Input|Scene Description:\nForest|Misty woods with towering trees and glowing plants.\nFuturistic city|Skyscrapers, flying cars, neon lights in a futuristic metropolis.\nMedieval|Castle courtyard with knights, cobblestones, and a fountain.\nBeach|Golden sands, rolling waves, and a vibrant sunset.\nApocalypse|Ruined buildings and desolation in a bleak wasteland.",
        }

        for instruction_type in order:
            user_instruction = kwargs.get(instruction_type)
            if user_instruction and instruction_type in system_instruction_template_list:
                result = query_llama2(user_instruction, system_instruction_template_list[instruction_type])
                text_list.append(result)

        return ", ".join(text_list)