import streamlit as st
from ui_components.methods.common_methods import promote_image_variant
from ui_components.methods.ml_methods import query_llama2
from utils.data_repo.data_repo import DataRepo
from shared.constants import AIModelType
import replicate


def style_explorer_element(project_uuid):
    st.markdown("***")
    data_repo = DataRepo() 

    _, a2, _ = st.columns([0.5, 1, 0.5])    
    prompt = a2.text_area("What's your base prompt?", key="prompt", help="This will be included at the beginning of each prompt")

    _, b2, b3, b4, b5, _ = st.columns([0.5, 1, 1, 1, 1, 0.5])    
    character_instructions = create_variate_option(b2, "character")  
    styling_instructions = create_variate_option(b3, "styling")          
    action_instructions = create_variate_option(b4, "action")    
    scene_instructions = create_variate_option(b5, "scene")

    model_name_list = list(set([m.name for m in data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)]))
    
    _, c2, _ = st.columns([0.25, 1, 0.25])
    with c2:
        models_to_use = st.multiselect("Which models would you like to use?", model_name_list, key="models_to_use", default=model_name_list, help="It'll rotate through the models you select.")

    _, d2, _ = st.columns([0.75, 1, 0.75])
    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=1, max_value=100, value=10, step=1, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    _, e2, _ = st.columns([0.5, 1, 0.5])
    if e2.button("Generate images", key="generate_images", use_container_width=True, type="primary"):
        counter = 0
        varied_text = ""
        num_models = len(models_to_use)
        num_images_per_model = number_to_generate // num_models
        for _ in range(num_images_per_model):
            for model_name in models_to_use:
                if counter % 4 == 0 and (styling_instructions or character_instructions or action_instructions or scene_instructions):
                    varied_text = create_prompt(styling_instructions, character_instructions, action_instructions, scene_instructions)
                prompt_with_variations = f"{prompt}, {varied_text}" if prompt else varied_text
                st.write(f"Prompt: '{prompt_with_variations}'")
                st.write(f"Model: {model_name}")                 
                counter += 1
    
    timing = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
    variants = timing.alternative_images_list
    
    st.markdown("***")
    num_columns = st.slider('Number of columns', min_value=1, max_value=10, value=4)

    num_items_per_page = 30
    num_pages = len(variants) // num_items_per_page
    if len(variants) % num_items_per_page > 0:
        num_pages += 1  # Add extra page if there are remaining items

    page_number = st.radio("Select page", options=range(1, num_pages + 1))

    start_index = (page_number - 1) * num_items_per_page
    end_index = start_index + num_items_per_page

    for i in range(start_index, min(end_index, len(variants)), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(variants):
                with cols[j]:
                    st.image(variants[i + j].location, use_column_width=True)
                    with st.expander(f'Variant #{i + j + 1}', False):
                        st.info("Instructions: PLACEHOLDER")
                                
                    if st.button(f"Add to timeline", key=f"Promote Variant #{i + j + 1} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image", use_container_width=True):
                        promote_image_variant(timing.uuid, i + j)                                            
                        st.rerun()
        st.markdown("***")


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

        system_instruction_template_list = {
            "character_instructions": "Input|Character Descriptions:\nSickly old man|Francois Leger,old Russian man, beaten-down look, wearing suit\nPretty young woman|Jules van Cohen,beautiful young woman, floral dress,vibrant\nIrish boy|James McCarthy,10 year old Irish boy,red hair,pink shirt,wheezing in a small voice\nYoung thug|Hughie Banks,23 y/o English football hooligan with skinned head",
            "styling_instructions": "Input|Style Description:\nmoody and emotion|watercolour style, dark colours and pastel tones.\nchildren's adventure|simple children's book illustration style with light colours\ngritty and realistic|Sin City style,black and white,realistic,strong lines.\nhighly abstract|abstract art style, vibrant colours and thick linework.",
            "action_instructions": "Input|Action Description:\ngoing on an adventure|exploring old ruins,with a flashlight\nbusy day in the city|walking through downtown at rushour\nfamily time|making dinner with the family\nbeing creepy|hiding in bushes,looking in window\nworking hard|finishing homework,late at night",
            "scene_instructions": "Input|Scene Description:\nForest|Misty woods with towering trees and glowing plants.\nFuturistic city|Skyscrapers, flying cars, neon lights in a futuristic metropolis.\nMedieval|Castle courtyard with knights, cobblestones, and a fountain.\nBeach|Golden sands, rolling waves, and a vibrant sunset.\nApocalypse|Ruined buildings and desolation in a bleak wasteland.",
        }

        for instruction_type, user_instruction in kwargs:
            if instruction_type in system_instruction_template_list and user_instruction:
                result = query_llama2(user_instruction, system_instruction_template_list[instruction_type])
                text_list.append(result)

        return ", ".join(text_list)