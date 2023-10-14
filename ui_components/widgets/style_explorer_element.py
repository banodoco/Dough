import streamlit as st
from utils.data_repo.data_repo import DataRepo
from shared.constants import AIModelType


def style_explorer_element(project_uuid):

    def get_varied_text(styling_instructions="", character_instructions="", action_instructions="", scene_instructions=""):
        text_list = []

        if styling_instructions:
            system_instructions = "PLACEHOLDER_STYLING"
            # result = query_model(styling_instructions, system_instructions)
            result = "Styling instructions"
            text_list.append(result)

        if character_instructions:
            system_instructions = "PLACEHOLDER_CHARACTER"
            # result = query_model(character_instructions, system_instructions)
            result = "Character instructions"
            text_list.append(result)

        if action_instructions:
            system_instructions = "PLACEHOLDER_ACTION"            
            # result = query_model(action_instructions, system_instructions)
            result = "Action instructions"
            text_list.append(result)

        if scene_instructions:
            system_instructions = "PLACEHOLDER_SCENE"
            # result = query_model(scene_instructions, system_instructions)
            result = "Scene instructions"
            text_list.append(result)

        return ", ".join(text_list)
    

    def create_variate_option(column, key):
        label = key.replace('_', ' ').capitalize()
        variate_option = column.checkbox(f"Vary {label.lower()}", key=f"{key}_checkbox")
        if variate_option:
            instructions = column.text_area(f"How would you like to vary the {label.lower()}?", key=f"{key}_textarea", help=f"It'll write a custom {label.lower()} prompt based on your instructions.")
        else:
            instructions = ""
        return instructions
    
    st.markdown("***")
    data_repo = DataRepo() 

    a1, a2, a3 = st.columns([0.5, 1, 0.5])    
    prompt = a2.text_area("What's your base prompt?", key="prompt", help="This will be included at the beginning of each prompt")

    b1, b2, b3, b4,b5,b6 = st.columns([0.5, 1, 1, 1, 1, 0.5])
    
    styling_instructions = create_variate_option(b2, "styling")    
    character_instructions = create_variate_option(b3, "character")    
    action_instructions = create_variate_option(b4, "action")    
    scene_instructions = create_variate_option(b5, "scene")

    model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)
    model_name_list = list(set([m.name for m in model_list]))
    
    c1, c2, c3 = st.columns([0.25, 1, 0.25])
    
    with c2:
        models_to_use = st.multiselect("Which models would you like to use?", model_name_list, key="models_to_use", default=model_name_list, help="It'll rotate through the models you select.")

    d1, d2, d3 = st.columns([0.5, 1, 0.5])

    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=1, max_value=100, value=10, step=1, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    e1, e2, e3 = st.columns([0.5, 1, 0.5])

    if e2.button("Generate images", key="generate_images", use_container_width=True, type="primary"):
        st.info("Generating images...")
        counter = 0
        varied_text = ""
        for _ in range(number_to_generate):
            for model_name in models_to_use:
                if counter % 4 == 0 and (styling_instructions or character_instructions or action_instructions or scene_instructions):
                    varied_text = get_varied_text(styling_instructions, character_instructions, action_instructions, scene_instructions)
                prompt_with_variations = f"{prompt}, {varied_text}" if prompt else varied_text
                st.write(f"Prompt: '{prompt_with_variations}'")
                st.write(f"Model: {model_name}")                 
                counter += 1
    
    timing = data_repo.get_timing_from_uuid("c414f700-680b-4712-a9c5-22c9935d7855")

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