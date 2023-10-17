import streamlit as st
from ui_components.methods.common_methods import promote_image_variant
from utils.data_repo.data_repo import DataRepo
from shared.constants import AIModelType


def mood_board_page(project_uuid):

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
    
    data_repo = DataRepo()
    st.subheader("Mood Board")
    a1, a2, a3 = st.columns([0.5, 1, 0.5])
    with a2:
        prompt = st.text_area("What's your prompt?", key="prompt")


    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        variate_styling = st.checkbox("Variate styling", key="variate_styling")
        if variate_styling:
            styling_instructions = st.text_area("How would you like to variate styling?", key="variate_styling_textarea")
        else:
            styling_instructions = ""

    with b2:
        variate_character = st.checkbox("Variate character", key="variate_character")
        if variate_character:
            character_instructions = st.text_area("How would you like to variate character?", key="variate_character_textarea")
        else:
            character_instructions = ""

    with b3:
        variate_action = st.checkbox("Variate action", key="variate_action")
        if variate_action:
            action_instructions = st.text_area("How would you like to variate action?", key="variate_action_textarea")
        else:
            action_instructions = ""

    with b4:
        variate_scene = st.checkbox("Variate scene", key="variate_scene")
        if variate_scene:
            scene_instructions = st.text_area("How would you like to variate the scene?", key="variate_scene_textarea")
        else:
            scene_instructions = ""

    model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)
    model_name_list = list(set([m.name for m in model_list]))
    
    c1, c2, c3 = st.columns([0.25, 1, 0.25])
    with c2:
        models_to_use = st.multiselect("Which models would you like to use?", model_name_list, key="models_to_use", default=model_name_list)

    d1, d2, d3 = st.columns([0.5, 1, 0.5])
    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=1, max_value=100, value=10, step=1, key="number_to_generate")  

    if st.button("Generate images", key="generate_images", use_container_width=True, type="primary"):
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