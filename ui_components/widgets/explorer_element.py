import json
import streamlit as st
from ui_components.methods.common_methods import process_inference_output,add_new_shot, save_uploaded_image
from ui_components.methods.file_methods import generate_pil_image
from ui_components.methods.ml_methods import query_llama2
from ui_components.widgets.add_key_frame_element import add_key_frame
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from shared.constants import QUEUE_INFERENCE_QUERIES, AIModelType, InferenceType, InternalFileTag, InternalFileType, SortOrder
from utils import st_memory
import time
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.replicate.constants import REPLICATE_MODEL
from PIL import Image, ImageFilter
import io
import cv2
import numpy as np





def explorer_element(project_uuid):
    st.markdown("***")
    data_repo = DataRepo()
    shot_list = data_repo.get_shot_list(project_uuid)
    project_settings = data_repo.get_project_setting(project_uuid)
    
    z1, z2 = st.columns([1,1])   
    with z1:
        with st.expander("Prompt Settings", expanded=True):
            a1, a2 = st.columns([1,1])   
            with a1:
                prompt = st_memory.text_area("What's your base prompt?", key="explorer_base_prompt", help="This exact text will be included in each prompt.")
            with a2:
                magic_prompt = st_memory.text_area("What's your magic prompt?", key="explorer_magic_prompt", help="A prompt will be generated based on this text.")
            
            y1,y2 = st.columns([1,1])            
            with y1:
                base_prompt_position = st_memory.radio("Which would you like to put first:", options=["Base Prompt", "Magic Prompt"], key="base_prompt_position", help="This will be included at the beginning of each prompt", horizontal=True)
            with y2:
                if magic_prompt != "":
                    chaos_level = st_memory.slider("How much chaos would you like to add to the magic prompt?", min_value=0, max_value=100, value=20, step=1, key="chaos_level", help="This will determine how random the generated prompt will be.")                    
                    temperature = chaos_level / 20
            
            c1, _ = st.columns([1, 1])
            with c1:
                negative_prompt = st_memory.text_area("Negative prompt", value="bad image, worst image, bad anatomy, washed out colors",\
                                                       key="explorer_neg_prompt", \
                                                        help="These are the things you wish to be excluded from the image")
    with z2:                
            with st.expander("Input Image Settings", expanded=True):
                use_input_image = st_memory.checkbox("Use input image", key="use_input_image", value=False)
                edge_pil_img = None
                if use_input_image:
                    a_1_1, a_1_2 = st.columns([1, 1])
                    with a_1_1:
                        input_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="explorer_input_image", help="This will be the base image for the generation.")
                        type_of_transformation = st_memory.radio("What type of transformation would you like to do?", options=["Evolve Image", "Maintain Structure"], key="type_of_transformation", help="Style Transfer will transfer the style of the image to the target image. Image Generation will generate an image based on the target image.", horizontal=True)
                    with a_1_2:                                        
                        if type_of_transformation == "Evolve Image":
                            strength_of_current_image = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_current_image", help="This will determine how much of the current image will be kept in the final image.")
                            # prompt_strength should be 1 - strength_of_current_image
                            prompt_strength = round(1 - (strength_of_current_image / 100), 2)

                            if input_image is not None:                                
                                input_image_bytes = input_image.getvalue()
                                pil_image = Image.open(io.BytesIO(input_image_bytes))
                                blur_radius = (100 - strength_of_current_image) / 3  # Adjust this formula as needed
                                blurred_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
                                st.image(blurred_image, use_column_width=True)

                        elif type_of_transformation == "Maintain Structure":
                            strength_of_current_image = st_memory.slider("How much of the current structure would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_current_image", help="This will determine how much of the current image will be kept in the final image.")                                        
                            condition_scale = strength_of_current_image / 10                                                

                            if input_image is not None:
                                input_image_bytes = input_image.getvalue()
                                pil_image = Image.open(io.BytesIO(input_image_bytes))
                                cv_image = np.array(pil_image)
                                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
                                lower_threshold = (100 - strength_of_current_image) * 3
                                upper_threshold = lower_threshold * 3
                                edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
                                edge_pil_img = Image.fromarray(edges)
                                st.image(edge_pil_img, use_column_width=True)
                else:
                    input_image = None
                    type_of_transformation = None
                    strength_of_current_image = None
                    prompt_strength = None
                    condition_scale = None
                    temperature = None

    models_to_use = ["stable_diffusion_xl"]
    _, d2,d3, _ = st.columns([0.5, 1,1, 0.5])
    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=0, max_value=100, value=4, step=4, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    with d3:
        st.write(" ")
        st.write(" ")        
        if st.button("Generate images", key="generate_images", use_container_width=True, type="primary"):
            ml_client = get_ml_client()
            counter = 0
            num_models = len(models_to_use)
            num_images_per_model = number_to_generate // num_models
            varied_text = ""
            for _ in range(num_images_per_model):
                for model_name in models_to_use:
                    if counter % 4 == 0:
                        varied_prompt = ""
                        varied_text = varied_prompt
                    if base_prompt_position == "Base Prompt":
                        prompt_with_variations = f"{prompt}, {varied_text}" if prompt else varied_text
                    else:  # base_prompt_position is "End"
                        prompt_with_variations = f"{varied_text}, {prompt}" if prompt else varied_text
                    # st.write(f"Prompt: '{prompt_with_variations}'")
                    # st.write(f"Model: {model_name}")
                    counter += 1
                    log = None
                    if not input_image:
                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            model_uuid=None,
                            guidance_scale=5,
                            seed=-1,                            
                            num_inference_steps=30,            
                            strength=1,
                            adapter_type=None,
                            prompt=prompt_with_variations,
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            project_uuid=project_uuid
                        )

                        model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)
                        model_dict = {}
                        for m in model_list:
                            model_dict[m.name] = m

                        replicate_model = REPLICATE_MODEL.get_model_by_db_obj(model_dict[model_name])
                        output, log = ml_client.predict_model_output_standardized(replicate_model, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                    else:
                        if type_of_transformation == "Evolve Image":
                            input_image_file = save_uploaded_image(input_image, project_uuid)
                            query_obj = MLQueryObject(
                                timing_uuid=None,
                                model_uuid=None,
                                image_uuid=input_image_file.uuid,
                                guidance_scale=5,
                                seed=-1,
                                num_inference_steps=30,
                                strength=prompt_strength,
                                adapter_type=None,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                height=project_settings.height,
                                width=project_settings.width,
                                project_uuid=project_uuid
                            )

                            output, log = ml_client.predict_model_output_standardized(REPLICATE_MODEL.sdxl, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                        elif type_of_transformation == "Maintain Structure":
                            input_image_file = save_uploaded_image(edge_pil_img, project_uuid)
                            query_obj = MLQueryObject(
                                timing_uuid=None,
                                model_uuid=None,
                                image_uuid=input_image_file.uuid,
                                guidance_scale=5,
                                seed=-1,
                                num_inference_steps=30,
                                strength=0.5,
                                adapter_type=None,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                height=project_settings.height,
                                width=project_settings.width,
                                project_uuid=project_uuid,
                                data={'condition_scale': condition_scale}
                            )

                            output, log = ml_client.predict_model_output_standardized(REPLICATE_MODEL.sdxl_controlnet, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                    if log:
                        inference_data = {
                            "inference_type": InferenceType.GALLERY_IMAGE_GENERATION.value,
                            "output": output,
                            "log_uuid": log.uuid,
                            "project_uuid": project_uuid
                        }
                        process_inference_output(**inference_data)

            st.info("Check the Generation Log to the left for the status.")

    project_setting = data_repo.get_project_setting(project_uuid)
    st.markdown("***")
    
    f1,f2 = st.columns([1, 1])
    num_columns = f1.slider('Number of columns:', min_value=3, max_value=7, value=5)
    num_items_per_page = f2.slider('Items per page:', min_value=10, max_value=50, value=20)
    st.markdown("***")

    tab1, tab2 = st.tabs(["Explorations", "Shortlist"])
    with tab1:
        k1,k2 = st.columns([5,1])
        page_number = k1.radio("Select page", options=range(1, project_setting.total_gallery_pages + 1), horizontal=True, key="main_gallery")
        open_detailed_view_for_all = k2.toggle("Open detailed view for all:", key='main_gallery_toggle')
        gallery_image_view(project_uuid, page_number, num_items_per_page, open_detailed_view_for_all, False, num_columns)
    with tab2:
        k1,k2 = st.columns([5,1])
        shortlist_page_number = k1.radio("Select page", options=range(1, project_setting.total_shortlist_gallery_pages + 1), horizontal=True, key="shortlist_gallery")
        with k2:
            open_detailed_view_for_all = st_memory.toggle("Open prompt details for all:", key='shortlist_gallery_toggle')
        gallery_image_view(project_uuid, shortlist_page_number, num_items_per_page, open_detailed_view_for_all, True, num_columns)

def gallery_image_view(project_uuid,page_number=1,num_items_per_page=20, open_detailed_view_for_all=False, shortlist=False, num_columns=2, view="main"):
    data_repo = DataRepo()
    
    project_settings = data_repo.get_project_setting(project_uuid)
    shot_list = data_repo.get_shot_list(project_uuid)
    
    gallery_image_list, res_payload = data_repo.get_all_file_list(
        file_type=InternalFileType.IMAGE.value, 
        tag=InternalFileTag.GALLERY_IMAGE.value if not shortlist else InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value, 
        project_id=project_uuid,
        page=page_number,
        data_per_page=num_items_per_page,
        sort_order=SortOrder.DESCENDING.value 
    )

    if not shortlist:
        if project_settings.total_gallery_pages != res_payload['total_pages']:
            project_settings.total_gallery_pages = res_payload['total_pages']
            st.rerun()
    else:
        if project_settings.total_shortlist_gallery_pages != res_payload['total_pages']:
            project_settings.total_shortlist_gallery_pages = res_payload['total_pages']
            st.rerun()

    total_image_count = res_payload['count']
    if gallery_image_list and len(gallery_image_list):
        start_index = 0
        end_index = min(start_index + num_items_per_page, total_image_count)
        shot_names = [s.name for s in shot_list]
        shot_names.append('**Create New Shot**')
        shot_names.insert(0, '')
        for i in range(start_index, end_index, num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(gallery_image_list):
                    with cols[j]:                        
                        st.image(gallery_image_list[i + j].location, use_column_width=True)

                        if shortlist:
                            if st.button("Remove from shortlist ➖", key=f"shortlist_{gallery_image_list[i + j].uuid}",use_container_width=True, help="Remove from shortlist"):
                                data_repo.update_file(gallery_image_list[i + j].uuid, tag=InternalFileTag.GALLERY_IMAGE.value)
                                st.success("Removed From Shortlist")
                                time.sleep(0.3)
                                st.rerun()

                        else:

                            if st.button("Add to shortlist ➕", key=f"shortlist_{gallery_image_list[i + j].uuid}",use_container_width=True, help="Add to shortlist"):
                                data_repo.update_file(gallery_image_list[i + j].uuid, tag=InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value)
                                st.success("Added To Shortlist")
                                time.sleep(0.3)
                                st.rerun()
                                                
                        if gallery_image_list[i + j].inference_log:
                            log = data_repo.get_inference_log_from_uuid(gallery_image_list[i + j].inference_log.uuid)
                            if log:
                                input_params = json.loads(log.input_params)
                                prompt = input_params.get('prompt', 'No prompt found')
                                model = json.loads(log.output_details)['model_name'].split('/')[-1]
                                if view == "main":
                                    with st.expander("Prompt Details", expanded=open_detailed_view_for_all):
                                        st.info(f"**Prompt:** {prompt}\n\n**Model:** {model}")
                                                                        
                                shot_name = st.selectbox('Add to shot:', shot_names, key=f"current_shot_sidebar_selector_{gallery_image_list[i + j].uuid}")
                                
                                if shot_name != "":
                                    if shot_name == "**Create New Shot**":
                                        shot_name = st.text_input("New shot name:", max_chars=40, key=f"shot_name_{gallery_image_list[i+j].uuid}")
                                        if st.button("Create new shot", key=f"create_new_{gallery_image_list[i + j].uuid}", use_container_width=True):
                                            new_shot = add_new_shot(project_uuid, name=shot_name)
                                            add_key_frame(gallery_image_list[i + j], False, new_shot.uuid, len(data_repo.get_timing_list_from_shot(new_shot.uuid)), refresh_state=False)
                                            # removing this from the gallery view
                                            data_repo.update_file(gallery_image_list[i + j].uuid, tag="")
                                            st.rerun()
                                        
                                    else:
                                        if st.button(f"Add to shot", key=f"add_{gallery_image_list[i + j].uuid}", help="Promote this variant to the primary image", use_container_width=True):
                                            shot_number = shot_names.index(shot_name) + 1
                                            shot_uuid = shot_list[shot_number - 2].uuid

                                            add_key_frame(gallery_image_list[i + j], False, shot_uuid, len(data_repo.get_timing_list_from_shot(shot_uuid)), refresh_state=False)
                                            # removing this from the gallery view
                                            data_repo.update_file(gallery_image_list[i + j].uuid, tag="")
                                            st.rerun()
                            else:
                                st.warning("No inference data")
                        else:
                            st.warning("No data found")
                                                    
            st.markdown("***")
    else:
        st.warning("No images present")


def create_variate_option(column, key):
    label = key.replace('_', ' ').capitalize()
    variate_option = column.checkbox(f"Vary {label.lower()}", key=f"{key}_checkbox")
    if variate_option:
        with column:
            instructions = st_memory.text_area(f"How would you like to vary the {label.lower()}?", key=f"{key}_textarea", help=f"It'll write a custom {label.lower()} prompt based on your instructions.")
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


def update_max_frame_per_shot_element(project_uuid):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)

    '''
    max_frames = st.number_input(label='Max frames per shot', min_value=1, value=project_settings.max_frames_per_shot)

    if max_frames != project_settings.max_frames_per_shot:
        project_settings.max_frames_per_shot = max_frames
        st.success("Updated")
        time.sleep(0.3)
        st.rerun()
    '''