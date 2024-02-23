import json
import streamlit as st
from ui_components.constants import GalleryImageViewType
from ui_components.methods.common_methods import get_canny_img, process_inference_output,add_new_shot, save_new_image, save_uploaded_image, execute_image_edit
from ui_components.methods.file_methods import generate_pil_image
from ui_components.methods.ml_methods import query_llama2, inpainting
from ui_components.widgets.add_key_frame_element import add_key_frame
from ui_components.widgets.inpainting_element import inpainting_element
from utils.common_utils import refresh_app
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from shared.constants import GPU_INFERENCE_ENABLED, QUEUE_INFERENCE_QUERIES, AIModelType, InferenceType, InternalFileTag, InternalFileType, SortOrder
from utils import st_memory
import time
from utils.enum import ExtendedEnum
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL
import numpy as np
from utils import st_memory


class InputImageStyling(ExtendedEnum):
    TEXT2IMAGE = "Text to Image"
    IMAGE2IMAGE = "Image to Image"
    CONTROLNET_CANNY = "ControlNet Canny"
    IPADAPTER_FACE = "IP-Adapter Face"
    IPADAPTER_PLUS = "IP-Adapter Plus"
    IPADPTER_FACE_AND_PLUS = "IP-Adapter Face & Plus"
    INPAINTING = "Inpainting"


def explorer_page(project_uuid):
    st.markdown(f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}]")
    st.markdown("***")

    with st.expander("✨ Generate Images", expanded=True):
        generate_images_element(position='explorer', project_uuid=project_uuid, timing_uuid=None, shot_uuid=None)
    st.markdown("***")

    gallery_image_view(project_uuid,False,view=['add_and_remove_from_shortlist','view_inference_details','shot_chooser'])

def generate_images_element(position='explorer', project_uuid=None, timing_uuid=None, shot_uuid=None):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)
    help_input='''This will generate a specific prompt based on your input.\n\n For example, "Sad scene of old Russian man, dreary style" might result in "Boris Karloff, 80 year old man wearing a suit, standing at funeral, dark blue watercolour."'''
    a1, a2 = st.columns([1,1])   

    #with a1 if 'switch_prompt_position' not in st.session_state or st.session_state['switch_prompt_position'] == False else a2:
    with a1:
        prompt = st_memory.text_area("What's your base prompt?", key="explorer_base_prompt", help="This exact text will be included for each generation.")

    # with a2 if 'switch_prompt_position' not in st.session_state or st.session_state['switch_prompt_position'] == False else a1:
    with a2:
        negative_prompt = st_memory.text_area("Negative prompt:", value="",\
                                        key="explorer_neg_prompt", \
                                                help="These are the things you wish to be excluded from the image")


   
    type_of_generation = st_memory.radio("How would you like to generate the image?", options=InputImageStyling.value_list(), key="type_of_generation_key", help="Evolve Image will evolve the image based on the prompt, while Maintain Structure will keep the structure of the image and change the style.",horizontal=True) 

    if type_of_generation != InputImageStyling.TEXT2IMAGE.value:
        if "input_image_1" not in st.session_state:
            st.session_state["input_image_1"] = None
            st.session_state["input_image_2"] = None

        uploaded_image_1 = None
        uploaded_image_2 = None

        # these require two images
        if type_of_generation == InputImageStyling.INPAINTING.value:
            if 'uploaded_image' not in st.session_state:
                st.session_state['uploaded_image'] = ""
            h1, h2 = st.columns([0.8, 3])
            with h1:
                if st.session_state['uploaded_image'] == "":
                    source_of_starting_image = st.radio("Image source:", options=["Upload","From Shot"], key=f"{inpainting_element}_starting_image", help="This will be the base image for the generation.", horizontal=True)
                    if source_of_starting_image == "Upload":
                        uploaded_image = st.file_uploader("Upload a starting image", type=["png", "jpg", "jpeg"], key=st.session_state[f"uploaded_image"], help="This will be the base image for the generation.")
                        if uploaded_image:
                            if st.button("Select as base image", key="select_as_base_image"):
                                st.session_state['uploaded_image'] = uploaded_image
                    else:
                        # taking image from shots
                        shot_list = data_repo.get_shot_list(project_uuid)
                        selection1, selection2 = st.columns([1, 1])            
                        shot_name = st.selectbox("Shot:", options=[shot.name for shot in shot_list], key=f"inpainting_shot_name", help="This will be the base image for the generation.")
                        shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                        frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                        list_of_timings = [i + 1 for i in range(len(frame_list))]
                        timing = st.selectbox("Frame #:", options=list_of_timings, key=f"inpainting_frame_number", help="This will be the base image for the generation.")
                        st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)
                        if timing:
                            if st.button("Select as base image", key="select_as_base_image"):
                                st.session_state['uploaded_image'] = frame_list[timing - 1].primary_image.location

            with h2:
                if st.session_state['uploaded_image']:                    
                    inpainting_element(h1)
                else:
                    st.info("<- Please select an image")

        else:

            def handle_image_input(column, type_of_generation, output_value_name, data_repo=None, project_uuid=None):

                with column:

                    if st.session_state.get(output_value_name) is None:

                        top0, top1, top2, top3 = st.columns([0.4,1, 1,0.4])
                        with top1:
                            st.info(f"{type_of_generation} input:")  # Dynamic title based on type_of_generation
                        with top2:                    
                            source_of_starting_image = st.radio("Image source:", options=["Upload","From Shot"], key=f"{output_value_name}_starting_image", help="This will be the base image for the generation.", horizontal=True)                
                                                    
                        if 'uploaded_image' not in st.session_state:
                            st.session_state['uploaded_image'] = None
                        if f"uploaded_image_{output_value_name}" not in st.session_state:
                            st.session_state[f"uploaded_image_{output_value_name}"] = f"0_{output_value_name}"
                        if source_of_starting_image == "Upload":

                            st.session_state['uploaded_image'] = st.file_uploader("Upload a starting image", type=["png", "jpg", "jpeg"], key=st.session_state[f"uploaded_image_{output_value_name}"], help="This will be the base image for the generation.")
                        else:
                            # taking image from shots
                            shot_list = data_repo.get_shot_list(project_uuid)
                            selection1, selection2 = st.columns([1, 1])
                            with selection1:
                                shot_name = st.selectbox("Shot:", options=[shot.name for shot in shot_list], key=f"{output_value_name}_shot_name", help="This will be the base image for the generation.")
                                shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                                frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                                list_of_timings = [i + 1 for i in range(len(frame_list))]
                                timing = st.selectbox("Frame #:", options=list_of_timings, key=f"{output_value_name}_frame_number", help="This will be the base image for the generation.")
                                if timing:
                                    st.session_state['uploaded_image'] = frame_list[timing - 1].primary_image.location
                            with selection2:
                                if frame_list and len(frame_list) and timing:
                                    st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)

                        # Trigger image processing
                        if st.button("Upload Image", key=f"{output_value_name}_upload_button", use_container_width=True):
                            st.session_state[output_value_name] = st.session_state['uploaded_image']
                            st.session_state[f"uploaded_image_{output_value_name}"] += 1                        
                            st.rerun()

                        return None
                    else:
                        # Display current image
                        st.info(f"{type_of_generation} image:")
                        half1, half2 = st.columns([1, 1])
                        with half1:                        
                            st.image(st.session_state[output_value_name], use_column_width=True)
                        # Slider for image adjustment based on type_of_generation
                        with half2:
                            strength_of_image = st.slider("Image strength:", min_value=0, max_value=100, value=50, step=1, key=f"{output_value_name}_strength", help="This will be the strength of the image for the generation.")
                                                
                            if st.button("Clear image", key=f"{output_value_name}_clear_button"):
                                st.session_state[output_value_name] = None
                                st.rerun()

                        return strength_of_image
            
            sub0, sub1, sub2, sub3 = st.columns([0.3,1, 1,0.3])
            if type_of_generation != InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                strength_of_image = handle_image_input(sub1, type_of_generation, "input_image_1", data_repo, project_uuid) 
            
            else:
                strength_of_image_1 = handle_image_input(sub1, "IP-Adapter Face", "input_image_1", data_repo, project_uuid)
                strength_of_image_2 = handle_image_input(sub2, "IP-Adapter Plus", "input_image_2", data_repo, project_uuid)
                strength_of_image = (strength_of_image_1, strength_of_image_2)
                # if both images are populated, create a "switch images" button 
                with sub2:
                    corner1, corner2 = st.columns([1, 1])
                    with corner2:
                        if st.session_state["input_image_1"] and st.session_state["input_image_2"]:
                            if st.button("Switch images 🔄", key="switch_images", use_container_width=True):
                                st.session_state["input_image_1"], st.session_state["input_image_2"] = st.session_state["input_image_2"], st.session_state["input_image_1"]
                                st.rerun()
    
    
    if position == 'explorer':
        _, d2,d3, _ = st.columns([0.25, 1,1, 0.25])
    else:
        d2, d3 = st.columns([1,1])
    with d2:  
        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=4, max_value=100, value=4, step=4, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    with d3:
        st.write(" ")
        # ------------------- Generating output -------------------------------------
        if st.session_state.get(position + '_generate_inference'):
            ml_client = get_ml_client()
            
            for _ in range(number_to_generate):
                log = None
                generation_method = InputImageStyling.value_list()[st.session_state['type_of_generation_key']]
                if generation_method == InputImageStyling.TEXT2IMAGE.value:
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        guidance_scale=8,
                        seed=-1,                            
                        num_inference_steps=25,            
                        strength=0.5,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={"shot_uuid": shot_uuid}
                    )
                    
                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                
                elif generation_method == InputImageStyling.IMAGE2IMAGE.value:
                    input_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={"shot_uuid": shot_uuid}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl_img2img, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                elif generation_method == InputImageStyling.CONTROLNET_CANNY.value:
                    edge_pil_img = get_canny_img(st.session_state["input_image_1"], low_threshold=50, high_threshold=150)    # redundant incase of local inference
                    input_img = edge_pil_img if not GPU_INFERENCE_ENABLED else st.session_state["input_image_1"]
                    input_image_file = save_new_image(input_img, project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=8,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image/100,
                        adapter_type=None,
                        prompt=prompt,
                        low_threshold=0.2,
                        high_threshold=0.7,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={'condition_scale': 1, "shot_uuid": shot_uuid}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl_controlnet, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                
                elif generation_method == InputImageStyling.IPADAPTER_FACE.value:
                    # validation
                    if not (st.session_state["input_image_1"]):
                        st.error('Please upload an image')
                        return

                    input_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image/100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={"shot_uuid": shot_uuid}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.ipadapter_face, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                
                elif generation_method == InputImageStyling.IPADAPTER_PLUS.value:
                    input_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image/100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={'condition_scale': 1, "shot_uuid": shot_uuid}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.ipadapter_plus, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                elif generation_method == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                    # validation
                    if not (st.session_state["input_image_2"] and st.session_state["input_image_1"]):
                        st.error('Please upload both images')
                        return

                    plus_image_file = save_new_image(st.session_state["input_image_1"], project_uuid)
                    face_image_file = save_new_image(st.session_state["input_image_2"], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=plus_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=(strength_of_image_1/100, strength_of_image_2/100), # (face, plus)
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={'file_image_2_uuid': face_image_file.uuid, "shot_uuid": shot_uuid}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.ipadapter_face_plus, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                elif generation_method == InputImageStyling.INPAINTING.value:
                    edited_image, log = inpainting(st.session_state['editing_image'], prompt, negative_prompt, project_settings.width, project_settings.height, shot_uuid,project_uuid)

                    
                    inference_data = {
                        "inference_type": InferenceType.FRAME_INPAINTING.value,
                        "output": edited_image,
                        "log_uuid": log.uuid,
                        "timing_uuid": st.session_state['current_frame_uuid'],
                        "promote_generation": False,
                        "stage": ""
                    }

                    process_inference_output(**inference_data)

                if log:
                    inference_data = {
                        "inference_type": InferenceType.GALLERY_IMAGE_GENERATION.value if position == 'explorer' else InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value,
                        "output": output,
                        "log_uuid": log.uuid,
                        "project_uuid": project_uuid,
                        "timing_uuid": timing_uuid,
                        "promote_new_generation": False,
                        "shot_uuid": shot_uuid if shot_uuid else "explorer"
                    }
                    
                    process_inference_output(**inference_data)

            st.info("Check the Generation Log to the left for the status.")
            time.sleep(0.5)
            toggle_generate_inference(position)
            st.rerun()

        # ----------- generate btn --------------
        if prompt == "":
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please enter a prompt to generate images")
        elif type_of_generation == InputImageStyling.IMAGE2IMAGE.value and st.session_state["input_image_1"] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.CONTROLNET_CANNY.value and st.session_state["input_image_1"] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.IPADAPTER_FACE.value and st.session_state["input_image_1"] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.IPADAPTER_PLUS.value and st.session_state["input_image_1"] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value and (st.session_state["input_image_1"] is None or st.session_state["input_image_2"] is None):
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload both images")        
        else:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", on_click=lambda: toggle_generate_inference(position))
            
def toggle_generate_inference(position):
    if position + '_generate_inference' not in st.session_state:
        st.session_state[position + '_generate_inference'] = True
    else:
        st.session_state[position + '_generate_inference'] = not st.session_state[position + '_generate_inference']

def gallery_image_view(project_uuid, shortlist=False, view=["main"], shot=None, sidebar=False):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)
    shot_list = data_repo.get_shot_list(project_uuid)
    shot_name_uuid_map = {s.name : s.uuid for s in shot_list}
    shot_uuid_list = [GalleryImageViewType.EXPLORER_ONLY.value]  
    
    if shortlist is False:
        st.markdown("***")
        st.markdown("### 🖼️ Gallery ----------")
        st.write("##### -\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-")

    h1,h2,h3,h4 = st.columns([3, 1, 1, 1])
       # by default only showing explorer views
    k1,k2 = st.columns([5,1])

    if sidebar != True:
        if shortlist is False:
            f1, f2 = st.columns([1, 1])
            with h2:
                num_columns = st_memory.slider('Number of columns:', min_value=3, max_value=7, value=4,key="num_columns_explorer")
            with h3:
                num_items_per_page = st_memory.slider('Items per page:', min_value=8, max_value=64, value=16, key="num_items_per_page_explorer")
        else:
            num_items_per_page = 4
            num_columns = 2

        if 'shot_chooser' in view:
            with h1:
                shot_chooser_1,shot_chooser_2 = st.columns([1, 1])
                with shot_chooser_1:        
                    options = ["Timeline", "Specific shots", "All"]                         
                    if shot is None:            
                        default_value = 0
                    else:
                        default_value = 1
                    show_images_associated_with_shots = st.selectbox("Images associated with:", options=options, index=default_value, key=f"show_images_associated_with_shots_explorer_{shortlist}")
                with shot_chooser_2:
                    if show_images_associated_with_shots == "Timeline":
                        shot_uuid_list = [GalleryImageViewType.EXPLORER_ONLY.value]
                    
                    elif show_images_associated_with_shots == "Specific shots":
                        specific_shots = [shot.name for shot in shot_list]  
                        if shot is None:
                            default_shot = specific_shots[0]
                        else:
                            default_shot = shot.name
                        shot_name_list = st.multiselect("Shots to show:", options=specific_shots, default=default_shot, key="specific_shots_explorer")
                        shot_uuid_list = [str(shot_name_uuid_map[s]) for s in shot_name_list] if shot_name_list else ['xyz']
                        
                    else:
                        shot_uuid_list = []
        else:
            shot_uuid_list = []

        if shortlist is False:
            with h1:
                page_number = st.radio("Select page:", options=range(1, project_settings.total_gallery_pages + 1), horizontal=True, key="main_gallery")
            with h4:
                st.write("")                
                if 'view_inference_details' in view:
                    open_detailed_view_for_all = st.toggle("Open all details:", key='main_gallery_toggle')
                
        else:
            with h1:
                project_setting = data_repo.get_project_setting(project_uuid)
                page_number = k1.radio("Select page", options=range(1, project_setting.total_shortlist_gallery_pages), horizontal=True, key="shortlist_gallery")
                open_detailed_view_for_all = False     
                
    else:
        project_setting = data_repo.get_project_setting(project_uuid)
        page_number = k1.radio("Select page", options=range(1, project_setting.total_shortlist_gallery_pages), horizontal=True, key="shortlist_gallery")
        open_detailed_view_for_all = False        
        num_items_per_page = 8
        num_columns = 2
    
    gallery_image_filter_data = {
        "file_type" : InternalFileType.IMAGE.value, 
        "tag" : InternalFileTag.GALLERY_IMAGE.value if not shortlist else InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value, 
        "project_id" : project_uuid,
        "page" : page_number or 1,
        "data_per_page" : num_items_per_page,
        "sort_order" : SortOrder.DESCENDING.value 
    }
    
    if shot_uuid_list and not sidebar:
        gallery_image_filter_data["shot_uuid_list"] = shot_uuid_list
    
    gallery_image_list, res_payload = data_repo.get_all_file_list(
        **gallery_image_filter_data
    )

    if not shortlist:
        if project_settings.total_gallery_pages != res_payload['total_pages']:
            project_settings.total_gallery_pages = res_payload['total_pages']
            st.rerun()
    else:
        if project_settings.total_shortlist_gallery_pages != res_payload['total_pages']:
            project_settings.total_shortlist_gallery_pages = res_payload['total_pages']
            st.rerun()

    if shortlist is False:
        _, fetch2, fetch3, _ = st.columns([0.25, 1, 1, 0.25])
        # st.markdown("***")
        explorer_stats = data_repo.get_explorer_pending_stats(project_uuid=project_uuid)
        
        if explorer_stats['temp_image_count'] + explorer_stats['pending_image_count']:        
            st.markdown("***")
            
            with fetch2:
                total_number_pending = explorer_stats['temp_image_count'] + explorer_stats['pending_image_count']
                if total_number_pending:
                    
                    if explorer_stats['temp_image_count'] == 0 and explorer_stats['pending_image_count'] > 0:
                        st.info(f"###### {explorer_stats['pending_image_count']} images pending generation")
                        button_text = "Check for new images"
                    elif explorer_stats['temp_image_count'] > 0 and explorer_stats['pending_image_count'] == 0:
                        st.info(f"###### {explorer_stats['temp_image_count']} new images generated")                        
                        button_text = "Pull new images"
                    else:
                        st.info(f"###### {explorer_stats['pending_image_count']} images pending generation and {explorer_stats['temp_image_count']} ready to be fetched")
                        button_text = "Check for/pull new images"
                    
                # st.info(f"###### {total_number_pending} images pending generation")
                # st.info(f"###### {explorer_stats['temp_image_count']} new images generated")     
                # st.info(f"###### {explorer_stats['pending_image_count']} images pending generation")     
            
            with fetch3:
                    if st.button(f"{button_text}", key=f"check_for_new_images_", use_container_width=True):
                        data_repo.update_temp_gallery_images(project_uuid)
                        if explorer_stats['temp_image_count']:                           
                            st.success("New images fetched")
                            time.sleep(0.3)
                        st.rerun()

    total_image_count = res_payload['count']
    if gallery_image_list and len(gallery_image_list):
        start_index = 0
        end_index = min(start_index + num_items_per_page, total_image_count)
        shot_names = [s.name for s in shot_list]
        shot_names.append('**Create New Shot**')        
        for i in range(start_index, end_index, num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(gallery_image_list):
                    with cols[j]:                        
                        st.image(gallery_image_list[i + j].location, use_column_width=True)
                                                # ---------- add to shot btn ---------------
                        if "last_shot_number" not in st.session_state:
                            st.session_state["last_shot_number"] = 0
                        if 'add_to_this_shot' in view or 'add_to_any_shot' in view:
                            if 'add_to_this_shot' in view:
                                shot_name = shot.name
                            else:
                                shot_name = st.selectbox('Add to shot:', shot_names, key=f"current_shot_sidebar_selector_{gallery_image_list[i + j].uuid}",index=st.session_state["last_shot_number"])
                            
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
                                    if st.button(f"Add to shot", key=f"add_{gallery_image_list[i + j].uuid}", use_container_width=True):
                                        shot_number = shot_names.index(shot_name)
                                        st.session_state["last_shot_number"] = shot_number 
                                        shot_uuid = shot_list[shot_number].uuid

                                        add_key_frame(gallery_image_list[i + j], False, shot_uuid, len(data_repo.get_timing_list_from_shot(shot_uuid)), refresh_state=False)
                                        # removing this from the gallery view
                                        data_repo.update_file(gallery_image_list[i + j].uuid, tag="")
                                        refresh_app(maintain_state=True)                  
                        # else:
                        #     st.error("The image is truncated and cannot be displayed.")
                        if 'add_and_remove_from_shortlist' in view:
                            if shortlist:
                                if st.button("Remove from shortlist ➖", key=f"shortlist_{gallery_image_list[i + j].uuid}",use_container_width=True):
                                    data_repo.update_file(gallery_image_list[i + j].uuid, tag=InternalFileTag.GALLERY_IMAGE.value)
                                    st.success("Removed From Shortlist")
                                    time.sleep(0.3)
                                    st.rerun()
                            else:
                                if st.button("Add to shortlist ➕", key=f"shortlist_{gallery_image_list[i + j].uuid}",use_container_width=True, help="The shortlist appears in a box on the left."):
                                    data_repo.update_file(gallery_image_list[i + j].uuid, tag=InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value)
                                    st.success("Added To Shortlist")
                                    time.sleep(0.3)
                                    st.rerun()

                        # -------- inference details --------------          
                        if gallery_image_list[i + j].inference_log:
                            log = gallery_image_list[i + j].inference_log # data_repo.get_inference_log_from_uuid(gallery_image_list[i + j].inference_log.uuid)
                            if log:
                                input_params = json.loads(log.input_params)
                                prompt = input_params.get('prompt', 'No prompt found')
                                model = json.loads(log.output_details)['model_name'].split('/')[-1]
                                if 'view_inference_details' in view:
                                    with st.expander("Prompt Details", expanded=open_detailed_view_for_all):
                                        st.info(f"**Prompt:** {prompt}\n\n**Model:** {model}")
                                
                            else:
                                st.warning("No inference data")
                        else:
                            st.warning("No data found")

                                  
            st.markdown("***")
    else:
        
        st.info("No images present in this view.")