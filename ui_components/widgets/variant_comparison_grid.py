import json
import time
import ast
import streamlit as st
import re
from shared.constants import AIModelCategory, InferenceParamType, InternalFileTag
from ui_components.constants import CreativeProcessType
from ui_components.methods.animation_style_methods import get_generation_settings_from_log, load_shot_settings
from ui_components.methods.common_methods import promote_image_variant, promote_video_variant
from ui_components.methods.file_methods import create_duplicate_file
from ui_components.methods.video_methods import sync_audio_and_duration
from ui_components.widgets.shot_view import create_video_download_button
from ui_components.models import InternalAIModelObject, InternalFileObject
from ui_components.widgets.add_key_frame_element import add_key_frame
from ui_components.widgets.animation_style_element import update_interpolation_settings
from utils import st_memory
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL



def variant_comparison_grid(ele_uuid, stage=CreativeProcessType.MOTION.value):
    '''
    UI element which compares different variant of images/videos. For images ele_uuid has to be timing_uuid
    and for videos it has to be shot_uuid.
    '''
    data_repo = DataRepo()

    timing_uuid, shot_uuid = None, None
    if stage == CreativeProcessType.MOTION.value:
        shot_uuid = ele_uuid
        shot = data_repo.get_shot_from_uuid(shot_uuid)
        variants = shot.interpolated_clip_list
        timing_list = data_repo.get_timing_list_from_shot(shot.uuid)
    else:
        timing_uuid = ele_uuid        
        timing = data_repo.get_timing_from_uuid(timing_uuid)
        variants = timing.alternative_images_list
        shot_uuid = timing.shot.uuid
        timing_list =""

    col1, col2, col3 = st.columns([1, 1,0.5])
    if stage == CreativeProcessType.MOTION.value:
        items_to_show = 2
        num_columns = 3
        with col1:
            st.markdown(f"### 🎞️ '{shot.name}' options  _________")
            st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")
    else:
        items_to_show = 5
        num_columns = 3

    # Updated logic for pagination
    num_pages = (len(variants) - 1) // items_to_show + ((len(variants) - 1) % items_to_show > 0)
    page = 1

    if num_pages > 1:
        page = col3.radio('Page:', options=list(range(1, num_pages + 1)), horizontal=True)

    if not len(variants):
        st.info("No options created yet.")
        st.markdown("***")
    else:
        current_variant = shot.primary_interpolated_video_index if stage == CreativeProcessType.MOTION.value else int(timing.primary_variant_index)

        st.markdown("***")
        cols = st.columns(num_columns)
        with cols[0]:
            h1, h2 = st.columns([1, 1])
            with h1:
                st.info(f"###### Variant #{current_variant + 1}")
            with h2:
                st.success("**Main variant**")
            # Display the main variant
            if stage == CreativeProcessType.MOTION.value:
                st.video(variants[current_variant].location, format='mp4', start_time=0) if (current_variant != -1 and variants[current_variant]) else st.error("No video present")
                create_video_download_button(variants[current_variant].location, tag="var_compare")
                variant_inference_detail_element(variants[current_variant], stage, shot_uuid, timing_list, tag="var_compare")                        

            else:
                st.image(variants[current_variant].location, use_column_width=True)
                image_variant_details(variants[current_variant])

        # Determine the start and end indices for additional variants on the current page
        additional_variants = [idx for idx in range(len(variants) - 1, -1, -1) if idx != current_variant]
        page_start = (page - 1) * items_to_show
        page_end = page_start + items_to_show
        page_indices = additional_variants[page_start:page_end]

        next_col = 1
        for i, variant_index in enumerate(page_indices):

            with cols[next_col]:
                h1, h2 = st.columns([1, 1])
                with h1:
                    st.info(f"###### Variant #{variant_index + 1}")
                with h2:
                    if st.button(f"Promote Variant #{variant_index + 1}", key=f"Promote Variant #{variant_index + 1} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image", use_container_width=True):
                        if stage == CreativeProcessType.MOTION.value:
                            promote_video_variant(shot.uuid, variants[variant_index].uuid)
                        else:
                            promote_image_variant(timing.uuid, variant_index)                    
                        st.rerun()

                if stage == CreativeProcessType.MOTION.value:                    
                    st.video(variants[variant_index].location, format='mp4', start_time=0) if variants[variant_index] else st.error("No video present")
                    create_video_download_button(variants[variant_index].location, tag="var_details")
                    variant_inference_detail_element(variants[variant_index], stage, shot_uuid, timing_list, tag="var_details")

                else:
                    if variants[variant_index]:
                        st.image(variants[variant_index].location, use_column_width=True)
                        image_variant_details(variants[variant_index])
                    else: 
                        st.error("No image present")

            next_col += 1

            # if there's only one item, show a line break
            if len(page_indices) == 1:
                st.markdown("***")        
            if next_col >= num_columns or i == len(page_indices) - 1 or len(page_indices) == i:
                next_col = 0  # Reset the column counter
                st.markdown("***")  # Add markdown line
                cols = st.columns(num_columns)  # Prepare for the next row            
                # Add markdown line if this is not the last variant in page_indices
                
def image_variant_details(variant: InternalFileObject):
    with st.expander("Settings", expanded=False):
        if variant.inference_params and 'query_dict' in variant.inference_params:
            query_dict = json.loads(variant.inference_params['query_dict'])
            st.markdown(f"Prompt:  {query_dict['prompt']}", unsafe_allow_html=True)
            st.markdown(f"Negative Prompt: {query_dict['negative_prompt']}", unsafe_allow_html=True)
            st.markdown(f"Dimension: {query_dict['width']}x{query_dict['height']}", unsafe_allow_html=True)
            st.markdown(f"Guidance scale: {query_dict['guidance_scale']}", unsafe_allow_html=True)
            model_name = variant.inference_log.model_name
            st.markdown(f"Model name: {model_name}", unsafe_allow_html=True)
            if model_name in []:
                st.markdown(f"Low threshold: {query_dict['low_threshold']}", unsafe_allow_html=True)
                st.markdown(f"High threshold: {query_dict['high_threshold']}", unsafe_allow_html=True)
            if model_name in [ML_MODEL.sdxl_img2img.display_name(), ML_MODEL.sdxl_controlnet.display_name(),
                                ML_MODEL.ipadapter_face.display_name(), ML_MODEL.ipadapter_plus.display_name()]:
                s = query_dict['strength']
                st.markdown(f"Strength: {s if s > 1 and s <= 100 else int(s * 100)}", unsafe_allow_html=True)
            if model_name in [ML_MODEL.ipadapter_face_plus.display_name()]:
                s = query_dict['strength']
                st.markdown(f"Face Img Strength: {s[0] if s[0] > 1 and s[0] <= 100 else int(s[0] * 100)}", unsafe_allow_html=True)
                st.markdown(f"Plus Img Strength: {s[1] if s[1] > 1 and s[1] <= 100 else int(s[1] * 100)}", unsafe_allow_html=True)
                
def variant_inference_detail_element(variant: InternalFileObject, stage, shot_uuid, timing_list="", tag="temp"):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    if stage == CreativeProcessType.MOTION.value:

        # with st.expander("Settings", expanded=True):
        btn1, btn2 = st.columns([1, 1])
        with btn1:
            open_data=False
            if st.button("See settings", key=f"view_{tag}_{variant.name}", help="This will reveal the settings for this variant below.", use_container_width=True):
                open_data=True

        with btn2:
            if st.button("Load settings", key=f"boot_{tag}_{variant.name}", help="This will load all the settings for this run below. In doing so, it'll remove the current settings and images - though they'll be available for all previous runs.", use_container_width=True):
                load_shot_settings(shot_uuid, variant.inference_log.uuid)
                st.success("Settings Loaded")
                time.sleep(0.3)
                st.rerun()
        
        if open_data:
            with st.expander("Settings", expanded=False):
                shot_meta_data = get_generation_settings_from_log(variant.inference_log.uuid)
                if shot_meta_data and shot_meta_data.get("main_setting_data", None):
                    st.markdown("##### Main settings ---")
                    for k, v in shot_meta_data.get("main_setting_data", {}).items():
                        # Bold the title
                        title = f"**{k.split(str(shot.uuid))[0][:-1]}:**"
                        
                        # Check if the key starts with 'lora_data'
                        if k.startswith('lora_data'):
                            if isinstance(v, list) and len(v) > 0:  # Check if v is a list and has more than 0 items
                                # Handle lora_data differently to format each item in the list
                                lora_items = [f"- {item.get('filename', 'No filename')} - {item.get('lora_strength', 'No strength')} strength" for item in v]
                                lora_data_formatted = "\n".join(lora_items)
                                st.markdown(f"{title} \n{lora_data_formatted}", unsafe_allow_html=True)
                            # If there are no items in the list, do not display anything for lora_data
                        else:
                            # For other keys, display as before but with the title in bold and using a colon
                            if v:  # Check if v is not empty or None
                                st.markdown(f"{title} {v}", unsafe_allow_html=True)
                            else:
                                # Optionally handle empty or None values differently here
                                pass

                st.markdown("##### Frame settings ---")
                st.write("To see the settings for each frame, click on the 'Boot up settings' button above and they'll load below.")
                st.button("Close settings", key=f"close_{tag}_{variant.name}", help="Close this section", use_container_width=True)
            


    if stage != CreativeProcessType.MOTION.value:
        h1, h2 = st.columns([1, 1])
        with h1:
            st.markdown(f"Add to shortlist:")
            add_variant_to_shortlist_element(variant, shot.project.uuid)
        with h2:
            add_variant_to_shot_element(variant, shot.project.uuid)

def prepare_values(inf_data, timing_list):
    settings = inf_data     # Map interpolation_type to indices
    interpolation_style_map = {
        'ease-in-out': 0,
        'ease-in': 1,
        'ease-out': 2,
        'linear': 3
    }

    values = {
        'type_of_frame_distribution': 1 if settings.get('type_of_frame_distribution') == 'dynamic' else 0,
        'linear_frame_distribution_value': settings.get('linear_frame_distribution_value', None),
        'type_of_key_frame_influence': 1 if settings.get('type_of_key_frame_influence') == 'dynamic' else 0,
        'length_of_key_frame_influence': float(settings.get('linear_key_frame_influence_value')) if settings.get('linear_key_frame_influence_value') else None,
        'type_of_cn_strength_distribution': 1 if settings.get('type_of_cn_strength_distribution') == 'dynamic' else 0,
        'linear_cn_strength_value': tuple(map(float, ast.literal_eval(settings.get('linear_cn_strength_value')))) if settings.get('linear_cn_strength_value') else None,
        'interpolation_style': interpolation_style_map[settings.get('interpolation_type')] if settings.get('interpolation_type', 'ease-in-out') in interpolation_style_map else None,
        'motion_scale': settings.get('motion_scale', None),            
        'negative_prompt_video': settings.get('negative_prompt', None),
        'relative_ipadapter_strength': settings.get('relative_ipadapter_strength', None),
        'relative_ipadapter_influence': settings.get('relative_ipadapter_influence', None),        
        'soft_scaled_cn_weights_multiple_video': settings.get('soft_scaled_cn_weights_multiplier', None)
    }

    # Add dynamic values
    dynamic_frame_distribution_values = settings['dynamic_frame_distribution_values'].split(',') if settings['dynamic_frame_distribution_values'] else []
    dynamic_key_frame_influence_values = settings['dynamic_key_frame_influence_values'].split(',') if settings['dynamic_key_frame_influence_values'] else []
    dynamic_cn_strength_values = settings['dynamic_cn_strength_values'].split(',') if settings['dynamic_cn_strength_values'] else []

    min_length = len(timing_list) if timing_list else 0

    for idx in range(min_length):

        # Process dynamic_frame_distribution_values
        if dynamic_frame_distribution_values:            
            values[f'dynamic_frame_distribution_values_{idx}'] = (
                int(dynamic_frame_distribution_values[idx]) 
                if dynamic_frame_distribution_values[idx] and dynamic_frame_distribution_values[idx].strip() 
                else None
            )        
        # Process dynamic_key_frame_influence_values
        if dynamic_key_frame_influence_values:            
            values[f'dynamic_key_frame_influence_values_{idx}'] = (
                float(dynamic_key_frame_influence_values[idx]) 
                if dynamic_key_frame_influence_values[idx] and dynamic_key_frame_influence_values[idx].strip() 
                else None
            )
        
        # Process dynamic_cn_strength_values
        if dynamic_cn_strength_values and idx * 2 <= len(dynamic_cn_strength_values):
            # Since idx starts from 1, we need to adjust the index for zero-based indexing
            adjusted_idx = idx * 2
            # Extract the two elements that form a tuple
            first_value = dynamic_cn_strength_values[adjusted_idx].strip('(')
            second_value = dynamic_cn_strength_values[adjusted_idx + 1].strip(')')
            # Convert both strings to floats and create a tuple
            value_tuple = (float(first_value), float(second_value))
            # Store the tuple in the dictionary with a key indicating its order
            values[f'dynamic_cn_strength_values_{idx}'] = value_tuple

    return values

def fetch_inference_data(file: InternalFileObject):
    if not file:
        return
    
    not_found_msg = 'No data available.'    
    inf_data = None
    # NOTE: generated videos also have other params stored inside origin_data > settings
    if file.inference_log and file.inference_log.input_params:
        inf_data = json.loads(file.inference_log.input_params)
        if 'origin_data' in inf_data and inf_data['origin_data']['inference_type'] == 'frame_interpolation':
            inf_data = inf_data['origin_data']['settings']
        else:
            for data_type in InferenceParamType.value_list():
                if data_type in inf_data:
                    del inf_data[data_type]
    
    inf_data = inf_data or not_found_msg

    return inf_data

def add_variant_to_shortlist_element(file: InternalFileObject, project_uuid):
    data_repo = DataRepo()
    
    if st.button("Add to shortlist ➕", key=f"shortlist_{file.uuid}",use_container_width=True, help="Add to shortlist"):
        duplicate_file = create_duplicate_file(file, project_uuid)
        data_repo.update_file(duplicate_file.uuid, tag=InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value)
        st.success("Added To Shortlist")
        time.sleep(0.3)
        st.rerun()

def add_variant_to_shot_element(file: InternalFileObject, project_uuid):
    data_repo = DataRepo()

    shot_list = data_repo.get_shot_list(project_uuid)
    shot_names = [s.name for s in shot_list]
    
    shot_name = st.selectbox('Add to shot:', shot_names, key=f"current_shot_variant_{file.uuid}")
    if shot_name:
        if st.button(f"Add to shot", key=f"add_{file.uuid}", help="Promote this variant to the primary image", use_container_width=True):
            shot_number = shot_names.index(shot_name)
            shot_uuid = shot_list[shot_number].uuid

            duplicate_file = create_duplicate_file(file, project_uuid)
            add_key_frame(duplicate_file, False, shot_uuid, len(data_repo.get_timing_list_from_shot(shot_uuid)), refresh_state=False, update_cur_frame_idx=False)
            st.rerun()
