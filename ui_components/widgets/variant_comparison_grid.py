import json
import time
import streamlit as st
from shared.constants import InferenceParamType, InternalFileTag
from ui_components.constants import CreativeProcessType
from ui_components.methods.common_methods import promote_image_variant, promote_video_variant
from ui_components.methods.file_methods import create_duplicate_file
from ui_components.methods.video_methods import sync_audio_and_duration
from ui_components.models import InternalFileObject
from ui_components.widgets.add_key_frame_element import add_key_frame
from ui_components.widgets.animation_style_element import update_interpolation_settings
from utils.data_repo.data_repo import DataRepo


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

    st.markdown("***")

    col1, col2 = st.columns([1, 1])
    items_to_show = col1.slider('Variants per page:', min_value=1, max_value=12, value=6)
    num_columns = col2.slider('Number of columns:', min_value=1, max_value=6, value=3)
    
    num_pages = (len(variants) + 1) // items_to_show
    if (len(variants) + 1) % items_to_show != 0:
        num_pages += 1


    page = 1
    if num_pages > 1:
        page = st.radio('Page:', options=list(range(1, num_pages + 1)), horizontal=True)

    if not len(variants):
        st.info("No variants present")
        return

    current_variant = shot.primary_interpolated_video_index if stage == CreativeProcessType.MOTION.value else int(
        timing.primary_variant_index)

    st.markdown("***")
    cols = st.columns(num_columns)
    with cols[0]:
        if stage == CreativeProcessType.MOTION.value:
            st.video(variants[current_variant].location, format='mp4', start_time=0) if (current_variant != -1 and variants[current_variant]) else st.error("No video present")
        else:
            st.image(variants[current_variant].location, use_column_width=True)
        with st.expander("Inference details"):
            variant_inference_detail_element(variants[current_variant], stage, shot_uuid, timing_list)
        
        st.success("**Main variant**")

    start = (page - 1) * items_to_show
    end = min(start + items_to_show-1, len(variants) - 1)

    next_col = 1
    for variant_index in range(end, start - 1, -1):
        if variant_index != current_variant:
            with cols[next_col]:
                if stage == CreativeProcessType.MOTION.value:
                    st.video(variants[variant_index].location, format='mp4', start_time=0) if variants[variant_index] else st.error("No video present")
                else:
                    st.image(variants[variant_index].location, use_column_width=True) if variants[variant_index] else st.error("No image present")                
                with st.expander("Inference details"):                         

                    variant_inference_detail_element(variants[variant_index], stage, shot_uuid, timing_list)
                if st.button(f"Promote Variant #{variant_index + 1}", key=f"Promote Variant #{variant_index + 1} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image", use_container_width=True):
                    if stage == CreativeProcessType.MOTION.value:
                        promote_video_variant(shot.uuid, variants[variant_index].uuid)
                    else:
                        promote_image_variant(timing.uuid, variant_index)                    
                    st.rerun()

            next_col += 1

        if next_col >= num_columns:
            cols = st.columns(num_columns)
            next_col = 0  # Reset column counter


def variant_inference_detail_element(variant, stage, shot_uuid, timing_list=""):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    st.markdown(f"Details:")
    inf_data = fetch_inference_data(variant)
    if 'image_prompt_list' in inf_data:
        del inf_data['image_prompt_list']
        del inf_data['image_list']
        del inf_data['output_format']
    
    st.write(inf_data)

    if stage != CreativeProcessType.MOTION.value:
        h1, h2 = st.columns([1, 1])
        with h1:
            st.markdown(f"Add to shortlist:")
            add_variant_to_shortlist_element(variant, shot.project.uuid)
        with h2:
            add_variant_to_shot_element(variant, shot.project.uuid)

    if stage == CreativeProcessType.MOTION.value:
        if st.button("Load up settings from this variant", key=f"{variant.name}", help="This will enter the settings from this variant into the inputs below - you can also use them on other shots", use_container_width=True):
            new_data = prepare_values(fetch_inference_data(variant), timing_list)
            update_interpolation_settings(values=new_data, timing_list=timing_list)                        
            st.success("Settings loaded - scroll down to run them.")                                                          
            st.rerun()
        if st.button("Sync audio/duration", key=f"{variant.uuid}", help="Updates video length and the attached audio", use_container_width=True):
            data_repo = DataRepo()
            _ = sync_audio_and_duration(variant, shot_uuid)
            _ = data_repo.get_shot_list(shot.project.uuid, invalidate_cache=True)
            st.success("Video synced")
            time.sleep(0.3)
            st.rerun()


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
        'frames_per_keyframe': settings.get('linear_frames_per_keyframe', None),
        'type_of_key_frame_influence': 1 if settings.get('type_of_key_frame_influence') == 'dynamic' else 0,
        'length_of_key_frame_influence': float(settings.get('linear_key_frame_influence_value')) if settings.get('linear_key_frame_influence_value') else None,
        'type_of_cn_strength_distribution': 1 if settings.get('type_of_cn_strength_distribution') == 'dynamic' else 0,
        'linear_cn_strength_value': float(settings.get('linear_cn_strength_value')) if settings.get('linear_cn_strength_value') else None,
        'interpolation_style': interpolation_style_map[settings.get('interpolation_type')] if settings.get('interpolation_type', 'ease-in-out') in interpolation_style_map else None,
        'motion_scale': settings.get('motion_scale', None),            
        'negative_prompt_video': settings.get('negative_prompt', None),
        'ip_adapter_weight_video': settings.get('ip_adapter_model_weight', None),
        'soft_scaled_cn_weights_multiple_video': settings.get('soft_scaled_cn_multiplier', None)
    }

    # Add dynamic values
    dynamic_frame_distribution_values = settings['dynamic_frames_per_keyframe'].split(',') if settings['dynamic_frames_per_keyframe'] else []
    dynamic_key_frame_influence_values = settings['dynamic_key_frame_influence_value'].split(',') if settings['dynamic_key_frame_influence_value'] else []
    dynamic_cn_strength_values = settings['dynamic_cn_strength_values'].split(',') if settings['dynamic_cn_strength_values'] else []

    min_length = min(len(timing_list), len(dynamic_frame_distribution_values), len(dynamic_key_frame_influence_values), len(dynamic_cn_strength_values))

    for idx in range(1, min_length + 1):
        values[f'dynamic_frame_distribution_values_{idx}'] = int(dynamic_frame_distribution_values[idx - 1]) if dynamic_frame_distribution_values[idx - 1] and dynamic_frame_distribution_values[idx - 1].strip() else None
        values[f'dynamic_key_frame_influence_values_{idx}'] = float(dynamic_key_frame_influence_values[idx - 1]) if dynamic_key_frame_influence_values[idx - 1] and dynamic_key_frame_influence_values[idx - 1].strip() else None
        values[f'dynamic_cn_strength_values_{idx}'] = float(dynamic_cn_strength_values[idx - 1]) if dynamic_cn_strength_values[idx - 1] and dynamic_cn_strength_values[idx - 1].strip() else None

    return values


def fetch_inference_data(file: InternalFileObject):
    if not file:
        return
    
    not_found_msg = 'No data available.'    
    inf_data = None
    # NOTE: generated videos also have other params stored inside origin_data > settings
    if file.inference_log and file.inference_log.input_params:
        inf_data = json.loads(file.inference_log.input_params)
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