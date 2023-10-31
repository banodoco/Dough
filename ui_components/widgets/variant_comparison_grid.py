import streamlit as st
from ui_components.constants import CreativeProcessType
from ui_components.methods.common_methods import promote_image_variant, promote_video_variant
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
        variants = shot.interpolated_clip_file_list
    else:
        timing_uuid = ele_uuid
        timing = data_repo.get_timing_from_uuid(timing_uuid)
        variants = timing.alternative_images_list

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
            st.video(variants[current_variant].location, format='mp4', start_time=0) if variants[current_variant] else st.error("No video present")
        else:
            st.image(variants[current_variant].location, use_column_width=True)
        st.success("**Main variant**")

    start = (page - 1) * items_to_show
    end = min(start + items_to_show, len(variants))

    next_col = 1
    for i in range(end - 1, start - 1, -1):
        variant_index = i
        if variant_index != current_variant:
            with cols[next_col]:
                if stage == CreativeProcessType.MOTION.value:
                    st.video(variants[variant_index].location, format='mp4', start_time=0) if variants[variant_index] else st.error("No video present")
                else:
                    st.image(variants[variant_index].location, use_column_width=True) if variants[variant_index] else st.error("No image present")
                
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