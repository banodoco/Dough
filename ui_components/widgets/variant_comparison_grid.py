import streamlit as st
from ui_components.constants import CreativeProcessType
from ui_components.methods.common_methods import promote_image_variant
from utils.data_repo.data_repo import DataRepo


def variant_comparison_grid(timing_uuid, stage=CreativeProcessType.MOTION.value):
    data_repo = DataRepo()

    timing = data_repo.get_timing_from_uuid(timing_uuid)
    variants = timing.alternative_images_list
    
    current_variant = timing.primary_interpolated_video_index if stage == CreativeProcessType.MOTION.value else int(
        timing.primary_variant_index)

    st.markdown("***")

    col1, col2 = st.columns([1, 1])
    items_to_show = col1.slider('Variants per page:', min_value=1, max_value=12, value=6)
    num_columns = col2.slider('Number of columns:', min_value=1, max_value=6, value=3)
    
        # Display the main variant first    
    num_pages = (len(variants) + 1) // items_to_show
    if (len(variants) + 1) % items_to_show != 0:
        num_pages += 1


    # Create a number input for page selection if there's more than one page
    page = 1
    if num_pages > 1:
        page = st.radio('Page:', options=list(range(1, num_pages + 1)), horizontal=True)

    st.markdown("***")

    # Display the main variant first
    cols = st.columns(num_columns)
    with cols[0]:
        if stage == CreativeProcessType.MOTION.value:
            st.video(variants[current_variant].location, format='mp4', start_time=0) if variants[current_variant] else st.error("No video present")
        else:
            st.image(variants[current_variant].location, use_column_width=True)
        st.success("**Main variant**")

    total_variants = len(variants)
    start = total_variants - (page * items_to_show)
    end = start + items_to_show
    if start < 0:
        start = 0
    # Start from the last variant
    next_col = 1
    for i in range(end - 1, start - 1, -1):
        variant_index = i
        if variant_index != current_variant:  # Skip the main variant
            with cols[next_col]:  # Use next_col to place the variant
                if stage == CreativeProcessType.MOTION.value:
                    st.video(variants[variant_index].location, format='mp4', start_time=0) if variants[variant_index] else st.error("No video present")
                else:
                    st.image(variants[variant_index].location, use_column_width=True)
                
                if st.button(f"Promote Variant #{variant_index + 1}", key=f"Promote Variant #{variant_index + 1} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image", use_container_width=True):
                    promote_image_variant(timing.uuid, variant_index)                                            
                    st.rerun()

            next_col += 1  # Move to the next column

        # Create new row after filling the current one
        if next_col >= num_columns:
            cols = st.columns(num_columns)
            next_col = 0  # Reset column counter