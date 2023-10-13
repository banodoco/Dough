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

    num_columns = st.slider('Number of columns', min_value=1, max_value=10, value=4)

    for i in range(0, len(variants), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(variants):
                with cols[j]:
                    if stage == CreativeProcessType.MOTION.value:
                        st.video(variants[i + j].location, format='mp4', start_time=0) if variants[i + j] else st.error("No video present")
                    else:
                        st.image(variants[i + j].location, use_column_width=True)

                    if i + j == current_variant:
                        st.success("**Main variant**")
                    else:                                        
                        if st.button(f"Promote Variant #{i + j + 1}", key=f"Promote Variant #{i + j + 1} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image", use_container_width=True):

                            promote_image_variant(timing.uuid, i + j)                                            
                            st.rerun()