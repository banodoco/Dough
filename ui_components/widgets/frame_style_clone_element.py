import streamlit as st
from shared.constants import AIModelCategory
from ui_components.constants import WorkflowStageType
from ui_components.methods.common_methods import clone_styling_settings
from ui_components.models import InternalAIModelObject
from ui_components.widgets.image_carousal import display_image
from utils.common_utils import reset_styling_settings

from utils.data_repo.data_repo import DataRepo

def style_cloning_element(timing_details):
    open_copier = st.checkbox("Copy styling settings from another frame")
    if open_copier is True:
        copy1, copy2 = st.columns([1, 1])
        with copy1:
            frame_index = st.number_input("Which frame would you like to copy styling settings from?", min_value=1, max_value=len(
                timing_details), value=st.session_state['current_frame_index'], step=1)
            if st.button("Copy styling settings from this frame"):
                clone_styling_settings(frame_index - 1, st.session_state['current_frame_uuid'])
                reset_styling_settings(st.session_state['current_frame_uuid'])
                st.rerun()

        with copy2:
            display_image(timing_details[frame_index  - 1].uuid, stage=WorkflowStageType.STYLED.value, clickable=False)
            
            if timing_details[frame_index - 1].primary_image.inference_params:
                st.text("Prompt: ")
                st.caption(timing_details[frame_index - 1].primary_image.inference_params.prompt)
                st.text("Negative Prompt: ")
                st.caption(timing_details[frame_index - 1].primary_image.inference_params.negative_prompt)
                
                if timing_details[frame_index - 1].primary_image.inference_params.model_uuid:
                    data_repo = DataRepo()
                    model: InternalAIModelObject = data_repo.get_ai_model_from_uuid(timing_details[frame_index - 1].primary_image.inference_params.model_uuid)
                    
                    st.text("Model:")
                    st.caption(model.name)

                    if model.category.lower() == AIModelCategory.CONTROLNET.value:
                        st.text("Adapter Type:")
                        st.caption(timing_details[frame_index - 1].primary_image.inference_params.adapter_type)