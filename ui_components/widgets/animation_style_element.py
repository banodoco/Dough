import time
import streamlit as st
from typing import List
from shared.constants import AnimationStyleType, AnimationToolType
from ui_components.constants import DefaultProjectSettingParams
from ui_components.methods.video_methods import create_single_interpolated_clip
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.motion_module import AnimateDiffCheckpoint

def animation_style_element(shot_uuid):
    motion_modules = AnimateDiffCheckpoint.get_name_list()
    variant_count = 1
    current_animation_style = AnimationStyleType.INTERPOLATION.value    # setting a default value

    if current_animation_style == AnimationStyleType.INTERPOLATION.value:
        animation_tool = st.radio("Animation Tool:", options=AnimationToolType.value_list(), key="animation_tool", horizontal=True)
        video_resolution = None

        settings = {
            "animation_tool": animation_tool
        }
        
        if animation_tool == AnimationToolType.ANIMATEDIFF.value:
            c1, c2 = st.columns([1,1])
            with c1:
                motion_module = st.selectbox("Which motion module would you like to use?", options=motion_modules, key="motion_module")
            with c2:
                sd_model_list = [
                    "Realistic_Vision_V5.0.safetensors",
                    "Counterfeit-V3.0_fp32.safetensors",
                    "epic_realism.safetensors",
                    "dreamshaper_v8.safetensors",
                    "deliberate_v3.safetensors"
                ]
                sd_model = st.selectbox("Which Stable Diffusion model would you like to use?", options=sd_model_list, key="sd_model")

            prompt_column_1, prompt_column_2 = st.columns([1, 1])

            with prompt_column_1:
                positive_prompt = st.text_area("Positive Prompt:", value=DefaultProjectSettingParams.batch_prompt, key="positive_prompt")
            
            with prompt_column_2:
                negative_prompt = st.text_area("Negative Prompt:", value=DefaultProjectSettingParams.batch_negative_prompt, key="negative_prompt")

            animate_col_1, animate_col_2, _ = st.columns([1, 1, 2])

            with animate_col_1:
                img_dimension_list = ["512x512", "512x768", "768x512"]
                img_dimension = st.selectbox("Image Dimension:", options=img_dimension_list, key="img_dimension")
            with animate_col_2:
                variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")
            
            normalise_speed = st.checkbox("Normalise Speed", value=True, key="normalise_speed")

            settings.update(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image_dimension=img_dimension,
                sampling_steps=30,
                motion_module=motion_module,
                model=sd_model,
                normalise_speed=normalise_speed
            )
        
        elif animation_tool == AnimationToolType.G_FILM.value:
            video_resolution = st.selectbox("Video Resolution:", options=["Full Resolution", "Preview"], key="video_resolution")
            

    elif current_animation_style == AnimationStyleType.IMAGE_TO_VIDEO.value:
        st.info("For image to video, you can select one or more prompts, and how many frames you want to generate for each prompt - it'll attempt to travel from one prompt to the next.")
        which_motion_module = st.selectbox("Which motion module would you like to use?", options=motion_modules, key="which_motion_module")

        # Initialize the list of dictionaries if it doesn't exist
        if 'travel_list' not in st.session_state:
            st.session_state['travel_list'] = []

        st.markdown("### Add to Prompt Travel List")
        prompt = st.text_area("Prompt")
        frame_count = st.number_input("How many frames would you like?", min_value=1, value=1, step=1)
        
        if st.button("Add to travel"):
            st.session_state['travel_list'].append({'prompt': prompt, 'frame_count': frame_count})

        st.markdown("***")
        st.markdown("### Travel List")

        # Display each item in the list
        if not st.session_state['travel_list']:
            st.error("The travel list is empty.")
        else:
            for i, item in enumerate(st.session_state['travel_list']):
                new_prompt = st.text_area(f"Prompt {i+1}", value=item['prompt'])
                bottom1, bottom2,bottom3 = st.columns([1, 2,1])
                with bottom1:
                    new_frame_count = st.number_input(f"Frame Count {i+1}", min_value=1, value=item['frame_count'], step=1)
                with bottom3:
                    if st.button(f"Delete Prompt {i+1}"):
                        del st.session_state['travel_list'][i]
                        st.rerun()
                # Update the item if it has been edited
                if new_prompt != item['prompt'] or new_frame_count != item['frame_count']:
                    st.session_state['travel_list'][i] = {'prompt': new_prompt, 'frame_count': new_frame_count}
                st.markdown("***")
        
        animate_col_1, animate_col_2 = st.columns([1, 3])

        with animate_col_1:
            variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")

    if st.button("Generate Animation Clip", key="generate_animation_clip"):
        vid_quality = "full" if video_resolution == "Full Resolution" else "preview"
        st.write("Generating animation clip...")
        settings.update(animation_style=current_animation_style)
        create_single_interpolated_clip(
            shot_uuid,
            vid_quality,
            settings,
            variant_count
        )
        st.rerun()
    