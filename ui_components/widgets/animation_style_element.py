import time
import streamlit as st
from typing import List
from shared.constants import AnimationStyleType, AnimationToolType
from ui_components.constants import DefaultProjectSettingParams
from ui_components.methods.video_methods import create_single_interpolated_clip
from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.motion_module import AnimateDiffCheckpoint
from ui_components.models import InternalFrameTimingObject, InternalShotObject

def animation_style_element(shot_uuid):
    motion_modules = AnimateDiffCheckpoint.get_name_list()
    variant_count = 1
    current_animation_style = AnimationStyleType.INTERPOLATION.value    # setting a default value
    data_repo = DataRepo()

    # if current_animation_style == AnimationStyleType.INTERPOLATION.value:
    animation_type = st.radio("Animation Interpolation:", options=['Creative Interpolation', "Video To Video"], key="animation_tool", horizontal=True, disabled=True)

    if animation_type == "Creative Interpolation":
        st.markdown("***")
        
        shot: InternalShotObject = data_repo.get_shot_from_uuid(st.session_state["shot_uuid"])
        timing_list: List[InternalFrameTimingObject] = shot.timing_list
        st.markdown("#### Keyframe Settings")
        if timing_list and len(timing_list):
            columns = st.columns(len(timing_list)) 
            disable_generate = False
            help = ""
            for idx, timing in enumerate(timing_list):
                if timing.primary_image and timing.primary_image.location:
                    columns[idx].image(timing.primary_image.location, use_column_width=True)
                    b = timing.primary_image.inference_params
                    prompt = columns[idx].text_area(f"Prompt {idx+1}", value=(b['prompt'] if b else ""), key=f"prompt_{idx+1}")                        
                else:
                    columns[idx].warning("No primary image present")
                    disable_generate = True
                    help = "You can't generate a video because one of your keyframes is missing an image."
        else:
            st.warning("No keyframes present")

        st.markdown("***")
        video_resolution = None

        settings = {
            "animation_tool": animation_type
        }

        st.markdown("#### Overall Settings")        
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
            vae_list = [
                "Baked",
                "Standard"]
            vae = st.selectbox("Which VAE would you like to use?", options=vae_list, key="vae_model")

        d1, d2 = st.columns([1, 1])

        with d1:
            ip_adapter_strength = st.slider("IP Adapter Strength", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="ip_adapter_strength")
        
        with d2:
            ip_adapter_noise = st.slider("IP Adapter Noise", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="ip_adapter_noise")

        interpolation_style = st.selectbox("Interpolation Style", options=["Big Dipper", "Linear", "Slerp", "Custom"], key="interpolation_style")
        if interpolation_style == "Big Dipper":
            interpolation_settings = "0=1.0,1=0.99,2=0.97,3=0.95,4=0.92,5=0.9,6=0.86,7=0.83,8=0.79,9=0.75,10=0.71,11=0.67,12=0.62,13=0.58,14=0.53,15=0.49,16=0.44,17=0.39,18=0.35,19=0.31,20=0.26,21=0.22,22=0.19,23=0.15,24=0.12,25=0.09,26=0.06,27=0.04,28=0.02,29=0.01,30=0.0,31=0.0,32=0.0,33=0.0,34=0.0,35=0.0,36=0.0,37=0.0,38=0.0,39=0.0,40=0.0,41=0.0,42=0.0,43=0.0,44=0.0,45=0.0,46=0.0,47=0.0,48=0.0,49=0.0,50=0.0,51=0.0,52=0.0,53=0.0,54=0.0,55=0.0,56=0.0,57=0.0,58=0.0,59=0.0,60=0.0,61=0.0,62=0.0,63=0.0"
        elif interpolation_style == "Linear":
            interpolation_settings = "0=1.0,1=0.99,2=0.97,3=0.95,4=0.92,5=0.9,6=0.86,7=0.83,8=0.79,9=0.75,10=0.71,11=0.67,12=0.62,13=0.58,14=0.53,15=0.49,16=0.44,17=0.39,18=0.35,19=0.31,20=0.26,21=0.22,22=0.19,23=0.15,24=0.12,25=0.09,26=0.06,27=0.04,28=0.02,29=0.01,30=0.0,31=0.0,32=0.0,33=0.0,34=0.0,35=0.0,36=0.0,37=0.0,38=0.0,39=0.0,40=0.0,41=0.0,42=0.0,43=0.0,44=0.0,45=0.0,46=0.0,47=0.0,48=0.0,49=0.0,50=0.0,51=0.0,52=0.0,53=0.0,54=0.0,55=0.0,56=0.0,57=0.0,58=0.0,59=0.0,60=0.0,61=0.0,62=0.0,63=0.0"
        elif interpolation_style == "Slerp":
            interpolation_settings = "0=1.0,1=0.99,2=0.97,3=0.95,4=0.92,5=0.9,6=0.86,7=0.83,8=0.79,9=0.75,10=0.71,11=0.67,12=0.62,13=0.58,14=0.53,15=0.49,16=0.44,17=0.39,18=0.35,19=0.31,20=0.26,21=0.22,22=0.19,23=0.15,24=0.12,25=0.09,26=0.06,27=0.04,28=0.02,29=0.01,30=0.0,31=0.0,32=0.0,33=0.0,34=0.0,35=0.0,36=0.0,37=0.0,38=0.0,39=0.0,40=0.0,41=0.0,42=0.0,43=0.0,44=0.0,45=0.0,46=0.0,47=0.0,48=0.0,49=0.0,50=0.0,51=0.0,52=0.0,53=0.0,54=0.0,55=0.0,56=0.0,57=0.0,58=0.0,59=0.0,60=0.0,61=0.0,62=0.0,63=0.0"
        if interpolation_style == "Custom":
            interpolation_settings = st.text_area("Custom Interpolation Style", value="0=1.0,1=0.99,2=0.97,3=0.95,4=0.92,5=0.9,6=0.86,7=0.83,8=0.79,9=0.75,10=0.71,11=0.67,12=0.62,13=0.58,14=0.53,15=0.49,16=0.44,17=0.39,18=0.35,19=0.31,20=0.26,21=0.22,22=0.19,23=0.15,24=0.12,25=0.09,26=0.06,27=0.04,28=0.02,29=0.01,30=0.0,31=0.0,32=0.0,33=0.0,34=0.0,35=0.0,36=0.0,37=0.0,38=0.0,39=0.0,40=0.0,41=0.0,42=0.0,43=0.0,44=0.0,45=0.0,46=0.0,47=0.0,48=0.0,49=0.0,50=0.0,51=0.0,52=0.0,53=0.0,54=0.0,55=0.0,56=0.0,57=0.0,58=0.0,59=0.0,60=0.0,61=0.0,62=0.0,63=0.0", key="custom_interpolation_style")

        st.markdown("***")
        st.markdown("#### Generation Settings")
        animate_col_1, _, _ = st.columns([1, 1, 2])

        with animate_col_1:
            # img_dimension_list = ["512x512", "512x768", "768x512"]
            # img_dimension = st.selectbox("Image Dimension:", options=img_dimension_list, key="img_dimension")  
            project_settings = data_repo.get_project_setting(shot.project.uuid)
            width = project_settings.width
            height = project_settings.height
            img_dimension = f"{width}x{height}"                
            variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")
        normalise_speed = True
        # normalise_speed = st.checkbox("Normalise Speed", value=True, key="normalise_speed")

        settings.update(
            # positive_prompt=positive_prompt,
            # negative_prompt=negative_prompt,
            image_dimension=img_dimension,
            sampling_steps=30,
            motion_module=motion_module,
            model=sd_model,
            normalise_speed=normalise_speed
        )
    
    elif animation_type == "Image To Video":
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
        
        st.markdown("***")

        animate_col_1, animate_col_2 = st.columns([1, 3])

        with animate_col_1:
            variant_count = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="variant_count")
    
    if st.button("Generate Animation Clip", key="generate_animation_clip", disabled=disable_generate, help=help):
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
    