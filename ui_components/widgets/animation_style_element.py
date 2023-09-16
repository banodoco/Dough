import time
import streamlit as st
from typing import List
from utils.data_repo.data_repo import DataRepo

def animation_style_element(current_frame_uuid, project_uuid):
    motion_modules = ["mm-v15-v2", "AD_Stabilized_Motion","TemporalDiff"]
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)
    current_animation_style = data_repo.get_timing_from_uuid(current_frame_uuid).animation_style

    if current_animation_style == "Interpolation":
        animation_tool = st.radio("Animation Tool:", options=['Animatediff', 'Google FiLM'], key="animation_tool", horizontal=True)
        video_resolution = st.radio("Video Resolution:", options=["Preview Resolution", "Full Resolution"], key="video_resolution", horizontal=True)

        if animation_tool == "Animatediff":
            which_motion_module = st.selectbox("Which motion module would you like to use?", options=motion_modules, key="which_motion_module")
            prompt_column_1, prompt_column_2 = st.columns([1, 1])

            with prompt_column_1:
                starting_prompt = st.text_area("Starting Prompt:", value=project_settings.default_prompt, key="starting_prompt")
            
            with prompt_column_2:
                ending_prompt = st.text_area("Ending Prompt:", value=project_settings.default_prompt, key="ending_prompt")

            animate_col_1, animate_col_2 = st.columns([1, 3])

            with animate_col_1:
                how_many_variants = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="how_many_variants")
            
            normalise_speed = st.checkbox("Normalise Speed", value=True, key="normalise_speed")
    
        if st.button("Generate Animation Clip", key="generate_animation_clip"):
            for _ in range(how_many_variants):
                st.write("Generating animation clip...")
                time.sleep(2)
                st.write("Lol, jk, this isn't done yet")
                time.sleep(2)
                st.experimental_rerun()

    elif current_animation_style == "Image to Video":
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
                        st.experimental_rerun()
                # Update the item if it has been edited
                if new_prompt != item['prompt'] or new_frame_count != item['frame_count']:
                    st.session_state['travel_list'][i] = {'prompt': new_prompt, 'frame_count': new_frame_count}
                st.markdown("***")
        
        animate_col_1, animate_col_2 = st.columns([1, 3])

        with animate_col_1:
            how_many_variants = st.number_input("How many variants?", min_value=1, max_value=100, value=1, step=1, key="how_many_variants")

        if st.button("Generate Animation Clip", key="generate_animation_clip"):
            for _ in range(how_many_variants):
                st.write("Generating animation clip...")
                time.sleep(2)
                st.write("Lol, jk, this isn't done yet")
                time.sleep(2)
                st.experimental_rerun()
    else:
        st.error("No animation style selected")