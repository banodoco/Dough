
import streamlit as st
from ui_components.common_methods import create_individual_clip, prompt_interpolation_model,create_video_without_interpolation,create_single_preview_video,create_full_preview_video,get_timing_details
from repository.local_repo.csv_repo import CSVProcessor, get_app_settings, get_project_settings, update_project_setting, update_specific_timing_value

which_function = st.selectbox("Which function?", ["create_individual_clip", "prompt_interpolation_model", "create_video_without_interpolation", "create_single_preview_video", "create_full_preview_video"])

morphing_type = st.selectbox("Morphing type?", ["Interpolation", "Direct Morphing"])

project_name = "on_the_precicipce"

which_item = int(st.number_input("Which item?", min_value=0, max_value=10, value=0))

if st.button("Run function"):
    update_specific_timing_value(project_name, which_item, "animation_style", morphing_type)
    update_specific_timing_value(project_name, which_item, "interpolation_steps", 3)
    timing_details = get_timing_details(project_name)
    if which_function == "create_individual_clip":
        output_video = create_individual_clip(int(which_item), project_name)
    elif which_function == "prompt_interpolation_model":
        output_video = prompt_interpolation_model(which_item, project_name)
    elif which_function == "create_video_without_interpolation":
        output_video = create_video_without_interpolation(which_item,project_name)
    elif which_function == "create_single_preview_video":
        output_video =  create_single_preview_video(which_item, project_name)
    elif which_function == "create_full_preview_video":
        output_video = create_full_preview_video(project_name, which_item)

    st.write(output_video)

    st.video(output_video)
