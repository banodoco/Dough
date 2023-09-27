import streamlit as st
from streamlit_image_comparison import image_comparison
import time
from PIL import Image
from ui_components.methods.common_methods import delete_frame, drawing_mode, ai_frame_editing_element, clone_styling_settings,add_key_frame,jump_to_single_frame_view_button
from ui_components.methods.ml_methods import trigger_restyling_process
from ui_components.methods.video_methods import create_or_get_single_preview_video
from ui_components.widgets.cropping_element import manual_cropping_element, precision_cropping_element, cropping_selector_element
from ui_components.widgets.frame_clip_generation_elements import current_individual_clip_element, current_preview_video_element, update_animation_style_element
from ui_components.widgets.frame_time_selector import single_frame_time_selector, single_frame_time_duration_setter
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.image_carousal import display_image
from ui_components.widgets.prompt_finder import prompt_finder_element
from ui_components.widgets.add_key_frame_element import add_key_frame_element
from ui_components.widgets.styling_element import styling_element
from ui_components.widgets.timeline_view import timeline_view
from ui_components.widgets.compare_to_other_variants import compare_to_other_variants
from ui_components.widgets.animation_style_element import animation_style_element
from ui_components.widgets.inpainting_element import inpainting_element
from ui_components.widgets.list_view import list_view_set_up, page_toggle, styling_list_view,motion_list_view
from streamlit_option_menu import option_menu
from utils import st_memory


import math
from ui_components.constants import WorkflowStageType
from utils.constants import ImageStage

from utils.data_repo.data_repo import DataRepo


def frame_styling_page(mainheader2, project_uuid: str):
    data_repo = DataRepo()

    timing_details = data_repo.get_timing_list_from_project(project_uuid)
    
    project_settings = data_repo.get_project_setting(project_uuid)

    if "strength" not in st.session_state:
        st.session_state['strength'] = project_settings.default_strength
        st.session_state['prompt_value'] = project_settings.default_prompt
        st.session_state['model'] = project_settings.default_model.uuid
        st.session_state['custom_pipeline'] = project_settings.default_custom_pipeline
        st.session_state['negative_prompt_value'] = project_settings.default_negative_prompt
        st.session_state['guidance_scale'] = project_settings.default_guidance_scale
        st.session_state['seed'] = project_settings.default_seed
        st.session_state['num_inference_steps'] = project_settings.default_num_inference_steps
        st.session_state['transformation_stage'] = project_settings.default_stage
        st.session_state['show_comparison'] = "Don't show"
    
    if "current_frame_uuid" not in st.session_state:
        timing = data_repo.get_timing_list_from_project(project_uuid)[0]
        st.session_state['current_frame_uuid'] = timing.uuid
    
    
    if 'frame_styling_view_type' not in st.session_state:
        st.session_state['frame_styling_view_type'] = "Individual View"
        st.session_state['frame_styling_view_type_index'] = 0


    if st.session_state['change_view_type'] == True:  
        st.session_state['change_view_type'] = False
        # round down st.session_state['which_image']to nearest 10

    
    if st.session_state['frame_styling_view_type'] == "List View":
        st.markdown(
            f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{st.session_state['page']}]")
    else:
        st.markdown(
            f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{st.session_state['page']}] > :blue[Frame #{st.session_state['current_frame_index']}]")

    project_settings = data_repo.get_project_setting(project_uuid)

    if st.session_state['frame_styling_view_type'] == "Individual View":
        with st.sidebar:
            frame_selector_widget()
                
        if st.session_state['page'] == "Motion":
                                                
            st.session_state['show_comparison'] = st_memory.radio("Show:", options=["Other Variants", "Preview Video in Context"], horizontal=True, key="show_comparison_radio_motion")

            if st.session_state['show_comparison'] == "Other Variants":
                compare_to_other_variants(timing_details, project_uuid, data_repo,stage="Motion")

            elif st.session_state['show_comparison'] == "Preview Video in Context":
                current_preview_video_element(st.session_state['current_frame_uuid'])
                        
            st.markdown("***")

            with st.expander("🎬 Choose Animation Style & Create Variants", expanded=True):

                update_animation_style_element(st.session_state['current_frame_uuid'], horizontal=True)

                animation_style_element(st.session_state['current_frame_uuid'], project_settings)


        elif st.session_state['page'] == "Styling":
            # carousal_of_images_element(project_uuid, stage=WorkflowStageType.STYLED.value)
            comparison_values = ["Other Variants", "Source Frame", "Previous & Next Frame", "None"]
            
            st.session_state['show_comparison'] = st_memory.radio("Show comparison to:", options=comparison_values, horizontal=True, key="show_comparison_radio")            

            if st.session_state['show_comparison'] == "Other Variants":
                compare_to_other_variants(timing_details, project_uuid, data_repo,stage="Styling")
                
            elif st.session_state['show_comparison'] == "Source Frame":
                if timing_details[st.session_state['current_frame_index']- 1].primary_image:
                    img2 = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
                else:
                    img2 = 'https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'
                
                img1 = timing_details[st.session_state['current_frame_index'] - 1].source_image.location if timing_details[st.session_state['current_frame_index'] - 1].source_image else 'https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'
                
                image_comparison(starting_position=50,
                                    img1=img1,
                                    img2=img2, make_responsive=False, label1=WorkflowStageType.SOURCE.value, label2=WorkflowStageType.STYLED.value)
                
            elif st.session_state['show_comparison'] == "Previous & Next Frame":

                mainimages1, mainimages2, mainimages3 = st.columns([1, 1, 1])

                with mainimages1:
                    if st.session_state['current_frame_index'] - 2 >= 0:
                        previous_image = data_repo.get_timing_from_frame_number(project_uuid, frame_number=st.session_state['current_frame_index'] - 2)
                        st.info(f"Previous image")
                        display_image(
                            timing_uuid=previous_image.uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

                        if st.button(f"Preview Interpolation From #{st.session_state['current_frame_index']-1} to #{st.session_state['current_frame_index']}", key=f"Preview Interpolation From #{st.session_state['current_frame_index']-1} to #{st.session_state['current_frame_index']}", use_container_width=True):
                            prev_frame_timing = data_repo.get_prev_timing(st.session_state['current_frame_uuid'])
                            create_or_get_single_preview_video(prev_frame_timing.uuid)
                            prev_frame_timing = data_repo.get_timing_from_uuid(prev_frame_timing.uuid)
                            st.video(prev_frame_timing.timed_clip.location)

                with mainimages2:
                    st.success(f"Current image")
                    display_image(
                        timing_uuid=st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)

                with mainimages3:
                    if st.session_state['current_frame_index'] + 1 <= len(timing_details):
                        next_image = data_repo.get_timing_from_frame_number(project_uuid, frame_number=st.session_state['current_frame_index'])
                        st.info(f"Next image")
                        display_image(timing_uuid=next_image.uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

                        if st.button(f"Preview Interpolation From #{st.session_state['current_frame_index']} to #{st.session_state['current_frame_index']+1}", key=f"Preview Interpolation From #{st.session_state['current_frame_index']} to #{st.session_state['current_frame_index']+1}", use_container_width=True):
                            create_or_get_single_preview_video(
                                st.session_state['current_frame_uuid'])
                            current_frame = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                            st.video(current_frame.timed_clip.location)

            elif st.session_state['show_comparison'] == "None":
                display_image(
                    timing_uuid=st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)

            st.markdown("***")

            if 'styling_view_index' not in st.session_state:
                st.session_state['styling_view_index'] = 0
                st.session_state['change_styling_view_type'] = False
                
            styling_views = ["Generate Variants", "Crop, Move & Rotate Image", "Inpainting & BG Removal","Draw On Image"]
                                                                                                    
            st.session_state['styling_view'] = option_menu(None, styling_views, icons=['magic', 'crop', "paint-bucket", 'pencil'], menu_icon="cast", default_index=st.session_state['styling_view_index'], key="styling_view_selector", orientation="horizontal", styles={
                                                                    "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#66A9BE"}})
                                    
            if st.session_state['styling_view_index'] != styling_views.index(st.session_state['styling_view']):
                st.session_state['styling_view_index'] = styling_views.index(st.session_state['styling_view'])                                                       

            if st.session_state['styling_view'] == "Generate Variants":

                with st.expander("🛠️ Generate Variants + Prompt Settings", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        styling_element(st.session_state['current_frame_uuid'], view_type="Single")
                    with col2:
                        detail1, detail2 = st.columns([1, 1])
                        with detail1:
                            st.session_state['individual_number_of_variants'] = st.number_input(
                                f"How many variants?", min_value=1, max_value=100, key=f"number_of_variants_{st.session_state['current_frame_index']}")

                        with detail2:
                            st.write("")
                            st.write("")

                            # TODO: add custom model validation such for sd img2img the value of strength can only be 1
                            if st.button(f"Generate variants", key=f"new_variations_{st.session_state['current_frame_index']}", help="This will generate new variants based on the settings to the left."):
                                for i in range(0, st.session_state['individual_number_of_variants']):
                                    trigger_restyling_process(
                                        st.session_state['current_frame_uuid'], 
                                        st.session_state['model'], 
                                        st.session_state['prompt'], 
                                        st.session_state['strength'], 
                                        st.session_state['negative_prompt'], 
                                        st.session_state['guidance_scale'], 
                                        st.session_state['seed'], 
                                        st.session_state['num_inference_steps'], 
                                        st.session_state['transformation_stage'], 
                                        st.session_state["promote_new_generation"], 
                                        st.session_state['custom_models'], 
                                        st.session_state['adapter_type'], 
                                        True, 
                                        st.session_state['low_threshold'], 
                                        st.session_state['high_threshold']
                                    )
                                st.experimental_rerun()

                        st.markdown("***")

                        st.info(
                            "You can restyle multiple frames at once in the List view.")

                        st.markdown("***")

                        open_copier = st.checkbox(
                            "Copy styling settings from another frame")
                        if open_copier is True:
                            copy1, copy2 = st.columns([1, 1])
                            with copy1:
                                which_frame_to_copy_from = st.number_input("Which frame would you like to copy styling settings from?", min_value=1, max_value=len(
                                    timing_details), value=st.session_state['current_frame_index'], step=1)
                                if st.button("Copy styling settings from this frame"):
                                    clone_styling_settings(which_frame_to_copy_from - 1, st.session_state['current_frame_uuid'])
                                    st.experimental_rerun()

                            with copy2:
                                display_image(
                                    idx=which_frame_to_copy_from, stage=WorkflowStageType.STYLED.value, clickable=False, timing_details=timing_details)
                                st.caption("Prompt:")
                                st.caption(
                                    timing_details[which_frame_to_copy_from].prompt)
                                if timing_details[which_frame_to_copy_from].model is not None:
                                    st.caption("Model:")
                                    st.caption(
                                        timing_details[which_frame_to_copy_from].model.name)
                                    
                with st.expander("🔍 Prompt Finder"):
                    prompt_finder_element(project_uuid)
            
            elif st.session_state['styling_view'] == "Crop, Move & Rotate Image":
                with st.expander("🤏 Crop, Move & Rotate Image", expanded=True):                    
                    cropping_selector_element(project_uuid)

            elif st.session_state['styling_view'] == "Inpainting & BG Removal":

                with st.expander("🌌 Inpainting, Background Removal & More", expanded=True):
                    
                    inpainting_element(st.session_state['current_frame_uuid'])

            elif st.session_state['styling_view'] == "Draw On Image":
                with st.expander("📝 Draw On Image", expanded=True):

                    which_stage_to_draw_on = st_memory.radio("Which stage to work on?", ["Styled Key Frame", "Unedited Key Frame"], horizontal=True, key="which_stage_drawing")
                    if which_stage_to_draw_on == "Styled Key Frame":
                        drawing_mode(timing_details,project_settings,project_uuid, stage=WorkflowStageType.STYLED.value)
                    elif which_stage_to_draw_on == "Unedited Key Frame":
                        drawing_mode(timing_details,project_settings,project_uuid, stage=WorkflowStageType.SOURCE.value)

            with st.expander("➕ Add Key Frame", expanded=True):

                selected_image, inherit_styling_settings, how_long_after, which_stage_for_starting_image = add_key_frame_element(timing_details, project_uuid)

                if st.button(f"Add key frame",type="primary",use_container_width=True):
                                
                    add_key_frame(selected_image, inherit_styling_settings, how_long_after, which_stage_for_starting_image)
                    st.experimental_rerun()

                        

    elif st.session_state['frame_styling_view_type'] == "List View":
        
        st.markdown("---")

        header_col_1, header_col_2, header_col_3 = st.columns([1, 5, 1])
        with header_col_1:                    
            st.session_state['list_view_type'] = st_memory.radio("View type:", options=["Timeline View","Detailed View"], key="list_view_type_slider")
        
        with header_col_3:
            shift_frames_setting = st.toggle("Shift Frames", help="If set to True, it will shift the frames after your adjustment forward by the amount of time you move.")

        if st.session_state['list_view_type'] == "Detailed View":
            
            with header_col_2:                
                num_pages, items_per_page = list_view_set_up(timing_details, project_uuid)
                start_index, end_index = page_toggle(num_pages, items_per_page,project_uuid)
            
            st.markdown("***")
                                                                                    
            if st.session_state['page'] == "Styling":

                with st.sidebar:                            
                    styling_element(st.session_state['current_frame_uuid'], view_type="List")
                
                styling_list_view(start_index, end_index, shift_frames_setting, project_uuid)
                                
                st.markdown("***")

            # Update the current page in session state
            elif st.session_state['page'] == "Motion":
                                                            
                motion_list_view(start_index, end_index, shift_frames_setting, project_uuid)

        elif st.session_state['list_view_type'] == "Timeline View":

            with header_col_2:  
                items_per_row = st.slider("How many frames per row?", min_value=1, max_value=10, value=5, step=1, key="items_per_row")
            with header_col_3:
                expand_all = st_memory.toggle("Expand All", key="expand_all")

            if st.session_state['page'] == "Styling":
                timeline_view(shift_frames_setting, project_uuid, items_per_row,expand_all,"Styling")
            elif st.session_state['page'] == "Motion":
                timeline_view(shift_frames_setting, project_uuid, items_per_row,expand_all,"Motion")

                


