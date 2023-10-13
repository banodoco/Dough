
import streamlit as st
from shared.constants import ViewType

from ui_components.methods.common_methods import process_inference_output
from ui_components.methods.ml_methods import trigger_restyling_process
from ui_components.widgets.cropping_element import cropping_selector_element
from ui_components.widgets.frame_clip_generation_elements import  current_preview_video_element, update_animation_style_element
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.frame_style_clone_element import style_cloning_element
from ui_components.widgets.image_carousal import display_image
from ui_components.widgets.prompt_finder import prompt_finder_element
from ui_components.widgets.add_key_frame_element import add_key_frame, add_key_frame_element
from ui_components.widgets.styling_element import styling_element
from ui_components.widgets.timeline_view import timeline_view
from ui_components.widgets.variant_comparison_element import compare_to_previous_and_next_frame, compare_to_source_frame, variant_comparison_element
from ui_components.widgets.animation_style_element import animation_style_element
from ui_components.widgets.inpainting_element import inpainting_element
from ui_components.widgets.drawing_element import drawing_element
from ui_components.widgets.sidebar_logger import sidebar_logger
from ui_components.widgets.list_view import list_view_set_up, page_toggle, styling_list_view,motion_list_view
from utils import st_memory

from ui_components.constants import CreativeProcessType, WorkflowStageType

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
                
        if st.session_state['page'] == CreativeProcessType.MOTION.value:

            idx = st.session_state['current_frame_index'] - 1
                                    
            st.session_state['show_comparison'] = st_memory.radio("Show:", options=["Other Variants", "Preview Video in Context"], horizontal=True, key="show_comparison_radio_motion")

            if st.session_state['show_comparison'] == "Other Variants":
                variant_comparison_element(st.session_state['current_frame_uuid'])

            elif st.session_state['show_comparison'] == "Preview Video in Context":
                current_preview_video_element(st.session_state['current_frame_uuid'])
                        
            st.markdown("***")

            with st.expander("🎬 Choose Animation Style & Create Variants", expanded=True):

                animation_style_element(st.session_state['current_frame_uuid'], project_uuid)


        elif st.session_state['page'] == CreativeProcessType.STYLING.value:
            # carousal_of_images_element(project_uuid, stage=WorkflowStageType.STYLED.value)
            comparison_values = [
                "Other Variants", "Source Frame", "Previous & Next Frame", "None"]
            
            st.session_state['show_comparison'] = st_memory.radio("Show comparison to:", options=comparison_values, horizontal=True, key="show_comparison_radio")
            

            if st.session_state['show_comparison'] == "Other Variants":
                variant_comparison_element(st.session_state['current_frame_uuid'], stage=CreativeProcessType.STYLING.value)
                
            elif st.session_state['show_comparison'] == "Source Frame":
                compare_to_source_frame(timing_details)
                
            elif st.session_state['show_comparison'] == "Previous & Next Frame":

                compare_to_previous_and_next_frame(project_uuid,timing_details)

            elif st.session_state['show_comparison'] == "None":

                display_image(timing_uuid=st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)

            st.markdown("***")
                                                
            st.session_state['styling_view'] = st_memory.menu('',["Generate Variants", "Crop, Move & Rotate Image", "Inpainting & BG Removal","Draw On Image"], icons=['magic', 'crop', "paint-bucket", 'pencil'], menu_icon="cast", default_index=st.session_state.get('styling_view_index', 0), key="styling_view_selector", orientation="horizontal", styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#66A9BE"}})
                                                  
            if st.session_state['styling_view'] == "Generate Variants":

                with st.expander("🛠️ Generate Variants + Prompt Settings", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        styling_element(st.session_state['current_frame_uuid'], view_type=ViewType.SINGLE.value)
                    with col2:
                        detail1, detail2 = st.columns([1, 1])
                        with detail1:
                            st.session_state['individual_number_of_variants'] = st.number_input(
                                f"How many variants?", min_value=1, max_value=100, key=f"number_of_variants_{st.session_state['current_frame_index']}")

                        with detail2:
                            # TODO: add custom model validation such for sd img2img the value of strength can only be 1
                            if st.button(f"Generate variants", key=f"new_variations_{st.session_state['current_frame_index']}", help="This will generate new variants based on the settings to the left."):
                                for i in range(0, max(st.session_state['individual_number_of_variants'], 1)):
                                    trigger_restyling_process(
                                        timing_uuid=st.session_state['current_frame_uuid'], 
                                        model_uuid=st.session_state['model'], 
                                        prompt=st.session_state['prompt'], 
                                        strength=st.session_state['strength'], 
                                        negative_prompt=st.session_state['negative_prompt'], 
                                        guidance_scale=st.session_state['guidance_scale'], 
                                        seed=st.session_state['seed'], 
                                        num_inference_steps=st.session_state['num_inference_steps'], 
                                        transformation_stage=st.session_state['transformation_stage'], 
                                        promote_new_generation=st.session_state["promote_new_generation"], 
                                        custom_models=st.session_state['custom_models'], 
                                        adapter_type=st.session_state['adapter_type'], 
                                        update_inference_settings=True, 
                                        add_image_in_params=st.session_state['add_image_in_params'],
                                        low_threshold=st.session_state['low_threshold'], 
                                        high_threshold=st.session_state['high_threshold'],
                                        canny_image=st.session_state['canny_image'] if 'canny_image' in st.session_state else None,
                                        lora_model_1_url=st.session_state['lora_model_1_url'] if ('lora_model_1_url' in st.session_state and st.session_state['lora_model_1_url']) else None,
                                        lora_model_2_url=st.session_state['lora_model_2_url'] if ('lora_model_2_url' in st.session_state and st.session_state['lora_model_2_url']) else None,
                                        lora_model_3_url=st.session_state['lora_model_3_url'] if ('lora_model_3_url' in st.session_state and st.session_state['lora_model_3_url']) else None,
                                    )
                                st.rerun()

                        st.markdown("***")

                        st.info(
                            "You can restyle multiple frames at once in the List view.")

                        st.markdown("***")

                        style_cloning_element(timing_details)
                                    
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
                                        
                    drawing_element(timing_details,project_settings,project_uuid)
                                        
            with st.expander("➕ Add Key Frame", expanded=True):

                selected_image, inherit_styling_settings, how_long_after, which_stage_for_starting_image = add_key_frame_element(timing_details, project_uuid)

                if st.button(f"Add key frame",type="primary",use_container_width=True):
                                
                    add_key_frame(selected_image, inherit_styling_settings, how_long_after, which_stage_for_starting_image)
                    st.rerun()
                        
    elif st.session_state['frame_styling_view_type'] == "List View":
        
        st.markdown("---")

        header_col_1, header_col_2, header_col_3, header_col_4, header_col_5 = st.columns([1.25,0.25,4, 1.5, 1.5])
        with header_col_1:                    
            st.session_state['list_view_type'] = st_memory.radio("View type:", options=["Timeline View","Detailed View"], key="list_view_type_slider")
        
        with header_col_5:
            shift_frames_setting = st.toggle("Shift Frames", help="If set to True, it will shift the frames after your adjustment forward by the amount of time you move.")

        if st.session_state['list_view_type'] == "Detailed View":
            
            with header_col_4:                
                num_pages, items_per_page = list_view_set_up(timing_details, project_uuid)
                start_index, end_index = page_toggle(num_pages, items_per_page,project_uuid, position='top')
            
            st.markdown("***")
                                                                                    
            if st.session_state['page'] == "Styling":

                with st.sidebar:                            
                    styling_element(st.session_state['current_frame_uuid'], view_type=ViewType.LIST.value)
                
                styling_list_view(start_index, end_index, shift_frames_setting, project_uuid)
                                
                st.markdown("***")
            
            # Update the current page in session state
            elif st.session_state['page'] == "Motion":
                                                            
                motion_list_view(start_index, end_index, shift_frames_setting, project_uuid)

            start_index, end_index = page_toggle(num_pages, items_per_page,project_uuid, position='bottom')

        elif st.session_state['list_view_type'] == "Timeline View":


            
            if st.session_state['page'] == "Styling":
                with st.sidebar:        
                    with st.expander("🌀 Batch Styling", expanded=False):                                        
                        styling_element(st.session_state['current_frame_uuid'], view_type=ViewType.LIST.value)
                timeline_view(shift_frames_setting, project_uuid, "Styling", header_col_3, header_col_4)
            elif st.session_state['page'] == "Motion":
                timeline_view(shift_frames_setting, project_uuid, "Motion", header_col_3, header_col_4)
      
    with st.sidebar:
        with st.expander("🔍 Inference Logging", expanded=True):
                        
            sidebar_logger(data_repo, project_uuid)
