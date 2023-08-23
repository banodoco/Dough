import streamlit as st
from streamlit_image_comparison import image_comparison
import time
from PIL import Image
from ui_components.common_methods import delete_frame, drawing_mode, promote_image_variant, save_uploaded_image, trigger_restyling_process, create_timings_row_at_frame_number, convert_to_minutes_and_seconds, create_full_preview_video, move_frame, calculate_desired_duration_of_individual_clip, create_or_get_single_preview_video, calculate_desired_duration_of_individual_clip, apply_image_transformations, get_pillow_image, ai_frame_editing_element, clone_styling_settings, zoom_inputs,calculate_dynamic_interpolations_steps,create_individual_clip,update_speed_of_video_clip,current_individual_clip_element,current_preview_video_element,update_animation_style_element
from ui_components.widgets.cropping_element import manual_cropping_element, precision_cropping_element
from ui_components.widgets.frame_switch_btn import back_and_forward_buttons
from ui_components.widgets.frame_time_selector import single_frame_time_selector
from ui_components.widgets.frame_selector import frame_selector_widget
from ui_components.widgets.image_carousal import carousal_of_images_element, display_image
from ui_components.widgets.prompt_finder import prompt_finder_element
from ui_components.widgets.styling_element import styling_element
from streamlit_option_menu import option_menu
from utils import st_memory


import math
from ui_components.constants import WorkflowStageType
from streamlit_extras.annotated_text import annotated_text
from utils.common_utils import save_or_host_file
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
    
    '''
    if "current_frame_uuid" not in st.session_state:
        timing = data_repo.get_timing_list_from_project(project_uuid)[0]
        st.session_state['current_frame_uuid'] = timing.uuid
    '''
    
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

            idx = st.session_state['current_frame_index'] - 1


            timing1, timing2, timing3 = st.columns([0.5, 1,1])

            with timing1:
                update_animation_style_element(st.session_state['current_frame_uuid'], horizontal=False)

            with timing2:
                num_timing_details = len(timing_details)
                current_individual_clip_element(st.session_state['current_frame_uuid'])

            with timing3:
                current_preview_video_element(st.session_state['current_frame_uuid'])

        elif st.session_state['page'] == "Styling":
            # carousal_of_images_element(project_uuid, stage=WorkflowStageType.STYLED.value)
            comparison_values = [
                "Other Variants", "Source Frame", "Previous & Next Frame", "None"]

            st.session_state['show_comparison'] = st_memory.radio(
                "Show comparison to:", options=comparison_values, horizontal=True, project_settings=project_settings, key="show_comparison_radio")

            timing = data_repo.get_timing_from_uuid(
                    st.session_state['current_frame_uuid'])
            variants = timing.alternative_images_list

            if st.session_state['show_comparison'] == "Other Variants":

                mainimages1, mainimages2 = st.columns([1, 1])

                aboveimage1, aboveimage2, aboveimage3 = st.columns(
                    [1, 0.25, 0.75])

                with aboveimage1:
                    st.info(
                        f"Current variant = {timing_details[st.session_state['current_frame_index'] - 1].primary_variant_index + 1}")

                with aboveimage2:
                    show_more_than_10_variants = st.checkbox(
                        "Show >10 variants", key="show_more_than_10_variants")

                with aboveimage3:
                    number_of_variants = len(variants)

                    if show_more_than_10_variants is True:
                        current_variant = int(
                            timing_details[st.session_state['current_frame_index'] - 1].primary_variant_index)
                        which_variant = st.radio(f'Main variant = {current_variant + 1}', range(1, 
                            number_of_variants + 1), index=number_of_variants-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_index']}")
                    else:
                        last_ten_variants = range(
                            max(1, number_of_variants - 10), number_of_variants + 1)
                        current_variant = int(
                            timing_details[st.session_state['current_frame_index'] - 1].primary_variant_index)
                        which_variant = st.radio(f'Main variant = {current_variant + 1}', last_ten_variants, index=len(
                            last_ten_variants)-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_index']}")

                with mainimages1:

                    project_settings = data_repo.get_project_setting(project_uuid)
                    st.success("**Main variant**")
                    if len(timing_details[st.session_state['current_frame_index'] - 1].alternative_images_list):
                        st.image(timing_details[st.session_state['current_frame_index'] - 1].primary_image_location,
                                    use_column_width=True)
                    else:
                        st.error("No variants found for this frame")

                with mainimages2:

                    if len(timing_details[st.session_state['current_frame_index'] - 1].alternative_images_list):
                        if which_variant - 1 == current_variant:
                            st.success("**Main variant**")

                        else:
                            st.info(f"**Variant #{which_variant}**")
                        
                        st.image(variants[which_variant- 1].location,
                                    use_column_width=True)

                        if which_variant- 1 != current_variant:

                            if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['current_frame_index']}", help="Promote this variant to the primary image"):
                                promote_image_variant(
                                    st.session_state['current_frame_uuid'], which_variant - 1)
                                time.sleep(0.5)
                                st.experimental_rerun()

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


            styling_views = ["Generate Variants", "Crop, Move & Rotate Image", "Inpainting & Background Removal","Draw On Image"]
            
            st.session_state['styling_view'] = option_menu(None, styling_views, icons=['magic', 'crop', "paint-bucket", 'pencil'], menu_icon="cast", default_index=0, key="styling_view_selector", orientation="horizontal", styles={
                                                                    "nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"}, "nav-link-selected": {"background-color": "#66A9BE"}})


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
                                    trigger_restyling_process(st.session_state['current_frame_uuid'], st.session_state['model'], st.session_state['prompt'], st.session_state['strength'], st.session_state['negative_prompt'], st.session_state['guidance_scale'], st.session_state['seed'], st.session_state[
                                                                'num_inference_steps'], st.session_state['transformation_stage'], st.session_state["promote_new_generation"], st.session_state['custom_models'], st.session_state['adapter_type'], True, st.session_state['low_threshold'], st.session_state['high_threshold'])
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
                    
                    selector1, selector2, selector3 = st.columns([1, 1, 1])
                    with selector1:
                        which_stage = st.radio("Which stage to work on?", ["Styled Key Frame", "Unedited Key Frame"], key="which_stage", horizontal=True)
                    with selector2:
                        how_to_crop = st_memory.radio("How to crop:", options=["Precision Cropping","Manual Cropping"], project_settings=project_settings, key="how_to_crop",horizontal=True)
                                            
                    if which_stage == "Styled Key Frame":
                        stage_name = WorkflowStageType.STYLED.value
                    elif which_stage == "Unedited Key Frame":
                        stage_name = WorkflowStageType.SOURCE.value
                                            
                    if how_to_crop == "Manual Cropping":
                        manual_cropping_element(stage_name, st.session_state['current_frame_uuid'])
                    elif how_to_crop == "Precision Cropping":
                        precision_cropping_element(stage_name, project_uuid)
                                
            elif st.session_state['styling_view'] == "Inpainting & Background Removal":

                with st.expander("🌌 Inpainting, Background Removal & More", expanded=True):
                    
                    which_stage_to_inpaint = st.radio("Which stage to work on?", ["Styled Key Frame", "Unedited Key Frame"], horizontal=True, key="which_stage_inpainting")
                    if which_stage_to_inpaint == "Styled Key Frame":
                        inpainting_stage = WorkflowStageType.STYLED.value
                    elif which_stage_to_inpaint == "Unedited Key Frame":
                        inpainting_stage = WorkflowStageType.SOURCE.value
                    
                    ai_frame_editing_element(st.session_state['current_frame_uuid'], inpainting_stage)

            elif st.session_state['styling_view'] == "Draw On Image":
                with st.expander("📝 Draw On Image", expanded=True):

                    which_stage_to_draw_on = st.radio("Which stage to work on?", ["Styled Key Frame", "Unedited Key Frame"], horizontal=True, key="which_stage_drawing")
                    if which_stage_to_draw_on == "Styled Key Frame":
                        drawing_mode(timing_details,project_settings,project_uuid, stage=WorkflowStageType.STYLED.value)
                    elif which_stage_to_draw_on == "Unedited Key Frame":
                        drawing_mode(timing_details,project_settings,project_uuid, stage=WorkflowStageType.SOURCE.value)
                        

    elif st.session_state['frame_styling_view_type'] == "List View":

                                                
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 1
        
        if not('index_of_current_page' in st.session_state and st.session_state['index_of_current_page']):
            st.session_state['index_of_current_page'] = 1

        items_per_page = 10
        num_pages = math.ceil(len(timing_details) / items_per_page) + 1
        
        st.markdown("---")
        
        st.session_state['current_page'] = st.radio("Select Page:", options=range(
            1, num_pages), horizontal=True, index=st.session_state['index_of_current_page'] - 1, key="page_selection_radio")

        if st.session_state['current_page'] != st.session_state['index_of_current_page']:
            st.session_state['index_of_current_page'] = st.session_state['current_page']
            st.experimental_rerun()

        st.markdown("---")

        start_index = (st.session_state['current_page'] - 1) * items_per_page         
        end_index = min(start_index + items_per_page,
                        len(timing_details))

                                                                                
        if st.session_state['page'] == "Styling":
            with st.sidebar:                            
                styling_element(st.session_state['current_frame_uuid'], view_type="List")

            timing_details = data_repo.get_timing_list_from_project(project_uuid)

            for i in range(start_index, end_index):
                index_of_current_item = i                              
                st.subheader(f"Frame {i+1}")
                image1, image2, image3 = st.columns([2, 3, 2])

                with image1:
                    display_image(
                        timing_uuid=timing_details[i].uuid, stage=WorkflowStageType.SOURCE.value, clickable=False)

                with image2:
                    display_image(
                        timing_uuid=timing_details[i].uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

                with image3:
                    time1, time2 = st.columns([1, 1])
                    with time1:
                        single_frame_time_selector(timing_details[i].uuid, 'sidebar')
                        st.info(
                            f"Duration: {timing_details[i].clip_duration:.2f} secs")

                    with time2:
                        st.write("") 

                    if st.button(f"Jump to single frame view for #{index_of_current_item}"):
                        st.session_state['current_frame_index'] = index_of_current_item
                        st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid
                        st.session_state['frame_styling_view_type'] = "Individual View"
                        st.session_state['change_view_type'] = True
                        st.experimental_rerun()
                    st.markdown("---")
                    btn1, btn2, btn3 = st.columns([2, 1, 1])
                    with btn1:
                        if st.button("Delete this keyframe", key=f'{index_of_current_item}'):
                            delete_frame(timing_details[i].uuid)
                            st.experimental_rerun()
                    with btn2:
                        if st.button("⬆️", key=f"Promote {index_of_current_item}"):
                            move_frame("Up", timing_details[i].uuid)
                            st.experimental_rerun()
                    with btn3:
                        if st.button("⬇️", key=f"Demote {index_of_current_item}"):
                            move_frame("Down", timing_details[i].uuid)
                            st.experimental_rerun()
            
            # Display radio buttons for pagination at the bottom
            st.markdown("***")

        # Update the current page in session state
        elif st.session_state['page'] == "Motion":
            num_timing_details = len(timing_details)
            shift1, shift2 = st.columns([2, 1.2])

            with shift2:
                shift_frames = st.checkbox(
                    "Shift Frames", help="This will shift the after your adjustment forward or backwards.")

            timing_details = data_repo.get_timing_list_from_project(project_uuid)       

            for idx in range(start_index, end_index):                      
                st.header(f"Frame {idx+1}")                        
                timing1, timing2, timing3 = st.columns([1, 1, 1])

                with timing1:
                    frame1, frame2,frame3 = st.columns([2,1,2])
                    with frame1:
                        if timing_details[idx].primary_image_location:
                            st.image(
                                timing_details[idx].primary_image_location)
                    with frame2:
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.info("     ➜")
                    with frame3:                                                
                        if idx+1 < num_timing_details and timing_details[idx+1].primary_image_location:
                            st.image(timing_details[idx+1].primary_image_location)
                        elif idx+1 == num_timing_details:
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")                            
                            st.markdown("<h1 style='text-align: center; color: black; font-family: Arial; font-size: 50px; font-weight: bold;'>FIN</h1>", unsafe_allow_html=True)

                    single_frame_time_selector(timing_details[idx].uuid, 'motion')
                    st.caption(f"Duration: {timing_details[idx].clip_duration:.2f} secs")

                    # calculate minimum and maximum values for slider
                    if idx == 0:
                        min_frame_time = 0.0  # make sure the value is a float
                    else:
                        min_frame_time = timing_details[idx].frame_time

                    if idx == num_timing_details - 1:
                        max_frame_time = timing_details[idx].frame_time + 10.0
                    elif idx < num_timing_details - 1:
                        max_frame_time = timing_details[idx+1].frame_time

                    # disable slider only if it's the first frame
                    slider_disabled = idx == 0
                    frame_time = st.slider(
                        f"#{idx+1} Frame Time = {round(timing_details[idx].frame_time, 3)}",
                        min_value=min_frame_time,
                        max_value=max_frame_time,
                        value=timing_details[idx].frame_time,
                        step=0.01,
                        disabled=slider_disabled,
                        key=f"frame_time_slider_{idx}"
                    )
                    
                    update_animation_style_element(timing_details[idx].uuid)

                # update timing details
                if timing_details[idx].frame_time != frame_time:
                        previous_frame_time = timing_details[idx].frame_time
                        data_repo.update_specific_timing(
                            timing_details[idx].uuid, frame_time=frame_time)
                        for a in range(i - 2, i + 2):
                            if a >= 0 and a < num_timing_details:
                                data_repo.update_specific_timing(timing_details[a].uuid, timing_video_id=None)
                        data_repo.update_specific_timing(timing_details[idx].uuid, preview_video_id=None)

                        if shift_frames is True:
                            diff_frame_time = frame_time - previous_frame_time
                            for j in range(i+1, num_timing_details+1):
                                new_frame_time = timing_details[j-1].frame_time + \
                                    diff_frame_time
                                data_repo.update_specific_timing(
                                    timing_details[j-1].uuid, frame_time=new_frame_time)
                                data_repo.update_specific_timing(
                                    timing_details[j-1].uuid, timed_clip_id=None)
                                data_repo.update_specific_timing(
                                    timing_details[j-1].uuid, preview_video_id=None)
                        st.experimental_rerun()

                with timing2:
                    current_individual_clip_element(timing_details[idx].uuid)
                
                with timing3:
                    current_preview_video_element(timing_details[idx].uuid)
                
    st.markdown("***")
        

    with st.expander("➕ Add Key Frame", expanded=True):

        add1, add2 = st.columns(2)

        selected_image = ""
        with add1:
            # removed "Frame From Video" for now
            image1,image2 = st.columns(2)
            with image1:
                source_of_starting_image = st.radio("Where would you like to get the starting image from?", [
                                                "Previous frame", "Uploaded image"], key="source_of_starting_image")
            if source_of_starting_image == "Previous frame":                
                with image2:
                    which_stage_for_starting_image = st.radio("Which stage would you like to use?", [
                                                          ImageStage.MAIN_VARIANT.value, ImageStage.SOURCE_IMAGE.value], key="which_stage_for_starting_image", horizontal=True)
                    which_number_for_starting_image = st.number_input("Which frame would you like to use?", min_value=1, max_value=
                                                                  max(1, len(timing_details)), value=st.session_state['current_frame_index'], step=1, key="which_number_for_starting_image")
                if which_stage_for_starting_image == ImageStage.SOURCE_IMAGE.value:
                    if timing_details[which_number_for_starting_image - 1].source_image != "":
                        selected_image = timing_details[which_number_for_starting_image - 1].source_image.location
                    else:
                        selected_image = ""
                elif which_stage_for_starting_image == ImageStage.MAIN_VARIANT.value:
                    selected_image = timing_details[which_number_for_starting_image - 1].primary_image_location
            elif source_of_starting_image == "Uploaded image":
                with image2:
                    uploaded_image = st.file_uploader(
                        "Upload an image", type=["png", "jpg", "jpeg"])
                    # FILE UPLOAD HANDLE--
                    if uploaded_image is not None:
                        # write uploaded_image to location videos/{project_name}/assets/frames/1_selected
                        image = Image.open(uploaded_image)
                        file_location = f"videos/{project_uuid}/assets/frames/1_selected/{uploaded_image.name}"
                        # with open(os.path.join(file_location), "wb") as f:
                        #     f.write(uploaded_image.getbuffer())
                        selected_image = save_or_host_file(image, file_location)
                        selected_image = selected_image or file_location
                    else:
                        selected_image = ""
                    which_number_for_starting_image = st.session_state['current_frame_index']

            
            how_long_after = st.slider(
                "How long after the current frame?", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
                        
            inherit_styling_settings = st_memory.radio("Inherit styling settings from the selected frame?", [
                                                       "Yes", "No"], key="inherit_styling_settings", horizontal=True, project_settings=project_settings)
            
            apply_zoom_effects = st_memory.radio("Apply zoom effects to inputted image?", [
                                                        "No","Yes"], key="apply_zoom_effects", horizontal=True, project_settings=project_settings)
            
            if apply_zoom_effects == "Yes":
                zoom_inputs(project_settings, position='new', horizontal=True)
                                            
        with add2:
            if selected_image:
                if apply_zoom_effects == "Yes":
                    selected_image = get_pillow_image(selected_image)
                    selected_image = apply_image_transformations(selected_image, st.session_state['zoom_level_input'], st.session_state['rotation_angle_input'], st.session_state['x_shift'], st.session_state['y_shift'])
                    project_update_data = {
                        "zoom_level": st.session_state['zoom_level_input'],
                        "rotation_angle_value": st.session_state['rotation_angle_input'],
                        "x_shift": st.session_state['x_shift'],
                        "y_shift": st.session_state['y_shift']
                    }
                    data_repo.update_project_setting(project_uuid, **project_update_data)

                st.info("Starting Image:")                
                st.image(selected_image)
            else:
                st.error("No Starting Image Found")

        if st.button(f"Add key frame",type="primary",use_container_width=True):
            
            def add_key_frame(selected_image, inherit_styling_settings, how_long_after):
                data_repo = DataRepo()
                project_uuid = st.session_state['project_uuid']
                timing_details = data_repo.get_timing_list_from_project(project_uuid)
                project_settings = data_repo.get_project_setting(project_uuid)
                

                if len(timing_details) == 0:
                    index_of_current_item = 1
                else:
                    index_of_current_item = min(len(timing_details), st.session_state['current_frame_index'])

                timing_details = data_repo.get_timing_list_from_project(project_uuid)

                if len(timing_details) == 0:
                    key_frame_time = 0.0
                elif index_of_current_item == len(timing_details):
                    key_frame_time = float(timing_details[index_of_current_item - 1].frame_time) + how_long_after
                else:
                    key_frame_time = (float(timing_details[index_of_current_item - 1].frame_time) + float(
                        timing_details[index_of_current_item].frame_time)) / 2.0

                if len(timing_details) == 0:
                    new_timing = create_timings_row_at_frame_number(project_uuid, 0)
                else:
                    new_timing = create_timings_row_at_frame_number(project_uuid, index_of_current_item, frame_time=key_frame_time)
                    
                    clip_duration = calculate_desired_duration_of_individual_clip(new_timing.uuid)
                    data_repo.update_specific_timing(new_timing.uuid, clip_duration=clip_duration)

                timing_details = data_repo.get_timing_list_from_project(project_uuid)
                if selected_image != "":
                    save_uploaded_image(selected_image, project_uuid, timing_details[index_of_current_item].uuid, "source")
                    save_uploaded_image(selected_image, project_uuid, timing_details[index_of_current_item].uuid, "styled")

                if inherit_styling_settings == "Yes":
                    clone_styling_settings(index_of_current_item - 1, timing_details[index_of_current_item].uuid)

                data_repo.update_specific_timing(timing_details[index_of_current_item].uuid, \
                                                    animation_style=project_settings.default_animation_style)

                if len(timing_details) == 1:
                    st.session_state['current_frame_index'] = 1
                    st.session_state['current_frame_uuid'] = timing_details[0].uuid
                else:
                    st.session_state['current_frame_index'] = min(len(timing_details), st.session_state['current_frame_index'] + 1)
                    st.session_state['current_frame_uuid'] = timing_details[st.session_state['current_frame_index'] - 1].uuid

                st.session_state['page'] = "Styling"
                st.session_state['section_index'] = 0
                st.experimental_rerun()
            add_key_frame(selected_image, inherit_styling_settings, how_long_after)
            st.experimental_rerun()

