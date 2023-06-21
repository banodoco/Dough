import streamlit as st
from streamlit_image_comparison import image_comparison
import time
import pandas as pd
import os
from PIL import Image
import requests as r
from streamlit_drawable_canvas import st_canvas
from shared.constants import GuidanceType, InternalFileType
from shared.file_upload.s3 import upload_file
from ui_components.common_methods import delete_frame, promote_image_variant, trigger_restyling_process, add_image_variant, create_timings_row_at_frame_number, extract_canny_lines, convert_to_minutes_and_seconds, styling_element, create_full_preview_video, back_and_forward_buttons, resize_and_rotate_element, move_frame, calculate_desired_duration_of_individual_clip, create_or_get_single_preview_video, calculate_desired_duration_of_individual_clip, single_frame_time_changer
from ui_components.common_methods import create_gif_preview, delete_frame, promote_image_variant, trigger_restyling_process, add_image_variant, prompt_interpolation_model, update_speed_of_video_clip, create_timings_row_at_frame_number, extract_canny_lines, get_duration_from_video, get_audio_bytes_for_slice, add_audio_to_video_slice, convert_to_minutes_and_seconds, styling_element, get_primary_variant_location, create_full_preview_video, back_and_forward_buttons, resize_and_rotate_element, manual_cropping_element, precision_cropping_element, move_frame, calculate_desired_duration_of_individual_clip, create_or_get_single_preview_video, calculate_desired_duration_of_individual_clip, single_frame_time_changer, apply_image_transformations, get_pillow_image, save_new_image, prompt_finder_element, preview_frame, carousal_of_images_element, display_image, ai_frame_editing_element, clone_styling_settings
from utils import st_memory
import uuid

import cv2
import uuid
import datetime
from pydub import AudioSegment
from io import BytesIO
import shutil
from streamlit_option_menu import option_menu
from moviepy.editor import concatenate_videoclips
import moviepy.editor
import math
from ui_components.constants import WorkflowStageType
from ui_components.models import InternalAppSettingObject, InternalFileObject, InternalFrameTimingObject
from streamlit_extras.annotated_text import annotated_text

from utils.data_repo.data_repo import DataRepo


# TODO: CORRECT-CODE
def frame_styling_page(mainheader2, project_uuid: str):
    data_repo = DataRepo()

    timing_details = data_repo.get_timing_list_from_project(project_uuid)

    if len(timing_details) == 0:
        if st.button("Create timings row"):
            create_timings_row_at_frame_number(project_uuid, 0)
            timing = data_repo.get_timing_from_frame_number(project_uuid, 0)
            data_repo.update_specific_timing(timing.uuid, frame_time=0.0)
            st.experimental_rerun()
    else:
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
            st.session_state['frame_styling_view_type'] = "List View"
            st.session_state['frame_styling_view_type_index'] = 0

            sections = ["Guidance", "Styling", "Motion"]

            # TODO: CORRECT-CODE
            view_types = ["Individual View", "List View"]

            if 'frame_styling_view_type_index' not in st.session_state:
                st.session_state['frame_styling_view_type_index'] = 0
                st.session_state['frame_styling_view_type'] = "Individual View"
                st.session_state['change_view_type'] = False

            if 'change_view_type' not in st.session_state:
                st.session_state['change_view_type'] = False

            if st.session_state['change_view_type'] == True:
                st.session_state['frame_styling_view_type_index'] = view_types.index(
                    st.session_state['frame_styling_view_type'])
            else:
                st.session_state['frame_styling_view_type_index'] = None

            def on_change_view_type(key):
                selection = st.session_state[key]
                if selection == "List View":
                    st.session_state['index_of_current_page'] = math.floor(
                        st.session_state['current_frame_uuid'] / 10)

            # Option menu
            st.session_state['frame_styling_view_type'] = option_menu(
                None,
                view_types,
                icons=['aspect-ratio', 'bookshelf', "hourglass", 'stopwatch'],
                menu_icon="cast",
                orientation="horizontal",
                key="section-selecto1r",
                styles={"nav-link": {"font-size": "15px", "margin": "0px", "--hover-color": "#eee"},
                        "nav-link-selected": {"background-color": "green"}},
                manual_select=st.session_state['frame_styling_view_type_index'],
                on_change=on_change_view_type
            )

            if st.session_state['change_view_type'] == True:
                st.session_state['change_view_type'] = False
                # round down st.session_state['current_frame_uuid']to nearest 10

            if st.session_state['frame_styling_view_type'] == "Individual View":

                if len(timing_details) > 1:
                    percentage = round(
                        (float(st.session_state['current_frame_index']) / float(len(timing_details)-1)) * 100.00)

                    st.progress(percentage)
                else:
                    st.progress(100)

                time1, time2 = st.columns([1, 1])

                with time1:

                    frame_number = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(
                        timing_details)-1, value=st.session_state['current_frame_index'], step=1, key="which_image_selector")
                    
                    st.session_state['current_frame_uuid'] = timing_details[frame_number].uuid
                    frame_index = next((i for i, t in enumerate(timing_details) if t.uuid == st.session_state['current_frame_uuid']), None)

                    if st.session_state['current_frame_index'] != frame_index:
                        st.session_state['current_frame_index'] = frame_index
                        st.session_state['reset_canvas'] = True
                        st.session_state['frame_styling_view_type_index'] = 0
                        st.session_state['frame_styling_view_type'] = "Individual View"

                        st.experimental_rerun()

                with time2:
                    single_frame_time_changer(st.session_state['current_frame_uuid'])

                with st.expander("Notes:"):

                    notes = st.text_area(
                        "Frame Notes:", value=timing_details[st.session_state['current_frame_index']].notes, height=100, key="notes")

                if notes != timing_details[st.session_state['current_frame_index']].notes:
                    data_repo.update_specific_timing(
                        st.session_state['current_frame_uuid'], notes=notes)
                    st.experimental_rerun()

                if st.session_state['page'] == "Guidance":
                    image_1_size = 2
                    image_2_size = 1.5
                elif st.session_state['page'] == "Styling":
                    image_1_size = 1.5
                    image_2_size = 2
                elif st.session_state['page'] == "Motion":
                    image_1_size = 1.5
                    image_2_size = 1.5

                image_1, image_2 = st.columns([image_1_size, image_2_size])
                with image_1:
                    st.caption(
                        f"Guidance Image for Frame #{st.session_state['current_frame_index']}:")
                    display_image(
                        timing_uuid=st.session_state['current_frame_uuid'], stage=WorkflowStageType.SOURCE.value, clickable=False)
                with image_2:
                    st.caption(
                        f"Main Styled Image for Frame #{st.session_state['current_frame_index']}:")
                    display_image(
                        timing_uuid=st.session_state['current_frame_uuid'], stage="Styled", clickable=False)
                st.markdown("***")

                if st.button("Delete key frame"):
                    delete_frame(st.session_state['current_frame_uuid'])
                    timing_details = data_repo.get_timing_list_from_project(
                        project_uuid)
                    st.experimental_rerun()

        if timing_details == []:
            st.info(
                "You need to select and load key frames first in the Key Frame Selection section.")

        else:

            if st.session_state['frame_styling_view_type'] == "List View":
                st.markdown(
                    f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{st.session_state['page']}]")

            else:
                st.markdown(
                    f"#### :red[{st.session_state['main_view_type']}] > **:green[{st.session_state['frame_styling_view_type']}]** > :orange[{st.session_state['page']}] > :blue[Frame #{st.session_state['current_frame_uuid']}]")

            project_settings = data_repo.get_project_setting(project_uuid)

            if st.session_state['frame_styling_view_type'] == "Individual View":
                if st.session_state['page'] == "Guidance":

                    carousal_of_images_element(project_uuid, stage=WorkflowStageType.SOURCE.value)

                    guidance_types = GuidanceType.value_list()
                    if 'how to guide_index' not in st.session_state:
                        if not project_settings.guidance_type:
                            st.session_state['how_to_guide_index'] = 0
                        else:
                            st.session_state['how_to_guide_index'] = guidance_types.index(
                                project_settings.guidance_type)
                    crop1, crop2 = st.columns([1, 1])
                    with crop1:
                        how_to_guide = st.radio("How to guide:", guidance_types, key="how_to_guide",
                                                horizontal=True, index=st.session_state['how_to_guide_index'])
                    if guidance_types.index(how_to_guide) != st.session_state['how_to_guide_index']:
                        st.session_state['how_to_guide_index'] = guidance_types.index(
                            how_to_guide)
                        data_repo.update_project_setting(
                            project_uuid, guidance_type=how_to_guide)
                        st.experimental_rerun()

                    if how_to_guide == GuidanceType.DRAWING.value:
                        canvas1, canvas2 = st.columns([1.25, 3])
                        timing = data_repo.get_timing_from_uuid(
                            st.session_state['current_frame_uuid'])

                        with canvas1:
                            width = int(project_settings.width)
                            height = int(project_settings.height)

                            if timing.source_image.location != "":
                                if timing.source_image.location.startswith("http"):
                                    canvas_image = r.get(
                                        timing.source_image.location)
                                    canvas_image = Image.open(
                                        BytesIO(canvas_image.content))
                                else:
                                    canvas_image = Image.open(
                                        timing.source_image.location)
                            else:
                                canvas_image = Image.new(
                                    "RGB", (width, height), "white")
                            if 'drawing_input' not in st.session_state:
                                st.session_state['drawing_input'] = 'Magic shapes 🪄'
                            col1, col2 = st.columns([6, 5])

                            with col1:
                                st.session_state['drawing_input'] = st.radio(
                                    "Drawing tool:",
                                    ("Draw lines ✏️", "Erase Lines ❌", "Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Make Lines ║", "Make squares □"), horizontal=True,
                                )

                                if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                                    drawing_mode = "transform"

                                elif st.session_state['drawing_input'] == "Make shapes 🪄":
                                    drawing_mode = "polygon"

                                elif st.session_state['drawing_input'] == "Draw lines ✏️":
                                    drawing_mode = "freedraw"

                                elif st.session_state['drawing_input'] == "Erase Lines ❌":
                                    drawing_mode = "freedraw"

                                elif st.session_state['drawing_input'] == "Make Lines ║":
                                    drawing_mode = "line"

                                elif st.session_state['drawing_input'] == "Make squares □":
                                    drawing_mode = "rect"

                            with col2:

                                stroke_width = st.slider(
                                    "Stroke width: ", 1, 100, 2)
                                if st.session_state['drawing_input'] == "Erase Lines ❌":
                                    stroke_colour = "#ffffff"
                                else:
                                    stroke_colour = st.color_picker(
                                        "Stroke color hex: ", value="#000000")
                                fill = st.checkbox("Fill shapes", value=False)
                                if fill == True:
                                    fill_color = st.color_picker(
                                        "Fill color hex: ")
                                else:
                                    fill_color = ""

                            if st.button("Clear Canny Image"):
                                data_repo.remove_source_image(
                                    st.session_state['current_frame_uuid'])
                                st.session_state['reset_canvas'] = True
                                st.experimental_rerun()

                            st.markdown("***")
                            back_and_forward_buttons()
                            st.markdown("***")

                            resize_and_rotate_element(
                                WorkflowStageType.SOURCE.value, project_uuid)

                        with canvas2:
                            realtime_update = True

                            if "reset_canvas" not in st.session_state:
                                st.session_state['reset_canvas'] = False

                            if st.session_state['reset_canvas'] != True:

                                canvas_result = st_canvas(
                                    fill_color=fill_color,
                                    stroke_width=stroke_width,
                                    stroke_color=stroke_colour,
                                    background_color="rgb(255, 255, 255)",
                                    background_image=canvas_image,
                                    update_streamlit=realtime_update,
                                    height=height,
                                    width=width,
                                    drawing_mode=drawing_mode,
                                    display_toolbar=True,
                                    key="full_app",
                                )

                                if 'image_created' not in st.session_state:
                                    st.session_state['image_created'] = 'no'

                                if canvas_result.image_data is not None:
                                    img_data = canvas_result.image_data
                                    im = Image.fromarray(
                                        img_data.astype("uint8"), mode="RGBA")
                            else:
                                st.session_state['reset_canvas'] = False
                                canvas_result = st_canvas()
                                time.sleep(0.1)
                                st.experimental_rerun()

                        if st.button("Save New Canny Image"):
                            if canvas_result.image_data is not None:
                                # overlay the canvas image on top of the canny image and save the result
                                # if canny image is from a url, then we need to download it first
                                if timing.source_image and timing.source_image.location:
                                    if timing.source_image.location.startswith("http"):
                                        canny_image = r.get(
                                            timing.source_image.location)
                                        canny_image = Image.open(
                                            BytesIO(canny_image.content))
                                    else:
                                        canny_image = Image.open(
                                            timing.source_image.location)
                                else:
                                    canny_image = Image.new(
                                        "RGB", (width, height), "white")

                                canny_image = canny_image.convert("RGBA")
                                # canvas_image = canvas_image.convert("RGBA")
                                canvas_image = im
                                canvas_image = canvas_image.convert("RGBA")

                                # converting the images to the same size and mode
                                if canny_image.size != canvas_image.size:
                                    canny_image = canny_image.resize(
                                        canvas_image.size)

                                if canny_image.mode != canvas_image.mode:
                                    canny_image = canny_image.convert(
                                        canvas_image.mode)

                                new_canny_image = Image.alpha_composite(
                                    canny_image, canvas_image)
                                if new_canny_image.mode != "RGB":
                                    new_canny_image = new_canny_image.convert(
                                        "RGB")

                                unique_file_name = str(uuid.uuid4()) + ".png"
                                file_location = f"videos/{timing.project.uuid}/assets/resources/masks/{unique_file_name}"
                                new_canny_image.save(file_location)
                                file_data = {
                                    "name": str(uuid.uuid4()) + ".png",
                                    "type": InternalFileType.IMAGE.value,
                                    "local_path": file_location
                                }

                                canny_image = data_repo.create_file(
                                    **file_data)
                                data_repo.update_specific_timing(
                                    st.session_state['current_frame_uuid'], source_image_id=canny_image.uuid)
                                st.success("New Canny Image Saved")
                                st.session_state['reset_canvas'] = True
                                time.sleep(1)
                                st.experimental_rerun()

                        st.markdown("***")
                        canny1, canny2, canny3 = st.columns([1, 1, 1.1])
                        with canny1:
                            st.markdown(
                                "#### Use Canny Image From Other Frame")
                            st.markdown(
                                "This will use a canny image from another frame. This will take a few seconds.")

                            if st.session_state['current_frame_index'] == 0:
                                value = 0
                            else:
                                value = st.session_state['current_frame_uuid'] - 1
                            canny_frame_number = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(
                                timing_details)-1, value=value, step=1, key="canny_frame_number")
                            canny_timing = timing_details[canny_frame_number]

                            if st.button("Use Guidance Image From Other Frame"):
                                if timing_details[canny_frame_number]["source_image"] != "":
                                    data_repo.update_specific_timing(
                                        st.session_state['current_frame_uuid'], source_image_id=timing_details[canny_frame_number].source_image.uuid)
                                    st.experimental_rerun()

                            if canny_timing.source_image.location != "":
                                st.image(canny_timing.source_image.location)
                            else:
                                st.error("No Guidance Image Found")
                        with canny2:
                            st.markdown("#### Upload Guidance Image")
                            st.markdown(
                                "This will upload a canny image from your computer. This will take a few seconds.")
                            uploaded_file = st.file_uploader("Choose a file")
                            if st.button("Upload Guidance Image"):
                                with open(os.path.join(f"videos/{timing.project.uuid}/assets/resources/masks", uploaded_file.name), "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                    st.success("Your file is uploaded")
                                    file_data = {
                                        "name": str(uuid.uuid4()) + ".png",
                                        "type": InternalFileType.IMAGE.value,
                                        "local_path": f"videos/{timing.project.uuid}/assets/resources/masks/{uploaded_file.name}"
                                    }
                                    image = data_repo.create_file(**file_data)
                                    data_repo.update_specific_timing(
                                        st.session_state['current_frame_uuid'], source_image_id=image.uuid)
                                    time.sleep(1.5)
                                    st.experimental_rerun()
                        with canny3:
                            st.markdown("#### Extract Canny From image")
                            st.markdown(
                                "This will extract a canny image from the current image. This will take a few seconds.")
                            source_of_image = st.radio("Which image would you like to use?", [
                                                       "Existing Frame", "Uploaded Image"])
                            if source_of_image == "Existing Frame":
                                which_frame = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(
                                    timing_details)-1, value=st.session_state['current_frame_index'], step=1)

                                existing_frame_timing = data_repo.get_timing_from_frame_number(
                                    project_uuid, which_frame)
                                if existing_frame_timing.primary_image_location:
                                    image_path = existing_frame_timing.primary_image_location
                                    st.image(image_path)
                                else:
                                    st.error("No Image Found")

                            elif source_of_image == "Uploaded Image":
                                uploaded_image = st.file_uploader(
                                    "Choose a file", key="uploaded_image")
                                if uploaded_image is not None:
                                    # download image as temp.png
                                    with open("temp.png", "wb") as f:
                                        f.write(uploaded_image.getbuffer())
                                        st.success("Your file is uploaded")
                                        uploaded_image = "videos/temp/assets/videos/0_raw/" + \
                                            str(uuid.uuid4()) + ".png"

                            threshold1, threshold2 = st.columns([1, 1])
                            with threshold1:
                                low_threshold = st.number_input(
                                    "Low Threshold", min_value=0, max_value=255, value=100, step=1)
                            with threshold2:
                                high_threshold = st.number_input(
                                    "High Threshold", min_value=0, max_value=255, value=200, step=1)

                            if 'canny_image' not in st.session_state:
                                st.session_state['canny_image'] = None

                            if st.button("Extract Canny From image"):
                                if source_of_image == "Existing Frame":
                                    canny_image = extract_canny_lines(
                                        image_path, project_uuid, low_threshold, high_threshold)
                                    st.session_state['canny_image'] = canny_image.uuid
                                elif source_of_image == "Uploaded Image":
                                    canny_image = extract_canny_lines(
                                        uploaded_image, project_uuid, low_threshold, high_threshold)
                                    st.session_state['canny_image'] = canny_image.uuid

                            if st.session_state['canny_image']:
                                canny_image = data_repo.get_file_from_uuid(st.session_state['canny_image'])
                                st.image(canny_image.location)
                                canny_action_1, canny_action_2 = st.columns([1, 1])

                                with canny_action_1:
                                    if st.button("Save Canny Image"):
                                        data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=st.session_state['canny_image'])
                                        st.session_state['reset_canvas'] = True
                                        st.session_state['canny_image'] = None
                                        st.experimental_rerun()
                                with canny_action_2:
                                    if st.button("Clear New Canny Image"):
                                        st.session_state['canny_image'] = None
                                        st.experimental_rerun()

                # if current item is 0
                    elif how_to_guide == "Images":
                        with crop2:
                            how_to_crop = st_memory.radio("How to crop:", options=[
                                                          "Manual Cropping", "Precision Cropping"], project_uuid=project_uuid, key="how_to_crop")

                        if how_to_crop == "Manual Cropping":

                            manual_cropping_element(
                                WorkflowStageType.SOURCE.value, st.session_state['current_frame_uuid'])

                        elif how_to_crop == "Precision Cropping":

                            precision_cropping_element(WorkflowStageType.SOURCE.value, project_uuid)

                        with st.expander("Replace Source Image", expanded=True):

                            canny1, canny2 = st.columns([1, 1])

                            with canny1:
                                st.markdown("#### Upload Source Image")
                                st.markdown(
                                    "This will upload a canny image from your computer. This will take a few seconds.")
                                uploaded_file = st.file_uploader(
                                    "Choose a file")
                                if st.button("Upload Source Image"):
                                    base_location = f"videos/{timing.project.uuid}/assets/resources/masks"
                                    if not os.path.exists(base_location):
                                        os.makedirs(base_location)

                                    with open(os.path.join(base_location, uploaded_file.name), "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                        st.success("Your file is uploaded")
                                        file_data = {
                                            "name": str(uuid.uuid4()) + ".png",
                                            "type": InternalFileType.IMAGE.value,
                                            "local_path": f"videos/{timing.project.uuid}/assets/resources/masks/{uploaded_file.name}"
                                        }
                                        source_image = data_repo.create_file(
                                            **file_data)
                                        data_repo.update_specific_timing(
                                            st.session_state['current_frame_uuid'], source_image_id=source_image.uuid)
                                        time.sleep(1.5)
                                        st.experimental_rerun()

                            with canny2:
                                st.markdown("#### Use Image From Other Frame")
                                st.markdown(
                                    "This will use a canny image from another frame. This will take a few seconds.")

                                if st.session_state['current_frame_uuid'] == 0:
                                    value = 0
                                else:
                                    value = st.session_state['current_frame_uuid'] - 1
                                which_stage = st.radio("Which stage would you like to use?", [
                                                       "Styled Image", "Source Image"])
                                which_number_image = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(
                                    timing_details)-1, value=value, step=1, key="canny_frame_number")

                                current_timing = data_repo.get_timing_from_frame_number(project_uuid,
                                    which_number_image)
                                if which_stage == "Source Image":
                                    if current_timing.source_image_location != "":
                                        selected_image = current_timing.source_image_location
                                        st.image(selected_image)
                                    else:
                                        st.error("No Source Image Found")
                                elif which_stage == "Styled Image":
                                    selected_image = current_timing.primary_image_location
                                    if selected_image != "":
                                        st.image(selected_image)
                                    else:
                                        st.error("No Image Found")

                                if st.button("Use Selected Image"):
                                    file_data = {
                                        "name": str(uuid.uuid4()) + ".png",
                                        "type": InternalFileType.IMAGE.value,
                                        "hosted_url": current_timing.primary_image_location
                                    }
                                    source_image = data_repo.create_file(
                                        **file_data)
                                    data_repo.update_specific_timing(
                                        st.session_state['current_frame_uuid'], source_image_id=source_image.uuid)
                                    st.experimental_rerun()

                        # with st.expander("Inpainting, Background Removal & More", expanded=False):
                        with st.expander("Inpainting, Background Removal & More"):
                            ai_frame_editing_element(st.session_state['current_frame_uuid'], WorkflowStageType.SOURCE.value)

                elif st.session_state['page'] == "Motion":

                    timing1, timing2 = st.columns([1, 1])

                    with timing1:
                        num_timing_details = len(timing_details)

                        shift1, shift2 = st.columns([2, 1.2])

                        with shift2:
                            shift_frames = st.checkbox(
                                "Shift Frames", help="This will shift the after your adjustment forward or backwards.")

                        for i in range(max(0, st.session_state['current_frame_index'] - 2), min(num_timing_details, st.session_state['current_frame_index'] + 3)):
                            # calculate minimum and maximum values for slider
                            if i == 0:
                                min_frame_time = 0.0  # make sure the value is a float
                            else:
                                min_frame_time = timing_details[i].frame_time

                            if i == num_timing_details - 1:
                                max_frame_time = timing_details[i].frame_time + 10.0
                            elif i < num_timing_details - 1:
                                max_frame_time = timing_details[i+1].frame_time

                            # disable slider only if it's the first frame
                            slider_disabled = i == 0

                            frame1, frame2, frame3 = st.columns([1, 1, 2])

                            with frame1:
                                if timing_details[i].primary_image_location:
                                    st.image(
                                        timing_details[i].primary_image_location)
                            with frame2:
                                if st.session_state['page'] != "Motion":
                                    single_frame_time_changer(timing_details[i].uuid)
                                st.caption(
                                    f"Duration: {calculate_desired_duration_of_individual_clip(timing_details, i):.2f} secs")

                            with frame3:
                                frame_time = st.slider(
                                    f"#{i} Frame Time = {round(timing_details[i].frame_time, 3)}",
                                    min_value=min_frame_time,
                                    max_value=max_frame_time,
                                    value=timing_details[i].frame_time,
                                    step=0.01,
                                    disabled=slider_disabled,
                                )

                            # update timing details
                            if timing_details[i]['frame_time'] != frame_time:
                                previous_frame_time = timing_details[i]['frame_time']
                                data_repo.update_specific_timing(
                                    timing_details[i].uuid, frame_time=frame_time)
                                for a in range(i - 1, i + 2):
                                    if a >= 0 and a < num_timing_details:
                                        data_repo.update_specific_timing(timing_details[a].uuid, timing_video_id=None)
                                data_repo.update_specific_timing(timing_details[i], preview_video_id=None)

                                if shift_frames is True:
                                    diff_frame_time = frame_time - previous_frame_time
                                    for j in range(i+1, num_timing_details):
                                        new_frame_time = timing_details[j].frame_time + \
                                            diff_frame_time
                                        data_repo.update_specific_timing(
                                            timing_details[j].uuid, frame_time=new_frame_time)
                                        data_repo.update_specific_timing(
                                            timing_details[j].uuid, timed_clip_id=None)
                                        data_repo.update_specific_timing(
                                            timing_details[j].uuid, preview_video_id=None)
                                st.experimental_rerun()

                    with timing2:
                        timing = data_repo.get_timing_from_uuid(
                            st.session_state['current_frame_uuid'])
                        variants = timing.alternative_images_list
                        if timing.preview_video:
                            st.video(timing.preview_video.location)
                        else:
                            st.error(
                                "No preview video available for this frame")
                        preview_settings_1, preview_settings_2 = st.columns([
                                                                            2, 1])
                        with preview_settings_1:
                            speed = st.slider(
                                "Preview Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.01)

                        with preview_settings_2:
                            st.write(" ")
                            if variants != [] and variants != None and variants != "":
                                if st.button("Generate New Preview Video"):
                                    preview_video = create_full_preview_video(
                                        st.session_state['current_frame_uuid'], speed)
                                    data_repo.update_specific_timing(
                                        st.session_state['current_frame_uuid'], preview_video_id=preview_video.uuid)
                                    st.experimental_rerun()

                        back_and_forward_buttons()

                    with st.expander("Animation style"):
                        animation1, animation2 = st.columns([1.5, 1])

                        with animation1:
                            project_settings = data_repo.get_project_setting(
                                project_uuid)

                            animation_styles = [
                                "Interpolation", "Direct Morphing"]

                            if 'index_of_animation_style' not in st.session_state:
                                st.session_state['index_of_animation_style'] = animation_styles.index(
                                    project_settings.default_animation_style)

                            animation_style = st.radio("Which animation style would you like to use for this frame?",
                                                       animation_styles, index=st.session_state['index_of_animation_style'])

                            if timing_details[st.session_state['current_frame_index']].animation_style == "":
                                data_repo.update_specific_timing(
                                    st.session_state['current_frame_uuid'], animation_style=project_settings.default_animation_style)
                                st.session_state['index_of_animation_style'] = animation_styles.index(
                                    project_settings.default_animation_style)

                                st.experimental_rerun()

                            if animation_styles.index(timing_details[st.session_state['current_frame_index']].animation_style) != st.session_state['index_of_animation_style']:
                                st.session_state['index_of_animation_style'] = animation_styles.index(
                                    timing_details[st.session_state['current_frame_index']].animation_style)
                                st.experimental_rerun()

                            animationbutton1, animationbutton2 = st.columns([
                                                                            1, 1])

                            with animationbutton1:

                                if animation_style != timing_details[st.session_state['current_frame_index']].animation_style:

                                    if st.button("Update this slides animation style"):
                                        data_repo.update_specific_timing(
                                            st.session_state['current_frame_uuid'], animation_style=animation_style)
                                        st.success("Animation style updated")
                                        data_repo.update_specific_timing(
                                            st.session_state['current_frame_uuid'], interpolated_video=None)

                                        if project_settings.default_animation_style == "":
                                            data_repo.update_project_setting(
                                                project_uuid, default_animation_style=animation_style)
                                        time.sleep(0.3)
                                        st.experimental_rerun()
                                else:
                                    st.info(
                                        f"{animation_style} is already the animation style for this frame.")

                            with animationbutton2:
                                if animation_style != project_settings.default_animation_style:
                                    if st.button(f"Change default animation style to {animation_style}", help="This will change the default animation style - but won't affect current frames."):
                                        data_repo.update_project_setting(
                                            project_uuid, default_animation_style=animation_style)

                        with animation2:

                            if animation_style == "Interpolation":
                                st.info(
                                    "This will fill the gaps between the current frame and the next frame with interpolated frames. This will make the animation smoother but will take longer to render.")
                            elif animation_style == "Direct Morphing":
                                st.info(
                                    "This will morph the current frame directly into the next frame. This will make the animation less smooth but can be used to nice effect.")

                    with st.expander("Clip speed adjustment"):

                        clip_data = []
                        start_pct = 0.0
                        total_duration = 0.0
                        st.subheader("Speed Adjustment")

                        while start_pct < 1.0:
                            st.info(f"##### Section {len(clip_data) + 1}")
                            end_pct = st.slider(
                                f"What percentage of the original clip should section {len(clip_data) + 1} go until?", min_value=start_pct, max_value=1.0, value=1.0, step=0.01)

                            if end_pct == 1.0:
                                remaining_duration = 1.0 - total_duration
                                remaining_pct = 1.0 - start_pct
                                speed_change = remaining_pct / remaining_duration
                                st.write(
                                    f"Speed change for the last section will be set to **{speed_change:.2f}x** to maintain the original video length.")
                            else:
                                speed_change = st.slider(
                                    f"What speed change should be applied to section {len(clip_data) + 1}?", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

                            clip_data.append({
                                "start_pct": start_pct,
                                "end_pct": end_pct,
                                "speed_change": speed_change
                            })

                            original_duration = end_pct - start_pct
                            final_duration = original_duration / \
                                (speed_change + 1e-6)
                            total_duration += final_duration

                            if speed_change > 1:
                                st.info(f"This will make the section from **{start_pct * 100:.0f}% to {end_pct * 100:.0f}%** of the video "
                                        f"**{speed_change:.2f}x** faster, so it lasts **{convert_to_minutes_and_seconds(final_duration)}**.")
                            else:
                                st.info(f"This will make the section from **{start_pct * 100:.0f}% to {end_pct * 100:.0f}%** of the video "
                                        f"**{1 / speed_change:.2f}x** slower, so it lasts **{convert_to_minutes_and_seconds(final_duration)}**.")

                            # Update the start_pct for the next section
                            start_pct = float(end_pct)

                            st.markdown("***")
                        st.write(clip_data)

                elif st.session_state['page'] == "Styling":

                    carousal_of_images_element(timing_details, stage="Styled")

                    comparison_values = [
                        "Other Variants", "Source Frame", "Previous & Next Frame", "None"]

                    st.session_state['show_comparison'] = st_memory.radio(
                        "Show comparison to:", options=comparison_values, horizontal=True, project_uuid=project_uuid, key="show_comparison_radio")

                    variants = timing_details[timing.aux_frame_index].alternative_images_list

                    if variants != [] and variants != None and variants != "":

                        primary_variant_location = timing_details[st.session_state['current_frame_index']].primary_image_location

                    if st.session_state['show_comparison'] == "Other Variants":

                        mainimages1, mainimages2 = st.columns([1, 1])

                        aboveimage1, aboveimage2, aboveimage3 = st.columns(
                            [1, 0.25, 0.75])

                        with aboveimage1:
                            st.info(
                                f"Current variant = {timing_details[st.session_state['current_frame_uuid']]['primary_image']}")

                        with aboveimage2:
                            show_more_than_10_variants = st.checkbox(
                                "Show >10 variants", key="show_more_than_10_variants")

                        with aboveimage3:

                            number_of_variants = len(variants)

                            if show_more_than_10_variants is True:
                                current_variant = int(
                                    timing_details[st.session_state['current_frame_uuid']]["primary_image"])
                                which_variant = st.radio(f'Main variant = {current_variant}', range(
                                    number_of_variants), index=number_of_variants-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_uuid']}")
                            else:

                                last_ten_variants = range(
                                    max(0, number_of_variants - 10), number_of_variants)
                                current_variant = int(
                                    timing_details[st.session_state['current_frame_uuid']]["primary_image"])
                                which_variant = st.radio(f'Main variant = {current_variant}', last_ten_variants, index=len(
                                    last_ten_variants)-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_uuid']}")

                        with mainimages1:

                            project_settings = data_repo.get_project_setting(project_uuid)
                            st.success("Main variant")
                            if timing_details[st.session_state['current_frame_uuid']]["alternative_images"] != "":
                                st.image(primary_variant_location,
                                         use_column_width=True)
                            else:
                                st.error("No variants found for this frame")

                        with mainimages2:

                            if timing_details[st.session_state['current_frame_uuid']]["alternative_images"] != "":

                                if which_variant == current_variant:
                                    st.success("Main variant")

                                else:
                                    st.info(f"Variant #{which_variant}")

                                st.image(variants[which_variant],
                                         use_column_width=True)

                                if which_variant != current_variant:

                                    if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['current_frame_uuid']}", help="Promote this variant to the primary image"):
                                        promote_image_variant(
                                            st.session_state['current_frame_uuid'], which_variant)
                                        time.sleep(0.5)
                                        st.experimental_rerun()

                    elif st.session_state['show_comparison'] == "Source Frame":
                        if timing_details[st.session_state['current_frame_uuid']]["alternative_images"] != "":
                            img2 = primary_variant_location
                        else:
                            img2 = 'https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'
                        image_comparison(starting_position=50,
                                         img1=timing_details[st.session_state['current_frame_uuid']
                                                             ]["source_image"],
                                         img2=img2, make_responsive=False, label1=WorkflowStageType.SOURCE.value, label2=WorkflowStageType.STYLED.value)
                    elif st.session_state['show_comparison'] == "Previous & Next Frame":

                        mainimages1, mainimages2, mainimages3 = st.columns([
                                                                           1, 1, 1])

                        with mainimages1:
                            if st.session_state['current_frame_index']-1 >= 0:
                                previous_image = data_repo.get_timing_from_frame_number(st.session_state['current_frame_index'] - 1)
                                st.info(f"Previous image")
                                display_image(
                                    timing_uuid=previous_image.uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

                                if st.button(f"Preview Interpolation From #{st.session_state['current_frame_index']-1} to #{st.session_state['current_frame_index']}", key=f"Preview Interpolation From #{st.session_state['current_frame_uuid']-1} to #{st.session_state['current_frame_uuid']}", use_container_width=True):
                                    prev_frame_timing = data_repo.get_prev_timing(st.session_state['current_frame_uuid'])
                                    create_or_get_single_preview_video(prev_frame_timing.uuid)
                                    prev_frame_timing = data_repo.get_timing_from_uuid(prev_frame_timing.uuid)
                                    st.video(prev_frame_timing.timed_clip.location)

                        with mainimages2:
                            st.success(f"Current image")
                            display_image(
                                timing_uuid=st.session_state['current_frame_uuid'], stage=WorkflowStageType.STYLED.value, clickable=False)

                        with mainimages3:
                            if st.session_state['current_frame_index']+1 < len(timing_details):
                                next_image = data_repo.get_timing_from_frame_number(st.session_state['current_frame_index'] + 1)
                                st.info(f"Next image")
                                display_image(timing_uuid=next_image.uuid, stage=WorkflowStageType.STYLED.value, clickable=False)

                                if st.button(f"Preview Interpolation From #{st.session_state['current_frame_index']} to #{st.session_state['current_frame_index']+1}", key=f"Preview Interpolation From #{st.session_state['current_frame_uuid']} to #{st.session_state['current_frame_uuid']+1}", use_container_width=True):
                                    create_or_get_single_preview_video(
                                        st.session_state['current_frame_uuid'])
                                    current_frame = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                                    st.video(current_frame.timed_clip.location)

                    elif st.session_state['show_comparison'] == "None":
                        display_image(
                            idx=st.session_state['current_frame_uuid'], stage="Styled", clickable=False, timing_details=timing_details)

                    st.markdown("***")

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

                                if st.button(f"Generate variants", key=f"new_variations_{st.session_state['current_frame_index']}", help="This will generate new variants based on the settings to the left."):
                                    for i in range(0, st.session_state['individual_number_of_variants']):
                                        trigger_restyling_process(st.session_state['current_frame_uuid'], st.session_state['model'], st.session_state['prompt'], st.session_state['strength'], st.session_state['custom_pipeline'], st.session_state['negative_prompt'], st.session_state['guidance_scale'], st.session_state['seed'], st.session_state[
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
                                    which_frame_to_copy_from = st.number_input("Which frame would you like to copy styling settings from?", min_value=0, max_value=len(
                                        timing_details)-1, value=st.session_state['current_frame_index']-1, step=1)
                                    if st.button("Copy styling settings from this frame"):
                                        clone_styling_settings(which_frame_to_copy_from, st.session_state['current_frame_uuid'])
                                        st.experimental_rerun()

                                with copy2:
                                    display_image(
                                        idx=which_frame_to_copy_from, stage="Styled", clickable=False, timing_details=timing_details)
                                    st.caption("Prompt:")
                                    st.caption(
                                        timing_details[which_frame_to_copy_from]["prompt"])
                                    st.caption("Model:")
                                    st.caption(
                                        timing_details[which_frame_to_copy_from]["model_id"])

                    with st.expander("Crop, Move & Rotate Image", expanded=False):
                        precision_cropping_element(WorkflowStageType.STYLED.value, project_uuid)

                    with st.expander("Inpainting, Background Removal & More", expanded=False):
                        ai_frame_editing_element(st.session_state['current_frame_uuid'], WorkflowStageType.STYLED.value)

                    with st.expander("Prompt Finder"):
                        prompt_finder_element(project_uuid)

                    with st.expander("Replace Frame"):

                        replace_with = st.radio("Replace with:", [
                                                "Uploaded Frame", "Previous Frame"], horizontal=True, key="replace_with_what")
                        replace1, replace2, replace3 = st.columns([2, 1, 1])

                        if replace_with == "Previous Frame":
                            with replace1:
                                which_stage_to_use_for_replacement = st.radio("Select stage to use:", [
                                                                              "Styled Key Frame", "Unedited Key Frame"], key="which_stage_to_use_for_replacement", horizontal=True)
                                which_image_to_use_for_replacement = st.number_input("Select image to use:", min_value=0, max_value=len(
                                    timing_details)-1, value=0, key="which_image_to_use_for_replacement")
                                if which_stage_to_use_for_replacement == "Unedited Key Frame":
                                    selected_image = timing_details[which_image_to_use_for_replacement].source_image
                                elif which_stage_to_use_for_replacement == "Styled Key Frame":
                                    selected_image = timing_details[which_image_to_use_for_replacement].primary_image
                                if st.button("Replace with selected frame", disabled=False):
                                    number_of_image_variants = add_image_variant(
                                        selected_image.uuid, st.session_state['current_frame_uuid'])
                                    promote_image_variant(
                                        st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                                    st.success("Replaced")
                                    time.sleep(1)
                                    st.experimental_rerun()
                            with replace2:
                                st.image(selected_image.location, width=300)

                        elif replace_with == "Uploaded Frame":
                            with replace1:
                                replacement_frame = st.file_uploader("Upload a replacement frame here", type=[
                                                                     "png", "jpeg"], accept_multiple_files=False, key="replacement_frame_upload")
                            with replace2:
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                if st.button("Replace frame", disabled=False):
                                    images_for_model = []
                                    with open(os.path.join(f"videos/{timing.project.uuid}/", replacement_frame.name), "wb") as f:
                                        f.write(replacement_frame.getbuffer())

                                    app_setting: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()
                                    uploaded_image_url = upload_file(
                                        f"videos/{timing.project.uuid}/{replacement_frame.name}",  app_setting.aws_access_key, app_setting.aws_secret_access_key)

                                    file_data = {
                                        "name": str(uuid.uuid4()) + ".png",
                                        "type": InternalFileType.IMAGE.value,
                                        "hosted_url": uploaded_image_url
                                    }
                                    replacement_image = data_repo.create_file(
                                        **file_data)

                                    number_of_image_variants = add_image_variant(
                                        replacement_image.uuid, st.session_state['current_frame_uuid'])
                                    promote_image_variant(
                                        st.session_state['current_frame_uuid'], number_of_image_variants - 1)

                                    # delete the uploaded file
                                    os.remove(
                                        f"videos/{timing.project.uuid}/{replacement_frame.name}")
                                    st.success("Replaced")
                                    time.sleep(1)
                                    st.experimental_rerun()

                st.markdown("***")
                extra_settings_1, extra_settings_2 = st.columns([1, 1])

                with extra_settings_2:

                    st.write("")

            elif st.session_state['frame_styling_view_type'] == "List View":
                if 'current_page' not in st.session_state:
                    st.session_state['current_page'] = 1
                    st.session_state['index_of_current_page'] = 1

                # Calculate number of pages
                items_per_page = 10
                num_pages = math.ceil(len(timing_details) / items_per_page)

                # Display radio buttons for pagination at the top
                st.markdown("---")

                st.session_state['current_page'] = st.radio("Select Page:", options=range(
                    0, num_pages+1), horizontal=True, index=st.session_state['index_of_current_page'], key="page_selection_radio")

                if st.session_state['current_page'] != st.session_state['index_of_current_page']:
                    st.session_state['index_of_current_page'] = st.session_state['current_page']
                    st.experimental_rerun()

                st.markdown("---")

                # Update the current page in session state

                # Display items for the current page only
                start_index = st.session_state['current_page'] * items_per_page
                end_index = min(start_index + items_per_page,
                                len(timing_details))

                for i in range(start_index, end_index):
                    index_of_current_item = i

                    timing_details = data_repo.get_timing_list_from_project(
                        project_uuid)
                    st.subheader(f"Frame {i}")

                    if st.session_state['page'] == "Styling":
                        image1_size = 1
                        image2_size = 2
                        image3_size = 1
                    elif st.session_state['page'] == "Guidance":
                        image1_size = 2
                        image2_size = 1
                        image3_size = 1
                    elif st.session_state['page'] == "Motion":
                        image1_size = 1
                        image2_size = 1
                        image3_size = 2

                    image1, image2, image3 = st.columns(
                        [image1_size, image2_size, image3_size])

                    with image1:

                        display_image(
                            idx=i, stage="Source", clickable=False, timing_details=timing_details)

                    with image2:
                        display_image(
                            idx=i, stage="Styled", clickable=False, timing_details=timing_details)

                    with image3:
                        time1, time2 = st.columns([1, 1])
                        with time1:

                            single_frame_time_changer(timing_details[i].uuid)

                            st.info(
                                f"Duration: {calculate_desired_duration_of_individual_clip(timing_details[i].uuid):.2f} secs")

                        with time2:
                            animation_styles = [
                                "Interpolation", "Direct Morphing"]

                            animation_styles = [
                                "Interpolation", "Direct Morphing"]

                            if f"animation_style_index_{index_of_current_item}" not in st.session_state:
                                st.session_state[f"animation_style_index_{index_of_current_item}"] = animation_styles.index(
                                    timing_details[index_of_current_item]['animation_style'])
                                st.session_state[f"animation_style_{index_of_current_item}"] = timing_details[
                                    index_of_current_item]['animation_style']

                            st.session_state[f"animation_style_{index_of_current_item}"] = st.radio(
                                "Animation style:", animation_styles, index=st.session_state[f"animation_style_index_{index_of_current_item}"], key=f"animation_style_radio_{i}", help="This is for the morph from the current frame to the next one.")

                            if st.session_state[f"animation_style_{index_of_current_item}"] != timing_details[index_of_current_item]["animation_style"]:
                                st.session_state[f"animation_style_index_{index_of_current_item}"] = animation_styles.index(
                                    st.session_state[f"animation_style_{index_of_current_item}"])
                                data_repo.update_specific_timing(timing_details[index_of_current_item].uuid, animation_style=st.session_state[f"animation_style_{index_of_current_item}"])
                                st.experimental_rerun()

                        if st.button(f"Jump to single frame view for #{index_of_current_item}"):
                            st.session_state['current_frame_index'] = index_of_current_item
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

            if st.session_state['page'] == "Styling":

                with st.sidebar:

                    if st.session_state['frame_styling_view_type'] == "List View":
                        styling_element(st.session_state['current_frame_uuid'], view_type="List")

    with st.expander("Add Key Frame", expanded=True):

        add1, add2 = st.columns(2)

        with add1:
            source_of_starting_image = st.radio("Where would you like to get the starting image from?", [
                                                "Previous frame", "Uploaded image", "Frame From Video"], key="source_of_starting_image")
            if source_of_starting_image == "Previous frame":
                which_stage_for_starting_image = st.radio("Which stage would you like to use?", [
                                                          "Styled Image", "Source Image"], key="which_stage_for_starting_image")
                which_number_for_starting_image = st.number_input("Which frame would you like to use?", min_value=0, max_value=
                                                                  max(0, len(timing_details)-1), value=st.session_state['current_frame_index'], step=1, key="which_number_for_starting_image")
                if which_stage_for_starting_image == "Source Image":
                    if timing_details[which_number_for_starting_image]["source_image"] != "":
                        selected_image = timing_details[which_number_for_starting_image]["source_image"]
                    else:
                        selected_image = ""
                elif which_stage_for_starting_image == "Styled Image":
                    selected_image = timing_details[which_number_for_starting_image].primary_image_location
            elif source_of_starting_image == "Uploaded image":
                uploaded_image = st.file_uploader(
                    "Upload an image", type=["png", "jpg", "jpeg"])
                if uploaded_image is not None:
                    # write uploaded_image to location videos/{project_name}/assets/frames/1_selected
                    file_location = f"videos/{project_uuid}/assets/frames/1_selected/{uploaded_image.name}"
                    with open(os.path.join(file_location), "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    selected_image = file_location
                else:
                    selected_image = ""
                which_number_for_starting_image = st.session_state['current_frame_uuid']

            elif source_of_starting_image == "Frame From Video":
                which_video = st.number_input("Which video would you like to use?", min_value=0, max_value=len(
                    timing_details)-1, value=st.session_state['current_frame_uuid'], step=1, key="which_number_for_starting_image")
                input_video = timing_details[which_video]["interpolated_video"]
                if input_video != "":
                    cap = cv2.VideoCapture(input_video)
                    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    which_frame = st.slider("Which frame would you like to use?", min_value=0,
                                            max_value=number_of_frames, value=0, step=1, key="which_frame_for_starting_image")
                    input_video = timing_details[which_video]["interpolated_video"]

                    selected_image = preview_frame(
                        project_uuid, input_video, which_frame)
                else:
                    st.error("No video found")
                which_number_for_starting_image = st.session_state['current_frame_uuid']

            how_long_after = st.slider(
                "How long after?", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            if project_settings.zoom_level != "":
                apply_current_image_transformations = st_memory.radio("Apply the last zoom, rotation & movement to new frame?", [
                                                                      "Yes", "No"], key="apply_zoom", horizontal=True, project_uuid=project_uuid)
            else:
                apply_current_image_transformations = "No"
                project_settings.zoom_level = 0

            also_make_this_the_primary_image = st_memory.radio("Also make this the primary image?", [
                                                               "Yes", "No"], key="also_make_this_the_primary_image", horizontal=True, project_uuid=project_uuid)

            inherit_styling_settings = st_memory.radio("Inherit styling settings from the selected frame?", [
                                                       "Yes", "No"], key="inherit_styling_settings", horizontal=True, project_uuid=project_uuid)

        with add2:
            if selected_image != "":
                if apply_current_image_transformations == "Yes":
                    selected_image = get_pillow_image(selected_image)
                    selected_image = apply_image_transformations(selected_image, project_settings.zoom_level, 
                        project_settings.rotation_angle_input_value, project_settings.x_shift, project_settings.y_shift)

                st.image(selected_image)
            else:
                st.error("No Image Found")

        if st.button(f"Add key frame"):
            if len(timing_details) == 0:
                index_of_current_item = 0
            else:
                index_of_current_item = st.session_state['current_frame_uuid']

            timing_details = data_repo.get_timing_list_from_project(project_uuid)

            if len(timing_details) == 0:
                key_frame_time = 0.0
            elif index_of_current_item == len(timing_details) - 1:
                key_frame_time = float(
                    timing_details[index_of_current_item].frame_time) + how_long_after
            else:
                st.write(timing_details[index_of_current_item].frame_time)
                st.write(timing_details[index_of_current_item + 1].frame_time)
                key_frame_time = (float(timing_details[index_of_current_item].frame_time) + float(
                    timing_details[index_of_current_item + 1].frame_time)) / 2.0

            if len(timing_details) == 0:
                new_timing = create_timings_row_at_frame_number(
                    project_uuid, 0)
                data_repo.update_specific_timing(
                    new_timing.uuid, frame_time=0.0)
            else:
                create_timings_row_at_frame_number(
                    project_uuid, index_of_current_item + 1)
                data_repo.update_specific_timing(
                    project_uuid, frame_time=key_frame_time)
            
            timing_details = data_repo.get_timing_list_from_project(project_uuid)
            if selected_image != "":
                selected_image = save_new_image(selected_image)
                file_data = {
                    "name": str(uuid.uuid4()) + ".png",
                    "type": InternalFileType.IMAGE.value,
                    "local_path": selected_image,
                    "project_id": project_uuid
                }

                new_source_image = data_repo.create_file(**file_data)
                data_repo.update_specific_timing(
                    timing_details[index_of_current_item + 1].uuid, source_image_id=new_source_image.uuid)
                
            if also_make_this_the_primary_image == "Yes":
                add_image_variant(
                    selected_image, timing_details[index_of_current_item + 1].uuid)
                promote_image_variant(timing_details[index_of_current_item + 1].uuid, 0)
            if inherit_styling_settings == "Yes":
                clone_styling_settings(which_number_for_starting_image, timing_details[index_of_current_item + 1].uuid)

            data_repo.update_specific_timing(timing_details[index_of_current_item + 1].uuid, \
                                             animation_style=project_settings.default_animation_style)
            
            if len(timing_details) == 1:
                st.session_state['current_frame_index'] = 0
            else:
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 1

            st.session_state['page'] = "Guidance"
            st.session_state['section_index'] = 0
            st.experimental_rerun()
