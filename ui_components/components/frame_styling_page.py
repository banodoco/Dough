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
from ui_components.common_methods import delete_frame, promote_image_variant, trigger_restyling_process, add_image_variant, create_timings_row_at_frame_number, extract_canny_lines, convert_to_minutes_and_seconds, styling_element, create_full_preview_video, back_and_forward_buttons, resize_and_rotate_element, crop_image_element, move_frame, calculate_desired_duration_of_individual_clip, create_or_get_single_preview_video, calculate_desired_duration_of_individual_clip, single_frame_time_changer
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

from utils.data_repo.data_repo import DataRepo


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
            st.session_state['which_stage_to_run_on'] = project_settings.default_stage
            st.session_state['show_comparison'] = "Don't show"

        if "current_frame_uuid" not in st.session_state:
            timing = data_repo.get_timing_list_from_project(project_uuid)[0]
            st.session_state['current_frame_uuid'] = timing.uuid

        if 'frame_styling_view_type' not in st.session_state:
            st.session_state['frame_styling_view_type'] = "List View"
            st.session_state['frame_styling_view_type_index'] = 0


        with st.sidebar:
            frame_number = st.number_input(f"Key frame # (out of {len(timing_details) - 1})", 0, len(
                timing_details) - 1, value=st.session_state['current_frame_index'], step=1, key="which_image_selector")
            st.session_state['current_frame_uuid'] = timing_details[frame_number].uuid
            
            frame_index = next((i for i, t in enumerate(timing_details) if t.uuid == st.session_state['current_frame_uuid']), None)
            if st.session_state['current_frame_index'] != frame_index:
                st.session_state['current_frame_index'] = frame_index
                st.session_state['reset_canvas'] = True
                st.session_state['frame_styling_view_type_index'] = 0
                st.session_state['frame_styling_view_type'] = "Individual View"
                st.experimental_rerun()

            timing = data_repo.get_timing_from_uuid(
                st.session_state['current_frame_uuid'])
            
            with st.expander("Notes:"):
                notes = st.text_area(
                    "Frame Notes:", value=timing.notes, height=100, key="notes")

            if notes != timing.notes:
                timing.notes = notes
                data_repo.update_specific_timing(timing.uuid, notes=notes)
                st.experimental_rerun()
            st.markdown("***")

        if timing_details == []:
            st.info(
                "You need to select and load key frames first in the Key Frame Selection section.")

        else:
            top1, top2, top3 = st.columns([4, 1, 3])
            with top1:
                view_types = ["Individual View", "List View"]
                st.session_state['frame_styling_view_type'] = st.radio(
                    "View type:", view_types, key="which_view_type", horizontal=True, index=st.session_state['frame_styling_view_type_index'])
                if view_types.index(st.session_state['frame_styling_view_type']) != st.session_state['frame_styling_view_type_index']:
                    st.session_state['frame_styling_view_type_index'] = view_types.index(
                        st.session_state['frame_styling_view_type'])
                    st.experimental_rerun()

            with top2:
                st.write("")

            project_settings = data_repo.get_project_setting(project_uuid)

            if st.session_state['frame_styling_view_type'] == "Individual View":

                if "section_index" not in st.session_state:
                    st.session_state['section_index'] = 0

                sections = ["Guidance", "Styling", "Motion"]

                st.session_state['section'] = option_menu(None, sections, icons=[
                                                          'pencil', 'palette', "hourglass", 'stopwatch'], menu_icon="cast", default_index=st.session_state['section_index'], orientation="horizontal")

                if st.session_state['section_index'] != sections.index(st.session_state['section']):
                    st.session_state['section_index'] = sections.index(
                        st.session_state['section'])
                    st.experimental_rerun()

                if st.session_state['section'] == "Guidance":

                    guidance_types = GuidanceType.value_list()
                    if 'how to guide_index' not in st.session_state:
                        if not project_settings.guidance_type:
                            st.session_state['how_to_guide_index'] = 0
                        else:
                            st.session_state['how_to_guide_index'] = guidance_types.index(
                                project_settings.guidance_type)
                            
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
                            col1, col2 = st.columns([6, 3])

                            with col1:
                                st.session_state['drawing_input'] = st.radio(
                                    "Drawing tool:",
                                    ("Draw lines ✏️", "Erase Lines ❌", "Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Make Lines ║"), horizontal=True,
                                )

                                if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                                    drawing_mode = "transform"
                                    stroke_colour = "rgba(0, 0, 0)"
                                elif st.session_state['drawing_input'] == "Make shapes 🪄":
                                    drawing_mode = "polygon"
                                    stroke_colour = "rgba(0, 0, 0)"
                                elif st.session_state['drawing_input'] == "Draw lines ✏️":
                                    drawing_mode = "freedraw"
                                    stroke_colour = "rgba(0, 0, 0)"
                                elif st.session_state['drawing_input'] == "Erase Lines ❌":
                                    drawing_mode = "freedraw"
                                    stroke_colour = "rgba(255, 255, 255)"
                                elif st.session_state['drawing_input'] == "Make Lines ║":
                                    drawing_mode = "line"
                                    stroke_colour = "rgba(0, 0, 0)"

                            with col2:
                                if st.session_state['drawing_input'] == "Draw lines ✏️" or st.session_state['drawing_input'] == "Make Lines ║":
                                    stroke_width = st.slider(
                                        "Stroke width: ", 1, 50, 2)
                                elif st.session_state['drawing_input'] == "Erase Lines ❌":
                                    stroke_width = st.slider(
                                        "Stroke width: ", 1, 100, 25)
                                else:
                                    stroke_width = 3

                            if st.button("Clear Canny Image"):
                                data_repo.remove_source_image(
                                    st.session_state['current_frame_uuid'])
                                st.session_state['reset_canvas'] = True
                                st.experimental_rerun()

                            st.markdown("***")
                            back_and_forward_buttons()
                            st.markdown("***")

                            resize_and_rotate_element(WorkflowStageType.SOURCE.value, project_uuid)

                        with canvas2:
                            realtime_update = True

                            if "reset_canvas" not in st.session_state:
                                st.session_state['reset_canvas'] = False

                            if st.session_state['reset_canvas'] != True:

                                canvas_result = st_canvas(
                                    fill_color="rgba(0, 0, 0)",
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
                                        canny_image = r.get(timing.source_image.location)
                                        canny_image = Image.open(BytesIO(canny_image.content))
                                    else:
                                        canny_image = Image.open(timing.source_image.location)
                                else:
                                    canny_image = Image.new("RGB", (width, height), "white")

                                canny_image = canny_image.convert("RGBA")
                                # canvas_image = canvas_image.convert("RGBA")
                                canvas_image = im
                                canvas_image = canvas_image.convert("RGBA")

                                # converting the images to the same size and mode
                                if canny_image.size != canvas_image.size:
                                    canny_image = canny_image.resize(canvas_image.size)

                                if canny_image.mode != canvas_image.mode:
                                    canny_image = canny_image.convert(canvas_image.mode)

                                new_canny_image = Image.alpha_composite(
                                    canny_image, canvas_image)
                                if new_canny_image.mode != "RGB":
                                    new_canny_image = new_canny_image.convert("RGB")

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
                                value = st.session_state['current_frame_index'] - 1

                            which_number_image_for_canny = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(
                                timing_details)-1, value=value, step=1, key="which_number_image_for_canny")
                            canny_timing = data_repo.get_timing_from_frame_number(project_uuid, which_number_image_for_canny)
                            if st.button("Use Source Image From Other Frame"):
                                if canny_timing.source_image.location != "":
                                    file_data = {
                                        "name": str(uuid.uuid4()) + ".png",
                                        "type": InternalFileType.IMAGE.value,
                                        "local_path": canny_timing.source_image.uuid
                                    }
                                    canny_file: InternalFileObject = data_repo.create_file(**file_data)
                                    data_repo.update_specific_timing(timing_details[st.session_state['current_frame_index']].uuid, source_image_id=canny_file.uuid)
                                    st.experimental_rerun()

                            if canny_timing.source_image.location != "":
                                st.image(canny_timing.source_image.location)
                            else:
                                st.error("No Source Image Found")

                        with canny2:
                            st.markdown("#### Upload Source Image")
                            st.markdown(
                                "This will upload a canny image from your computer. This will take a few seconds.")
                            uploaded_file = st.file_uploader("Choose a file")
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
                                    image = data_repo.create_file(**file_data)
                                    data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=image.uuid)
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
                                
                                existing_frame_timing = data_repo.get_timing_from_frame_number(project_uuid, which_frame)
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
                                        uploaded_image = "videos/temp/assets/videos/0_raw/" + str(uuid.uuid4()) +".png"

                            threshold1, threshold2 = st.columns([1, 1])
                            with threshold1:
                                low_threshold = st.number_input(
                                    "Low Threshold", min_value=0, max_value=255, value=100, step=1)
                            with threshold2:
                                high_threshold = st.number_input(
                                    "High Threshold", min_value=0, max_value=255, value=200, step=1)

                            if st.button("Extract Canny From image"):
                                if source_of_image == "Existing Frame":
                                    canny_image = extract_canny_lines(image_path, timing.project.uuid, low_threshold, high_threshold)
                                elif source_of_image == "Uploaded Image":
                                    canny_image = extract_canny_lines(uploaded_image, timing.project.uuid, low_threshold, high_threshold)
                                
                                st.session_state['canny_image'] = canny_image.uuid

                            if 'canny_image' in st.session_state and st.session_state['canny_image']:
                                canny_image = data_repo.get_file_from_uuid(st.session_state['canny_image'])
                                st.image(canny_image.location)
                                if st.button("Save Canny Image"):
                                    data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=canny_image.uuid)
                                    st.session_state['reset_canvas'] = True
                                    st.session_state['canny_image'] = None
                                    st.experimental_rerun()

                # if current item is 0
                    elif how_to_guide == "Images":
                        crop_image_element("Source")
                        with st.expander("Replace Source Image", expanded=False):

                            canny1, canny2 = st.columns([1, 1])

                            with canny1:
                                st.markdown("#### Upload Source Image")
                                st.markdown("This will upload a canny image from your computer. This will take a few seconds.")
                                uploaded_file = st.file_uploader("Choose a file")
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
                                        source_image = data_repo.create_file(**file_data)
                                        data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=source_image.uuid)
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
                                    timing_details)-1, value=value, step=1, key="which_number_image_for_canny")
                                
                                current_timing = data_repo.get_timing_from_frame_number(
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
                                    source_image = data_repo.create_file(**file_data)
                                    data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=source_image.uuid)
                                    st.experimental_rerun()

                elif st.session_state['section'] == "Motion":

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
                                    st.image(timing_details[i].primary_image_location)
                            with frame2:
                                single_frame_time_changer(timing_details[i].uuid)
                                st.caption(
                                    f"Duration: {calculate_desired_duration_of_individual_clip(timing_details[i].uuid):.2f} secs")

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
                            if timing_details[i].frame_time != frame_time:
                                previous_frame_time = timing_details[i].frame_time
                                data_repo.update_specific_timing(timing_details[i].uuid, frame_time=frame_time)
                                for a in range(st.session_state['current_frame_index'] - 1, st.session_state['current_frame_index'] + 1):
                                    data_repo.update_specific_timing(timing_details[a].uuid, timed_clip_id=None)
                                data_repo.update_specific_timing(timing_details[i].uuid, preview_video_id=None)
                                if shift_frames is True:
                                    diff_frame_time = frame_time - previous_frame_time
                                    for j in range(i+1, num_timing_details):
                                        new_frame_time = timing_details[j].frame_time + \
                                            diff_frame_time
                                        data_repo.update_specific_timing(timing_details[j].uuid, frame_time=new_frame_time)
                                        data_repo.update_specific_timing(timing_details[j].uuid, timed_clip_id=None)
                                        data_repo.update_specific_timing(timing_details[j].uuid, preview_video_id=None)
                                st.experimental_rerun()

                    with timing2:
                        timing = data_repo.get_timing_from_uuid(st.session_state['current_frame_uuid'])
                        variants = timing.alternative_images_list
                        if timing.preview_video:
                            st.video(timing.preview_video.location)
                        else:
                            st.error("No preview video available for this frame")
                        preview_settings_1, preview_settings_2 = st.columns([2, 1])
                        with preview_settings_1:
                            speed = st.slider(
                                "Preview Speed", min_value=0.1, max_value=2.0, value=1.0, step=0.01)

                        with preview_settings_2:
                            st.write(" ")
                            if variants != [] and variants != None and variants != "":
                                if st.button("Generate New Preview Video"):
                                    preview_video = create_full_preview_video(st.session_state['current_frame_uuid'], speed)
                                    data_repo.update_specific_timing(st.session_state['current_frame_uuid'], preview_video_id=preview_video.uuid)
                                    st.experimental_rerun()

                        back_and_forward_buttons()

                    with st.expander("Animation style"):
                        animation1, animation2 = st.columns([1.5, 1])

                        with animation1:
                            project_settings = data_repo.get_project_setting(project_uuid)

                            animation_styles = ["Interpolation", "Direct Morphing"]

                            if 'index_of_animation_style' not in st.session_state:
                                st.session_state['index_of_animation_style'] = animation_styles.index(
                                    project_settings.default_animation_style)

                            animation_style = st.radio("Which animation style would you like to use for this frame?",
                                                       animation_styles, index=st.session_state['index_of_animation_style'])

                            if timing_details[st.session_state['current_frame_index']].animation_style == "":
                                data_repo.update_specific_timing(st.session_state['current_frame_uuid'], animation_style=project_settings.default_animation_style)
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
                                        data_repo.update_specific_timing(st.session_state['current_frame_uuid'], animation_style=animation_style)
                                        st.success("Animation style updated")
                                        data_repo.update_specific_timing(st.session_state['current_frame_uuid'], interpolated_video=None)

                                        if project_settings.default_animation_style == "":
                                            data_repo.update_project_setting(project_uuid, default_animation_style=animation_style)
                                        time.sleep(0.3)
                                        st.experimental_rerun()
                                else:
                                    st.info(
                                        f"{animation_style} is already the animation style for this frame.")

                            with animationbutton2:
                                if animation_style != project_settings.default_animation_style:
                                    if st.button(f"Change default animation style to {animation_style}", help="This will change the default animation style - but won't affect current frames."):
                                        data_repo.update_project_setting(project_uuid, default_animation_style=animation_style)

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

                elif st.session_state['section'] == "Styling":

                    with top3:
                        comparison_values = ["None"]
                        if timing.aux_frame_index != 0 and timing.source_image.location:
                            comparison_values.append("Source Frame")
                        
                        prev_timing = data_repo.get_prev_timing(timing.uuid)
                        if prev_timing and prev_timing.alternative_images:
                            comparison_values.append("Previous Frame")
                            
                        if len(timing_details) > timing.aux_frame_index+1:
                            if timing_details[timing.aux_frame_index+1].alternative_images != "":
                                comparison_values.append("Next Frame")

                        st.session_state['show_comparison'] = st.radio(
                            "Show comparison to:", options=comparison_values, horizontal=True)

                    mainimages1, mainimages2 = st.columns([1.5, 1])

                    variants = timing_details[timing.aux_frame_index].alternative_images_list

                    primary_image_location = timing.primary_image_location if timing.primary_image_location else None

                    with mainimages1:
                        if st.session_state['show_comparison'] == "None":
                            project_settings = data_repo.get_project_setting(project_uuid)
                            if primary_image_location:
                                st.image(primary_image_location, use_column_width=True)
                            else:
                                st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True)
                        elif st.session_state['show_comparison'] == "Source Frame":
                            if primary_image_location:
                                img2 = primary_image_location
                            else:
                                img2 = 'https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'
                            image_comparison(starting_position=50,
                                             img1=timing.source_image.location,
                                             img2=img2, make_responsive=False)
                        elif st.session_state['show_comparison'] == "Previous Frame":
                            previous_image = data_repo.get_prev_timing(timing.uuid)
                            image_comparison(starting_position=50,
                                             img1=previous_image.source_image.location,
                                             img2=primary_image_location, make_responsive=False)
                        elif st.session_state['show_comparison'] == "Next Frame":
                            next_image = data_repo.get_next_timing(timing.uuid)
                            image_comparison(starting_position=50,
                                             img1=primary_image_location,
                                             img2=next_image.source_image.location, make_responsive=False)

                        elif st.session_state['show_comparison'] == "Previous Frame":
                            st.write("")

                        detail1, detail2, detail3, detail4 = st.columns(
                            [2.5, 2.5, 3.5, 2])

                        with detail1:
                            st.session_state['individual_number_of_variants'] = st.number_input(
                                f"How many variants?", min_value=1, max_value=10, key=f"number_of_variants_{st.session_state['current_frame_uuid']}")

                        with detail2:
                            st.write("")
                            st.write("")

                            if st.button(f"Generate variants", key=f"new_variations_{timing.aux_frame_index}", help="This will generate new variants based on the settings to the left."):
                                for i in range(0, st.session_state['individual_number_of_variants']):
                                    index_of_current_item = timing.aux_frame_index
                                    trigger_restyling_process(
                                        timing.uuid, 
                                        st.session_state['model'], 
                                        st.session_state['prompt'], 
                                        st.session_state['strength'], 
                                        st.session_state['custom_pipeline'], 
                                        st.session_state['negative_prompt'], 
                                        st.session_state['guidance_scale'], 
                                        st.session_state['seed'], 
                                        st.session_state['num_inference_steps'], 
                                        st.session_state['which_stage_to_run_on'], 
                                        st.session_state["promote_new_generation"],  
                                        st.session_state['custom_models'], 
                                        st.session_state['adapter_type'], 
                                        True, 
                                        st.session_state['low_threshold'], 
                                        st.session_state['high_threshold']
                                    )
                                st.experimental_rerun()
                        with detail3:
                            st.write("")
                            st.write("")
                            if st.button(f"Re-run w/ saved settings", key=f"re_run_on_this_frame_{timing.aux_frame_index}", help="This will re-run the restyling process on this frame."):
                                index_of_current_item = timing.aux_frame_index
                                trigger_restyling_process(
                                    timing.uuid, 
                                    st.session_state['model'], 
                                    st.session_state['prompt'], 
                                    st.session_state['strength'], 
                                    st.session_state['custom_pipeline'], 
                                    st.session_state['negative_prompt'], 
                                    st.session_state['guidance_scale'], 
                                    st.session_state['seed'], 
                                    st.session_state['num_inference_steps'], 
                                    st.session_state['which_stage_to_run_on'], 
                                    st.session_state["promote_new_generation"],
                                    st.session_state['custom_models'], 
                                    st.session_state['adapter_type'], 
                                    False, 
                                    st.session_state['low_threshold'], 
                                    st.session_state['high_threshold']
                                )
                                st.experimental_rerun()
                        with detail4:
                            st.write("")

                            st.write("")

                    with mainimages2:
                        if timing.alternative_images:
                            variants = timing.alternative_images_list
                            number_of_variants = len(variants)
                            back_and_forward_buttons()
                            current_variant_uuid = timing.primary_image.uuid
                            which_variant = st.radio(f'Main variant = {timing.primary_variant_index}', range(
                                number_of_variants), index=number_of_variants-1, horizontal=True, key=f"Main variant for {st.session_state['current_frame_uuid']}")
                            st.image(variants[which_variant].location,
                                     use_column_width=True)

                            if which_variant == timing.primary_variant_index:
                                st.write("")
                                st.success("Main variant")
                                st.write("")
                            else:
                                st.write("")
                                if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['current_frame_uuid']}", help="Promote this variant to the primary image"):
                                    promote_image_variant(
                                        st.session_state['current_frame_uuid'], which_variant)
                                    time.sleep(0.5)
                                    st.experimental_rerun()

                    with st.expander("Compare to previous and next images", expanded=True):
                        
                        # TODO: unused function, remove this
                        def preview_individual_video(index_of_current_item, timing_details, project_name):
                            image = data_repo.get_primary_image_location(timing_details[index_of_current_item].uuid)

                            if image != "":
                                st.image(image, use_column_width=True,
                                         caption=f"Image #{index_of_current_item}")
                            if st.button(f"Preview Interpolation From #{index_of_current_item} to #{index_of_current_item+1}", key=f"Preview Interpolation From #{index_of_current_item} to #{index_of_current_item+1}"):
                                create_or_get_single_preview_video(
                                    index_of_current_item, project_name)
                                timing_details = data_repo.get_timing_list_from_project(project_uuid)
                                st.video(timing_details[index_of_current_item]['timed_clip'])

                        img1, img2 = st.columns(2)
                        with img1:
                            # if it's the first image, don't show a previous image
                            if timing.aux_frame_index != 0:
                                prev_frame_timing = data_repo.get_prev_timing(timing.uuid)
                                if prev_frame_timing and prev_frame_timing.primary_image:
                                    st.image(prev_frame_timing.primary_image.location, use_column_width=True,
                                             caption=f"Image #{prev_frame_timing.aux_frame_index}")
                                if st.button(f"Preview Interpolation From #{prev_frame_timing.aux_frame_index} to #{st.session_state['current_frame_index']}", key=f"Preview Interpolation From #{prev_frame_timing.aux_frame_index} to #{st.session_state['current_frame_uuid']}"):
                                    create_or_get_single_preview_video(prev_frame_timing.uuid)
                                    prev_frame_timing = data_repo.get_prev_timing(timing.uuid)
                                    st.video(prev_frame_timing.timed_clip.location)
                            else:
                                st.write("")

                        with img2:
                            # if it's the last image, don't show a next image
                            if timing.aux_frame_index != len(timing_details)-1:
                                next_image_number = st.session_state['current_frame_index']+1
                                next_frame_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(project_uuid, next_image_number)
                                if next_frame_timing and next_frame_timing.primary_image:
                                    st.image(next_frame_timing.primary_image.location, use_column_width=True,
                                             caption=f"Image #{next_image_number}")
                                if st.button(f"Preview Interpolation From #{timing.aux_frame_index} to #{next_image_number}", key=f"Preview Interpolation From #{timing.aux_frame_index} to #{next_image_number}"):
                                    create_or_get_single_preview_video(next_frame_timing.uuid)
                                    next_frame_timing: InternalFrameTimingObject = data_repo.get_timing_from_frame_number(project_uuid, next_image_number)
                                    st.video(next_frame_timing.timed_clip.location)
                            else:
                                st.write("")

                    with st.expander("🛠️ Saved Settings"):
                        st.info(
                            "These are the settings that will be used when you click 'Re-run w/ saved settings'.")
                        updated_prompt = st.text_area(
                            "Prompt:", value=timing_details[st.session_state['current_frame_index']].prompt, height=100)
                        if st.button("Save prompt"):
                            timing_details[st.session_state['current_frame_uuid']
                                           ]["prompt"] = updated_prompt
                            data_repo.update_specific_timing(st.session_state['current_frame_uuid'], prompt=updated_prompt)
                            st.experimental_rerun()

                    resize_and_rotate_element(WorkflowStageType.STYLED.value, project_uuid)

                    with st.expander("Replace Frame"):
                        replace_with = st.radio(
                            "Replace with:", ["Uploaded Frame", "Previous Frame"], horizontal=True)
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
                                    number_of_image_variants = add_image_variant(selected_image.uuid, st.session_state['current_frame_uuid'])
                                    promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                                    st.success("Replaced")
                                    time.sleep(1)
                                    st.experimental_rerun()
                            with replace2:
                                st.image(selected_image.location, width=300)

                        elif replace_with == "Uploaded Frame":
                            with replace1:
                                replacement_frame = st.file_uploader(
                                    "Upload a replacement frame here", type="png", accept_multiple_files=False, key="replacement_frame")
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
                                    replacement_image = data_repo.create_file(**file_data)

                                    number_of_image_variants = add_image_variant(replacement_image.uuid, st.session_state['current_frame_uuid'])
                                    promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)

                                    # delete the uploaded file
                                    os.remove(f"videos/{timing.project.uuid}/{replacement_frame.name}")
                                    st.success("Replaced")
                                    time.sleep(1)
                                    st.experimental_rerun()

                st.markdown("***")
                extra_settings_1, extra_settings_2 = st.columns([1, 1])

                with extra_settings_2:
                    with st.expander("Delete Key Frame", expanded=True):
                        if st.button("Delete key frame"):
                            index_of_current_item = timing.aux_frame_index
                            delete_frame(timing.uuid)
                            st.experimental_rerun()

            elif st.session_state['frame_styling_view_type'] == "List View":
                if 'current_page' not in st.session_state:
                    st.session_state['current_page'] = 0

                # Calculate number of pages
                items_per_page = 20
                num_pages = math.ceil(len(timing_details) / items_per_page)

                # Display radio buttons for pagination at the top
                st.markdown("---")
                page_selection = st.radio("Select Page", options=range(
                    1, num_pages+1), horizontal=True)

                st.markdown("---")

                # Update the current page in session state
                st.session_state['current_page'] = page_selection - 1

                # Display items for the current page only
                start_index = st.session_state['current_page'] * items_per_page
                end_index = min(start_index + items_per_page,
                                len(timing_details))

                for i in range(start_index, end_index):
                    index_of_current_item = i
                    
                    timing_details = data_repo.get_timing_list_from_project(project_uuid)
                    st.subheader(f"Frame {i}")

                    image1, image2, image3 = st.columns([1, 1, 1])

                    with image1:
                        source_image = timing_details[i].source_image
                        if source_image and source_image.location:
                            st.image(source_image.location, use_column_width=True,
                                     caption=f"Source image")

                    with image2:
                        primary_image = timing.primary_image_location
                        if primary_image:
                            st.image(primary_image, use_column_width=True,
                                     caption=f"Styled image")

                    with image3:
                        time1, time2 = st.columns([1, 1])
                        with time1:

                            single_frame_time_changer(timing_details[i].uuid)

                            st.info(
                                f"Duration: {calculate_desired_duration_of_individual_clip(timing_details[i].uuid):.2f} secs")

                        with time2:
                            animation_styles = ["Interpolation", "Direct Morphing"]

                            animation_style = st.radio("Animation style:", animation_styles, index=animation_styles.index(
                                timing_details[i].animation_style), key=f"animation_style_{i}", help="This is for the morph from the current frame to the next one.")

                            if timing_details[i].animation_style != animation_style:
                                data_repo.update_specific_timing(timing_details[i].uuid, animation_style=animation_style)
                                st.experimental_rerun()

                        if st.button(f"Jump to single frame view for #{index_of_current_item}"):
                            st.session_state['current_frame_index'] = index_of_current_item
                            st.session_state['frame_styling_view_type'] = "Individual View"
                            st.session_state['frame_styling_view_type_index'] = 0
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

                st.markdown("---")

                # Update the current page in session state

            with st.sidebar:
                element = st.radio("Select element:", [
                                   "Styling", "Timeline"], index=0, key="element", horizontal=True)
                if element == "Styling":
                    styling_element(timing.uuid)
                elif element == "Timeline":
                    stage = st.radio("Select stage:", [
                                     "Extracted Key Frames", "Current Main Variants"], index=0, key="stage", horizontal=True)

                    number_to_show = st.slider(
                        "Number of frames to show:", min_value=1, max_value=20, value=3, key="number_to_show")
                    for i in range(st.session_state['current_frame_uuid'] - number_to_show, st.session_state['current_frame_uuid'] + (number_to_show + 1)):
                        if i >= 0 and i < len(timing_details):
                            if i == st.session_state['current_frame_uuid']:
                                st.info(f"Frame {i}")
                            else:
                                st.write(f"Frame {i}")
                            if stage == "Current Main Variants":
                                st.image(timing_details[i].primary_image_location, use_column_width=True)
                            elif stage == "Extracted Key Frames":
                                st.image(
                                    timing_details[i].source_image, use_column_width=True)
                            st.markdown("---")

    with st.expander("Add Key Frame", expanded=True):
        if len(timing_details) == 0:
            st.info("The time on this will automatically be set to 0.0")
            selected_image = ""

        # if it's the last frame, ask how long after the previous frame to add the new frame
        else:
            add1, add2 = st.columns(2)

            with add1:
                source_of_starting_image = st.radio("Where would you like to get the starting image from?", [
                                                    "Previous frame", "Uploaded image"], key="source_of_starting_image")
                if source_of_starting_image == "Previous frame":
                    which_stage_for_starting_image = st.radio("Which stage would you like to use?", [
                                                              "Styled Image", "Source Image"], key="which_stage_for_starting_image")
                    which_number_for_starting_image = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(
                        timing_details)-1, value=st.session_state['current_frame_index'], step=1, key="which_number_for_starting_image")
                    if which_stage_for_starting_image == "Source Image":
                        if timing_details[which_number_for_starting_image].source_image.location != "":
                            selected_image = timing_details[which_number_for_starting_image].source_image.location
                        else:
                            selected_image = ""
                    elif which_stage_for_starting_image == "Styled Image":
                        selected_image = timing_details[which_number_for_starting_image].primary_image_location
                elif source_of_starting_image == "Uploaded image":
                    uploaded_image = st.file_uploader(
                        "Upload an image", type=["png", "jpg", "jpeg"])
                    if uploaded_image is not None:
                        # write uploaded_image to location videos/{project_name}/assets/frames/1_selected
                        file_location = f"videos/{timing.project.uuid}/assets/frames/1_selected/{uploaded_image.name}"
                        with open(os.path.join(file_location), "wb") as f:
                            f.write(uploaded_image.getbuffer())
                        selected_image = file_location
                    else:
                        selected_image = ""

                how_long_after = st.slider(
                    "How long after?", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

            with add2:
                if selected_image != "":
                    st.image(selected_image)
                else:
                    st.error("No Image Found")

        if st.button(f"Add key frame"):
            if len(timing_details) == 0:
                index_of_current_item = 0
            else:
                index_of_current_item = st.session_state['current_frame_index']
            st.info(
                f"This will add a key frame after frame #{index_of_current_item}")
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
                new_timing = create_timings_row_at_frame_number(project_uuid, 0)
                data_repo.update_specific_timing(new_timing.uuid, frame_time=0.0)
            else:
                create_timings_row_at_frame_number(project_uuid, index_of_current_item + 1)
                timing_details = data_repo.get_timing_list_from_project(project_uuid)
                data_repo.update_specific_timing(timing_details[index_of_current_item + 1].uuid, frame_time=key_frame_time)

            st.success(f"Key frame added at {key_frame_time} seconds")
            time.sleep(1)
            
            timing_details = data_repo.get_timing_list_from_project(project_uuid)
            if selected_image != "":
                file_data = {
                    "name": str(uuid.uuid4()) + ".png",
                    "type": InternalFileType.IMAGE.value,
                    "local_path": selected_image
                }
                selected_image = data_repo.create_file(**file_data)
                data_repo.update_specific_timing(timing_details[index_of_current_item + 1].uuid, source_image_id=selected_image.uuid)
                data_repo.update_specific_timing(timing_details[index_of_current_item + 1].uuid, animation_style=project_settings.default_animation_style)

            if len(timing_details) == 1:
                st.session_state['current_frame_index'] = 0
            else:
                st.session_state['current_frame_index'] = st.session_state['current_frame_index'] + 1

            st.session_state['section'] = "Guidance"
            st.session_state['section_index'] = 0
            st.experimental_rerun()
