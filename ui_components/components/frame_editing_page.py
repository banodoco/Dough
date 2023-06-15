from io import BytesIO
import os
import time
from typing import List
import uuid
import streamlit as st
from PIL import Image
import requests as r
from streamlit_drawable_canvas import st_canvas
from streamlit_image_comparison import image_comparison
from shared.constants import InternalFileTag, InternalFileType
from shared.file_upload.s3 import upload_file

from ui_components.common_methods import add_image_variant, create_or_update_mask, execute_image_edit, extract_frame, promote_image_variant
from ui_components.models import InternalAppSettingObject, InternalFrameTimingObject, InternalProjectObject, InternalSettingObject
from utils.data_repo.data_repo import DataRepo


def frame_editing_page(project_uuid: str):
    # if 0_extract folder is empty, show error
    data_repo = DataRepo()
    project: InternalProjectObject = data_repo.get_project_from_uuid(
        project_uuid)
    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)

    if len(timing_details) == 0:
        st.info("You need to add  key frames first in the Key Frame Selection section.")

    else:
        project_settings: InternalSettingObject = data_repo.get_project_setting(
            project_uuid)

        # initiative value
        if "current_frame_uuid" not in st.session_state:
            st.session_state['current_frame_uuid'] = timing_details[0].uuid

        def reset_new_image():
            st.session_state['edited_image'] = ""

        if "which_stage" not in st.session_state:
            st.session_state['which_stage'] = "Unedited Key Frame"
            st.session_state['which_stage_index'] = 0

        f1, f2, f3 = st.columns([1, 2, 1])
        with f1:
            frame_number = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(
                timing_details)-1, on_change=reset_new_image, value=st.session_state['current_frame_index'])
            
            st.session_state['current_frame_uuid'] = timing_details[frame_number].uuid
            frame_index = next((i for i, t in enumerate(timing_details) if t.uuid == st.session_state['current_frame_uuid']), None)

            if st.session_state['current_frame_index'] != frame_index:
                st.session_state['current_frame_index'] = frame_index
                st.experimental_rerun()
        with f2:
            st.session_state['which_stage'] = st.radio('Select stage:', [
                                                       "Unedited Key Frame", "Styled Key Frame"], horizontal=True, on_change=reset_new_image, index=st.session_state['which_stage_index'])
            if st.session_state['which_stage'] == "Styled Key Frame" and st.session_state['which_stage_index'] == 0:
                st.session_state['which_stage_index'] = 1
                st.experimental_rerun()

        with f3:
            st.write("")
            if st.session_state['which_stage'] == "Unedited Key Frame":
                if st.button("Reset Key Frame", help="This will reset the base key frame to the original unedited version. This will not affect the video."):
                    timing = data_repo.get_timing_from_uuid(
                        st.session_state['current_frame_uuid'])
                    extract_frame(
                        st.session_state['current_frame_uuid'], project_settings.input_video, timing.frame_number)
                    st.experimental_rerun()
            elif st.session_state['which_stage'] == "Styled Key Frame":
                if st.button("Reset Style", help="This will reset the style of the key frame to the original one."):
                    data_repo.update_specific_timing(
                        st.session_state['current_frame_uuid'], primary_image=0)
                    st.experimental_rerun()

        if "edited_image" not in st.session_state:
            st.session_state.edited_image = ""

        timing = data_repo.get_timing_from_uuid(
            st.session_state['current_frame_uuid'])
        if st.session_state['which_stage'] == "Styled Key Frame" and not len(timing.alternative_images_list):
            st.info("You need to add a style first in the Style Selection section.")
        else:

            if st.session_state['which_stage'] == "Unedited Key Frame":
                editing_image = timing.source_image.location
            elif st.session_state['which_stage'] == "Styled Key Frame":
                primary_image = timing.primary_image
                editing_image = timing.primary_image_location

            width = int(project_settings.width)
            height = int(project_settings.height)

            st.sidebar.markdown("### Select Area To Edit:")
            if 'index_of_type_of_mask_selection' not in st.session_state:
                st.session_state['index_of_type_of_mask_selection'] = 0
            mask_selection_options = ["Automated Background Selection", "Automated Layer Selection",
                                      "Manual Background Selection", "Re-Use Previous Mask", "Invert Previous Mask", "Edit Canny Image"]
            type_of_mask_selection = st.sidebar.radio("How would you like to select what to edit?",
                                                      mask_selection_options, horizontal=True, index=st.session_state['index_of_type_of_mask_selection'])
            if st.session_state['index_of_type_of_mask_selection'] != mask_selection_options.index(type_of_mask_selection):
                st.session_state['index_of_type_of_mask_selection'] = mask_selection_options.index(
                    type_of_mask_selection)
                st.experimental_rerun()

            if "which_layer" not in st.session_state:
                st.session_state['which_layer'] = "Background"
                st.session_state['which_layer_index'] = 0

            if type_of_mask_selection == "Automated Layer Selection":
                layers = ["Background", "Middleground", "Foreground"]
                st.session_state['which_layer'] = st.sidebar.multiselect(
                    "Which layers would you like to replace?", layers)

            if type_of_mask_selection == "Manual Background Selection":
                if st.session_state['edited_image'] == "":
                    if editing_image.startswith("http"):
                        canvas_image = r.get(editing_image)
                        canvas_image = Image.open(
                            BytesIO(canvas_image.content))
                    else:
                        canvas_image = Image.open(editing_image)
                    if 'drawing_input' not in st.session_state:
                        st.session_state['drawing_input'] = 'Magic shapes 🪄'
                    col1, col2 = st.columns([6, 3])

                    with col1:
                        st.session_state['drawing_input'] = st.sidebar.radio(
                            "Drawing tool:",
                            ("Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Draw lines ✏️"), horizontal=True,
                        )

                    if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                        drawing_mode = "transform"
                        st.sidebar.info(
                            "To delete something, just move it outside of the image! 🥴")
                    elif st.session_state['drawing_input'] == "Make shapes 🪄":
                        drawing_mode = "polygon"
                        st.sidebar.info("To end a shape, right click!")
                    elif st.session_state['drawing_input'] == "Draw lines ✏️":
                        drawing_mode = "freedraw"
                        st.sidebar.info("To draw, draw! ")

                    with col2:
                        if drawing_mode == "freedraw":
                            stroke_width = st.slider(
                                "Stroke width: ", 1, 25, 12)
                        else:
                            stroke_width = 3

                    realtime_update = True

                    canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0)",
                        stroke_width=stroke_width,
                        stroke_color="rgba(0, 0, 0)",
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
                        create_or_update_mask(
                            project.name, st.session_state['current_frame_uuid'], im)
                else:
                    image_comparison(
                        img1=editing_image,
                        img2=st.session_state['edited_image'], starting_position=5, label1="Original", label2="Edited"
                    )

                    if st.button("Reset Canvas"):
                        st.session_state['edited_image'] = ""
                        st.experimental_rerun()

            elif type_of_mask_selection == "Edit Canny Image":
                timing = data_repo.get_timing_from_uuid(
                    st.session_state['current_frame_uuid'])
                if timing.canny_image.location == "":
                    st.error("No Canny Image Found From Key Frame")

                else:
                    if timing.canny_image.location .startswith("http"):
                        canvas_image = r.get(timing.canny_image.location)
                        canvas_image = Image.open(
                            BytesIO(canvas_image.content))
                    else:
                        canvas_image = Image.open(timing.canny_image.location)
                    if 'drawing_input' not in st.session_state:
                        st.session_state['drawing_input'] = 'Magic shapes 🪄'
                    col1, col2 = st.columns([6, 3])

                    with col1:
                        st.session_state['drawing_input'] = st.sidebar.radio(
                            "Drawing tool:",
                            ("Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Draw lines ✏️", "Erase Lines ❌"), horizontal=True,
                        )

                    if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                        drawing_mode = "transform"
                        st.sidebar.info(
                            "To delete something, just move it outside of the image! 🥴")
                        stroke_colour = "rgba(0, 0, 0)"
                    elif st.session_state['drawing_input'] == "Make shapes 🪄":
                        drawing_mode = "polygon"
                        stroke_colour = "rgba(0, 0, 0)"
                        st.sidebar.info("To end a shape, right click!")
                    elif st.session_state['drawing_input'] == "Draw lines ✏️":
                        drawing_mode = "freedraw"
                        stroke_colour = "rgba(0, 0, 0)"
                        st.sidebar.info("To draw, draw! ")
                    elif st.session_state['drawing_input'] == "Erase Lines ❌":
                        drawing_mode = "freedraw"
                        stroke_colour = "rgba(255, 255, 255)"
                        st.sidebar.info("To erase, draw! ")

                    with col2:
                        if drawing_mode == "freedraw":
                            stroke_width = st.sidebar.slider(
                                "Stroke width: ", 1, 25, 1)
                        else:
                            stroke_width = 3

                    realtime_update = True

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

                    if st.button("Save New Canny Image"):
                        if canvas_result.image_data is not None:
                            # overlay the canvas image on top of the canny image and save the result
                            # if canny image is from a url, then we need to download it first
                            timing = data_repo.get_timing_from_uuid(
                                st.session_state['current_frame_uuid'])
                            if timing.canny_image.location.startswith("http"):
                                canny_image = r.get(
                                    timing.canny_image.location)
                                canny_image = Image.open(
                                    BytesIO(canny_image.content))
                            else:
                                canny_image = Image.open(
                                    timing.canny_image.location)
                            canny_image = canny_image.convert("RGBA")
                            canvas_image = im
                            canvas_image = canvas_image.convert("RGBA")
                            new_canny_image = Image.alpha_composite(
                                canny_image, canvas_image)
                            new_canny_image = new_canny_image.convert("RGB")
                            file_path = f"videos/{project.name}/assets/resources/masks/{st.session_state['current_frame_uuid']}.png"
                            new_canny_image.save(file_path)
                            file_data = {
                                "name": str(uuid.uuid4()) + ".png",
                                "type": InternalFileType.IMAGE.value,
                                "local_path": file_path
                            }
                            canny_image = data_repo.create_file(**file_data)
                            data_repo.update_specific_timing(
                                st.session_state['current_frame_uuid'], canny_image_id=canny_image.uuid)
                            st.experimental_rerun()

                        else:
                            st.error("No Image Found")

                canny1, canny2 = st.columns([1, 1])
                with canny1:
                    st.markdown("#### Use Canny Image From Other Frame")
                    st.markdown(
                        "This will use a canny image from another frame. This will take a few seconds.")
                    which_number_image_for_canny = st.number_input(
                        "Which frame would you like to use?", min_value=0, max_value=len(timing_details)-1, value=0, step=1)
                    if st.button("Use Canny Image From Other Frame"):
                        timing = data_repo.get_timing_from_frame_number(
                            project.uuid,
                            which_number_image_for_canny)
                        if timing.canny_image.location != "":
                            data_repo.update_specific_timing(
                                st.session_state['current_frame_uuid'], "canny_image", timing.canny_image.location)
                            st.experimental_rerun()
                    if timing.canny_image.location == "":
                        st.error("No Canny Image Found From Key Frame")
                with canny2:
                    st.markdown("#### Upload Canny Image")
                    st.markdown(
                        "This will upload a canny image from your computer. This will take a few seconds.")
                    uploaded_file = st.file_uploader("Choose a file")
                    if st.button("Upload Canny Image"):
                        with open(os.path.join(f"videos/{project.name}/assets/resources/masks", uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            st.success(
                                "Your backgrounds are uploaded file - they should appear in the dropdown.")
                            file_data = {
                                "name": str(uuid.uuid4()) + ".png",
                                "type": InternalFileType.IMAGE.value,
                                "local_path": f"videos/{project.name}/assets/resources/masks/{uploaded_file.name}"
                            }
                            canny_image = data_repo.create_file(**file_data)
                            data_repo.update_specific_timing(
                                st.session_state['current_frame_uuid'], canny_image_id=canny_image.uuid)
                            time.sleep(1.5)
                            st.experimental_rerun()

            elif type_of_mask_selection == "Automated Background Selection" or type_of_mask_selection == "Automated Layer Selection" or type_of_mask_selection == "Re-Use Previous Mask" or type_of_mask_selection == "Invert Previous Mask":
                if type_of_mask_selection == "Re-Use Previous Mask" or type_of_mask_selection == "Invert Previous Mask":
                    if timing_details[st.session_state['current_frame_uuid']]["mask"] == "":
                        st.sidebar.info(
                            "You don't have a previous mask to re-use.")
                    else:
                        mask1, mask2 = st.sidebar.columns([2, 1])
                        with mask1:
                            if type_of_mask_selection == "Re-Use Previous Mask":
                                st.info(
                                    "This will update the **black pixels** in the mask with the pixels from the image you are editing.")
                            elif type_of_mask_selection == "Invert Previous Mask":
                                st.info(
                                    "This will update the **white pixels** in the mask with the pixels from the image you are editing.")
                            st.image(
                                timing_details[st.session_state['current_frame_uuid']]["mask"], use_column_width=True)
                if st.session_state['edited_image'] == "":
                    st.image(editing_image, use_column_width=True)
                else:
                    image_comparison(
                        img1=editing_image,
                        img2=st.session_state['edited_image'], starting_position=5, label1="Original", label2="Edited"
                    )
                    if st.button("Reset Canvas"):
                        st.session_state['edited_image'] = ""
                        st.experimental_rerun()

            st.sidebar.markdown("### Edit Individual Image:")

            if "type_of_mask_replacement" not in st.session_state:
                st.session_state["type_of_mask_replacement"] = "Replace With Image"
                st.session_state["index_of_type_of_mask_replacement"] = 0

            types_of_mask_replacement = ["Replace With Image", "Inpainting"]
            st.session_state["type_of_mask_replacement"] = st.sidebar.radio(
                "Select type of edit", types_of_mask_replacement, horizontal=True, index=st.session_state["index_of_type_of_mask_replacement"])

            if st.session_state["index_of_type_of_mask_replacement"] != types_of_mask_replacement.index(st.session_state["type_of_mask_replacement"]):
                st.session_state["index_of_type_of_mask_replacement"] = types_of_mask_replacement.index(
                    st.session_state["type_of_mask_replacement"])
                st.experimental_rerun()

            if st.session_state["type_of_mask_replacement"] == "Replace With Image":
                prompt = ""
                negative_prompt = ""
                background_image_list = data_repo.get_all_file_list(InternalFileType.IMAGE.value, InternalFileTag.BACKGROUND_IMAGE.value)
                # background_list = [f for f in os.listdir(
                #     f'videos/{project.name}/assets/resources/backgrounds') if f.endswith('.png')]
                
                background_list = [(f.name, f.uuid) for f in background_image_list]

                sources_of_images = ["Uploaded", "From Other Frame"]
                if 'index_of_source_of_image' not in st.session_state:
                    st.session_state['index_of_source_of_image'] = 0
                source_of_image = st.sidebar.radio(
                    "Select type of image", sources_of_images, horizontal=True, index=st.session_state['index_of_source_of_image'])

                if st.session_state['index_of_source_of_image'] != sources_of_images.index(source_of_image):
                    st.session_state['index_of_source_of_image'] = sources_of_images.index(
                        source_of_image)
                    st.experimental_rerun()

                if source_of_image == "Uploaded":
                    btn1, btn2 = st.sidebar.columns([1, 1])
                    with btn1:
                        uploaded_files = st.file_uploader(
                            "Add more background images here", accept_multiple_files=True)
                        if st.button("Upload Backgrounds"):
                            for uploaded_file in uploaded_files:
                                with open(os.path.join(f"videos/{project.name}/assets/resources/backgrounds", uploaded_file.name), "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                    st.success(
                                        "Your backgrounds are uploaded file - they should appear in the dropdown.")
                                    file_data = {
                                        "name": str(uuid.uuid4()) + ".png",
                                        "type": InternalFileType.IMAGE.value,
                                        "local_path": f"videos/{project.name}/assets/resources/backgrounds/{uploaded_file.name}"
                                    }
                                    canny_image = data_repo.create_file(**file_data)
                                    background_list.append((canny_image.name, canny_image.uuid))
                                    time.sleep(1.5)
                                    st.experimental_rerun()
                    with btn2:
                        background_selection = st.sidebar.selectbox(
                            "Range background", background_list)
                        
                        if background_selection:
                            background_image_uuid = background_selection[1]
                            background_image = data_repo.get_file_from_uuid(background_image_uuid)
                            if background_list != []:
                                st.image(f"{background_image}", use_column_width=True)

                elif source_of_image == "From Other Frame":
                    btn1, btn2 = st.sidebar.columns([1, 1])
                    with btn1:
                        which_stage_to_use = st.radio("Select stage to use:", [
                                                      "Unedited Key Frame", "Styled Key Frame"])
                        which_image_to_use = st.number_input(
                            "Select image to use:", min_value=0, max_value=len(timing_details)-1, value=0)
                        
                        timing = data_repo.get_timing_from_frame_number(
                            project.uuid,
                            which_image_to_use
                        )
                        if which_stage_to_use == "Unedited Key Frame":
                            background_image = timing.source_image

                        elif which_stage_to_use == "Styled Key Frame":
                            primary_image = timing.primary_image
                            background_image = timing.background_image
                    with btn2:
                        st.image(background_image, use_column_width=True)

            elif st.session_state["type_of_mask_replacement"] == "Inpainting":
                btn1, btn2 = st.sidebar.columns([1, 1])
                with btn1:
                    prompt = st.text_area("Prompt:", help="Describe the whole image, but focus on the details you want changed!",
                                          value=st.session_state['project_settings']["last_prompt"])
                with btn2:
                    negative_prompt = st.text_area("Negative Prompt:", help="Enter any things you want to make the model avoid!",
                                                   value=st.session_state['project_settings']["last_negative_prompt"])

            edit1, edit2 = st.sidebar.columns(2)

            with edit1:
                if st.button(f'Run Edit On Current Image'):
                    if st.session_state["type_of_mask_replacement"] == "Inpainting":
                        editted_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project.name,
                                                                              "", editing_image, prompt, negative_prompt, width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                    elif st.session_state["type_of_mask_replacement"] == "Replace With Image":
                        editted_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project.name,
                                                                              background_image, editing_image, "", "", width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                    st.session_state['edited_image'] = editted_image.uuid
                    st.experimental_rerun()
            with edit2:
                if st.session_state['edited_image'] != "":
                    if st.button("Promote Last Edit", type="primary"):
                        if st.session_state['which_stage'] == "Unedited Key Frame":
                            data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=st.session_state['edited_image'])
                        elif st.session_state['which_stage'] == "Styled Key Frame":
                            number_of_image_variants = add_image_variant(st.session_state['edited_image'], st.session_state['current_frame_uuid'])
                            promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                        st.session_state['edited_image'] = ""
                        st.experimental_rerun()
                else:
                    if st.button("Run Edit & Promote"):
                        if st.session_state["type_of_mask_replacement"] == "Inpainting":
                            st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project.name,
                                                                                  "", editing_image, prompt, negative_prompt, width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                        elif st.session_state["type_of_mask_replacement"] == "Replace With Image":
                            st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project.name,
                                                                                  background_image, editing_image, "", "", width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                        if st.session_state['which_stage'] == "Unedited Key Frame":
                            data_repo.update_specific_timing_value(st.session_state['current_frame_uuid'], source_image_id=st.session_state['edited_image'])
                        elif st.session_state['which_stage'] == "Styled Key Frame":
                            number_of_image_variants = add_image_variant(st.session_state['edited_image'], st.session_state['current_frame_uuid'])
                            promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)

                        st.session_state['edited_image'] = ""
                        st.success("Image promoted!")
                        st.experimental_rerun()

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
                        timing = data_repo.get_timing_from_frame_number()
                        if which_stage_to_use_for_replacement == "Unedited Key Frame":
                            background_image = timing.source_image
                        elif which_stage_to_use_for_replacement == "Styled Key Frame":
                            primary_image = timing.primary_image
                            background_image = timing.background_image

                        if st.button("Replace with selected frame", disabled=False):
                            if st.session_state['which_stage'] == "Unedited Key Frame":
                                data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=background_image.uuid)
                            elif st.session_state['which_stage'] == "Styled Key Frame":
                                number_of_image_variants = add_image_variant(background_image, st.session_state['current_frame_uuid'])
                                promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)
                            st.success("Replaced")
                            time.sleep(1)
                            st.experimental_rerun()
                    with replace2:
                        st.image(background_image, width=300)

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
                            with open(os.path.join(f"videos/{project.name}/", replacement_frame.name), "wb") as f:
                                f.write(replacement_frame.getbuffer())
                            
                            app_setting: InternalAppSettingObject = data_repo.get_app_setting_from_uuid()

                            uploaded_image_url = upload_file(
                                f"videos/{project.name}/{replacement_frame.name}", app_setting.aws_access_key, app_setting.aws_secret_access_key)
                            
                            file_data = {
                                "name": str(uuid.uuid4()) + ".png",
                                "type": InternalFileType.IMAGE.value,
                                "hosted_url": uploaded_image_url
                            }
                            uploaded_image = data_repo.create_file(**file_data)
                            
                            if st.session_state['which_stage'] == "Unedited Key Frame":
                                data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=uploaded_image.uuid)
                            elif st.session_state['which_stage'] == "Styled Key Frame":
                                number_of_image_variants = add_image_variant(uploaded_image.uuid, st.session_state['current_frame_uuid'])
                                promote_image_variant(st.session_state['current_frame_uuid'], number_of_image_variants - 1)

                            # delete the uploaded file
                            os.remove(f"videos/{project.name}/{replacement_frame.name}")
                            st.success("Replaced")
                            time.sleep(1)
                            st.experimental_rerun()

        st.sidebar.markdown("***")
        st.sidebar.markdown("### Batch Run Edits:")
        st.sidebar.write(
            "This will batch run the settings you have above on a batch of images.")
        batch_run_range = st.sidebar.slider(
            "Select range:", 1, 0, (0, len(timing_details)-1))
        if st.session_state['which_stage'] == "Unedited Key Frame":
            st.sidebar.warning(
                "This will overwrite the source images in the range you select - you can always reset them if you wish.")
        elif st.session_state['which_stage'] == "Styled Key Frame":
            make_primary_variant = st.sidebar.checkbox(
                "Make primary variant", value=True, help="If you want to make the edited image the primary variant, tick this box. If you want to keep the original primary variant, untick this box.")
        if st.sidebar.button("Batch Run Edit"):
            for i in range(batch_run_range[1]+1):
                if st.session_state["type_of_mask_replacement"] == "Inpainting":
                    background_image = ""
                if st.session_state['which_stage'] == "Unedited Key Frame":
                    editing_image = timing_details[i].source_image
                    edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project.name, background_image,
                                                      editing_image, prompt, negative_prompt, width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                elif st.session_state['which_stage'] == "Styled Key Frame":
                    editing_image = timing_details[i].primary_image
                    edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project.name, background_image,
                                                      editing_image, prompt, negative_prompt, width, height, st.session_state['which_layer'], st.session_state['current_frame_uuid'])
                    number_of_image_variants = add_image_variant(edited_image, i)
                    promote_image_variant(i, number_of_image_variants-1)
                    
            st.experimental_rerun()
