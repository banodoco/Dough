import uuid
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shared.constants import InferenceType
from ui_components.constants import CROPPED_IMG_LOCAL_PATH, MASK_IMG_LOCAL_PATH, TEMP_MASK_FILE, DefaultProjectSettingParams, WorkflowStageType
from ui_components.methods.file_methods import add_temp_file_to_project, save_or_host_file
from utils.data_repo.data_repo import DataRepo

from utils.data_repo.data_repo import DataRepo
from ui_components.methods.common_methods import add_image_variant, process_inference_output, promote_image_variant
from ui_components.models import InternalSettingObject


def inpainting_element(options_width, image, position="explorer"):
    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        st.session_state['project_uuid'])

    if "current_mask" not in st.session_state:
        st.session_state["current_mask"] = ""

    main_col_1, main_col_2 = st.columns([0.5, 3])
    width = int(project_settings.width)
    height = int(project_settings.height)

    if st.session_state['current_mask'] != "":
        with main_col_2:                
            st.image(st.session_state['current_mask'], width=project_settings.width)
        
            if st.button("Clear Mask", use_container_width=True, key=f"clear_inpaint_mak_{position}"):
                st.session_state['current_mask'] = ""
                st.session_state['mask_to_use'] = ""
                st.rerun()
    else:
        with main_col_1:
            canvas_image = Image.open(image)
            if 'drawing_input' not in st.session_state:
                st.session_state['drawing_input'] = 'Magic shapes 🪄'

            with options_width:
                st.session_state['drawing_input'] = st.radio(
                    "Drawing tool:",
                    ("Make shapes 🪄", "Move shapes 🏋🏾‍♂️", "Make squares □", "Draw lines ✏️"), horizontal=True)

                if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                    drawing_mode = "transform"
                    st.info(
                        "To delete something, double click on it.")
                elif st.session_state['drawing_input'] == "Make shapes 🪄":
                    drawing_mode = "polygon"
                    st.info("To end a shape, right click!")
                elif st.session_state['drawing_input'] == "Draw lines ✏️":
                    drawing_mode = "freedraw"
                    st.info("To draw, draw! ")
                elif st.session_state['drawing_input'] == "Make squares □":
                    drawing_mode = "rect"

                with options_width:
                    if drawing_mode == "freedraw":
                        stroke_width = st.slider(
                            "Stroke width: ", 1, 25, 12)
                    else:
                        stroke_width = 3

        with main_col_2:
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
            
            if position == "explorer":
                if st.button("Pick new image", use_container_width=True):
                    st.session_state["uploaded_image"] = ""
                    st.rerun()

            # saves both the image and mask in the session state
            if st.button("Save Mask", use_container_width=True):
                img_data = canvas_result.image_data
                im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
                im_rgb = Image.new("RGB", im.size, (255, 255, 255))
                im_rgb.paste(im, mask=im.split()[3])  # Paste the mask onto the RGB image
                im = im_rgb
                im = ImageOps.invert(im)  # Inverting for sdxl inpainting
                st.session_state['editing_image'] = image
                mask_file_path = "videos/temp/" + str(uuid.uuid4()) + ".png"
                mask_file_path = save_or_host_file(im, mask_file_path) or mask_file_path
                st.session_state['mask_to_use'] = mask_file_path
                
                # Ensure the mask is in the same size as the canvas_image
                im_resized = im.resize(canvas_image.size)
                im = ImageOps.invert(im)
                
                # Create an image with the mask faded to 50% and overlay it over the canvas_image
                faded_mask_overlay = Image.blend(canvas_image.convert("RGBA"), im_resized.convert("RGBA"), alpha=0.3)
                
                # Save the image with the mask overlayed on top of it to session state
                st.session_state['current_mask'] = faded_mask_overlay
                st.rerun()
    
def inpainting_image_input(project_uuid, position="explorer"):
    data_repo = DataRepo()
    options_width, canvas_width = st.columns([1.2, 3])
    if not ('uploaded_image' in st.session_state and st.session_state["uploaded_image"]):
        st.session_state['uploaded_image'] = ""
        with options_width:
            if st.session_state['uploaded_image'] == "" or st.session_state['uploaded_image'] is None:
                source_of_starting_image = st.radio("Image source:", options=["Upload","From Shot"], key=f"starting_image_{position}", help="This will be the base image for the generation.", horizontal=True)
                if source_of_starting_image == "Upload":
                    uploaded_image = st.file_uploader("Upload a starting image", type=["png", "jpg", "jpeg"], key=f"uploaded_image_{position}", help="This will be the base image for the generation.")
                    if uploaded_image:
                        if st.button("Select as base image", key=f"inpainting_base_image_{position}"):
                            st.session_state['uploaded_image'] = uploaded_image
                else:
                    # taking image from shots
                    shot_list = data_repo.get_shot_list(project_uuid)      
                    shot_name = st.selectbox("Shot:", options=[shot.name for shot in shot_list], key=f"inpainting_shot_name_{position}", help="This will be the base image for the generation.")
                    shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                    frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                    list_of_timings = [i + 1 for i in range(len(frame_list))]
                    timing = st.selectbox("Frame #:", options=list_of_timings, key=f"inpainting_frame_number_{position}", help="This will be the base image for the generation.")
                    st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)
                    if timing:
                        if st.button("Select as base image", key="inpainting_base_image_2_{position}"):
                            st.session_state['uploaded_image'] = frame_list[timing - 1].primary_image.location

    with canvas_width:
        if st.session_state['uploaded_image']:
            inpainting_element(options_width, st.session_state['uploaded_image'], position)
        else:
            st.info("<- Please select an image")

def replace_with_image(stage, output_file, current_frame_uuid, promote=False):
    data_repo = DataRepo()

    if stage == WorkflowStageType.SOURCE.value:
        data_repo.update_specific_timing(current_frame_uuid, source_image_id=output_file.uuid)
    elif stage == WorkflowStageType.STYLED.value:
        number_of_image_variants = add_image_variant(output_file.uuid, current_frame_uuid)
        if promote:
            promote_image_variant(current_frame_uuid, number_of_image_variants - 1)
    
    st.rerun()

# cropped_img here is a PIL image object
def inpaint_in_black_space_element(cropped_img, project_uuid, stage=WorkflowStageType.SOURCE.value):
    from ui_components.methods.ml_methods import inpainting

    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        project_uuid)

    st.markdown("##### Inpaint in black space:")

    inpaint_prompt = st.text_area("Prompt", value=st.session_state['explorer_base_prompt'])
    inpaint_negative_prompt = st.text_input(
        "Negative Prompt", value='edge,branches, frame, fractals, text' + DefaultProjectSettingParams.batch_negative_prompt)
    if 'precision_cropping_inpainted_image_uuid' not in st.session_state:
        st.session_state['precision_cropping_inpainted_image_uuid'] = ""

    def inpaint(promote=False):
        width = int(project_settings.width)
        height = int(project_settings.height)

        saved_cropped_img = cropped_img.resize(
            (width, height), Image.ANTIALIAS)
        
        hosted_cropped_img_path = save_or_host_file(saved_cropped_img, CROPPED_IMG_LOCAL_PATH)
        mask = Image.new('RGB', cropped_img.size)
        width, height = cropped_img.size
        for x in range(width):
            for y in range(height):
                pixel = cropped_img.getpixel((x, y))
                if cropped_img.mode == 'RGB':
                    r, g, b = pixel
                elif cropped_img.mode == 'RGBA':
                    r, g, b, a = pixel
                elif cropped_img.mode == 'L':
                    brightness = pixel
                else:
                    raise ValueError(
                        f'Unsupported image mode: {cropped_img.mode}')

                if r == 0 and g == 0 and b == 0:
                    mask.putpixel((x, y), (0, 0, 0))  # Black
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if 0 <= x + i < width and 0 <= y + j < height:
                                mask.putpixel((x + i, y + j),
                                              (0, 0, 0))  # Black
                else:
                    mask.putpixel((x, y), (255, 255, 255))  # White

        mask = ImageOps.invert(mask)
        hosted_url = save_or_host_file(mask, MASK_IMG_LOCAL_PATH)
        add_temp_file_to_project(project_uuid, TEMP_MASK_FILE, hosted_url or MASK_IMG_LOCAL_PATH)

        cropped_img_path = hosted_cropped_img_path if hosted_cropped_img_path else CROPPED_IMG_LOCAL_PATH
        output, log = inpainting(cropped_img_path, inpaint_prompt,
                                    inpaint_negative_prompt, st.session_state['current_frame_uuid'], True)
        
        inference_data = {
            "inference_type": InferenceType.FRAME_INPAINTING.value,
            "output": output,
            "log_uuid": log.uuid,
            "timing_uuid": st.session_state['current_frame_uuid'],
            "promote_generation": promote,
            "stage": stage
        }

        process_inference_output(**inference_data)


    if st.button("Inpaint"):
        inpaint(promote=False)
        
    
    if st.button("Inpaint and Promote"):
        inpaint(promote=True)