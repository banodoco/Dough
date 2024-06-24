import io
import uuid
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from shared.constants import QUEUE_INFERENCE_QUERIES, InferenceType
from ui_components.constants import (
    CROPPED_IMG_LOCAL_PATH,
    MASK_IMG_LOCAL_PATH,
    TEMP_MASK_FILE,
    DefaultProjectSettingParams,
    WorkflowStageType,
)
from ui_components.methods.file_methods import (
    add_temp_file_to_project,
    detect_and_draw_contour,
    save_or_host_file,
    zoom_and_crop,
)
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo

from utils.data_repo.data_repo import DataRepo
from ui_components.methods.common_methods import (
    add_image_variant,
    apply_coord_transformations,
    process_inference_output,
    promote_image_variant,
)
from ui_components.models import InternalProjectObject, InternalSettingObject
from utils.ml_processor.constants import ML_MODEL
from utils.ml_processor.ml_interface import get_ml_client


def inpainting_element(options_width, image, position="explorer"):
    data_repo = DataRepo()
    project_settings: InternalSettingObject = data_repo.get_project_setting(st.session_state["project_uuid"])

    if "current_mask" not in st.session_state:
        st.session_state["current_mask"] = ""

    main_col_1, main_col_2 = st.columns([0.5, 3])
    original_width = int(project_settings.width)
    original_height = int(project_settings.height)
    enlarged_width = int(original_width * 1.2)
    enlarged_height = int(original_height * 1.2)

    if st.session_state["current_mask"] != "":
        with main_col_2:
            st.image(st.session_state["current_mask"], width=project_settings.width)
            st.info("The bright white areas will be inpainted, the faded areas be kept.")

            if st.button("Reset area to inpaint", use_container_width=True, key=f"clear_inpaint_mak_{position}"):
                st.session_state["current_mask"] = ""
                st.session_state["mask_to_use"] = ""
                st.rerun()
    else:
        with main_col_1:
            canvas_image = image if isinstance(image, Image.Image) else Image.open(image)
            # Create a new image with black background and place the original image in the center
            new_canvas = Image.new("RGB", (enlarged_width, enlarged_height))
            for y in range(enlarged_height):
                for x in range(enlarged_width):
                    if (x // 50) % 2 == (y // 50) % 2:
                        new_canvas.putpixel((x, y), (255, 255, 255))
                    else:
                        new_canvas.putpixel((x, y), (240, 240, 240))
            offset = ((enlarged_width - original_width) // 2, (enlarged_height - original_height) // 2)
            new_canvas.paste(canvas_image, offset)

            if "drawing_input" not in st.session_state:
                st.session_state["drawing_input"] = "Magic shapes 🪄"

            with options_width:
                # Check if the mode switch button has been pressed and toggle the mode
                if "mode" not in st.session_state:
                    st.session_state["mode"] = "draw"  # Default mode

                # Display the drawing tool options when in draw mode
                if st.session_state["mode"] == "draw":
                    if st.button("Switch to move mode"):
                        st.session_state["mode"] = "move"
                        st.session_state["drawing_input"] = "Move shapes 🏋🏾‍♂️"
                        st.rerun()
                    st.session_state["drawing_input"] = st.radio(
                        "Drawing tool:",
                        ("Draw lines ✏️", "Make squares □", "Make shapes 🪄"),
                        horizontal=True,
                        key="drawing_tool",
                        help="It'll inpaint over the area you paint black - apart from the area outside the border.",
                    )
                    st.info(
                        "You can draw the mask on the canvas - anything apart from the buffer will be inpainted."
                    )

                else:

                    if st.button("Switch to draw mode"):
                        st.session_state["mode"] = "draw"
                        # Optionally reset to a default drawing tool
                        st.session_state["drawing_input"] = "Draw lines ✏️"
                        st.rerun()
                    st.info("You can move the shapes around to adjust the mask.")
                # Set drawing mode based on the current state
                if st.session_state["drawing_input"] == "Move shapes 🏋🏾‍♂️":
                    drawing_mode = "transform"

                elif st.session_state["drawing_input"] == "Make shapes 🪄":
                    drawing_mode = "polygon"

                elif st.session_state["drawing_input"] == "Draw lines ✏️":
                    drawing_mode = "freedraw"

                elif st.session_state["drawing_input"] == "Make squares □":
                    drawing_mode = "rect"

                # Adjust stroke width based on the drawing mode
                if drawing_mode == "freedraw":
                    stroke_width = st.slider("Stroke width: ", 1, 100, 50)
                else:
                    stroke_width = 3

                st.markdown("***")

        with main_col_2:
            realtime_update = True
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color="rgba(0, 0, 0)",
                background_color="rgb(255, 255, 255)",
                background_image=new_canvas,
                update_streamlit=realtime_update,
                height=enlarged_height,
                width=enlarged_width,
                drawing_mode=drawing_mode,
                display_toolbar=True,
                key="full_app",
            )

            if "image_created" not in st.session_state:
                st.session_state["image_created"] = "no"
        with main_col_1:
            if position == "explorer":
                if st.button("Pick new image", use_container_width=True):
                    st.session_state["uploaded_image"] = ""
                    st.rerun()
        with main_col_2:
            if st.button("Save area to inpaint", use_container_width=True):
                img_data = canvas_result.image_data
                im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
                im = ImageOps.crop(im, border=(offset[0], offset[1]))  # Cropping back to original size
                im_rgb = Image.new("RGB", im.size, (255, 255, 255))
                im_rgb.paste(im, mask=im.split()[3])  # Paste the mask onto the RGB image
                im = im_rgb
                im = ImageOps.invert(im)  # Inverting for sdxl inpainting

                st.session_state["editing_image"] = (
                    image if isinstance(image, Image.Image) else Image.open(image)
                )
                mask_file_path = "videos/temp/" + str(uuid.uuid4()) + ".png"
                mask_file_path = save_or_host_file(im, mask_file_path) or mask_file_path
                st.session_state["mask_to_use"] = mask_file_path

                # Ensure the mask is in the same size as the canvas_image
                im_resized = im.resize(canvas_image.size)
                im = ImageOps.invert(im)

                # Create an image with the mask faded to 50% and overlay it over the canvas_image
                faded_mask_overlay = Image.blend(
                    canvas_image.convert("RGBA"), im_resized.convert("RGBA"), alpha=0.7
                )

                # Save the image with the mask overlayed on top of it to session state
                st.session_state["current_mask"] = faded_mask_overlay
                st.rerun()


def inpainting_image_input(project_uuid, position="explorer"):
    data_repo = DataRepo()
    options_width, canvas_width = st.columns([1.2, 3])
    project_settings: InternalSettingObject = data_repo.get_project_setting(project_uuid)
    if not ("uploaded_image" in st.session_state and st.session_state["uploaded_image"]):
        st.session_state["uploaded_image"] = ""
        with options_width:
            if st.session_state["uploaded_image"] == "" or st.session_state["uploaded_image"] is None:
                source_of_starting_image = st.radio(
                    "Image source:",
                    options=["Upload", "From Shot"],
                    key=f"starting_image_{position}",
                    help="This will be the base image for the generation.",
                    horizontal=True,
                )
                if source_of_starting_image == "Upload":
                    uploaded_image = st.file_uploader(
                        "Upload a starting image",
                        type=["png", "jpg", "jpeg"],
                        key=f"uploaded_image_{position}",
                        help="This will be the base image for the generation.",
                    )
                    if uploaded_image:
                        if st.button("Select as base image", key=f"inpainting_base_image_{position}"):
                            uploaded_image = Image.open(uploaded_image)
                            uploaded_image = zoom_and_crop(
                                uploaded_image, project_settings.width, project_settings.height
                            )
                            st.session_state["uploaded_image"] = uploaded_image
                else:
                    # taking image from shots
                    shot_list = data_repo.get_shot_list(project_uuid)
                    shot_name = st.selectbox(
                        "Shot:",
                        options=[shot.name for shot in shot_list],
                        key=f"inpainting_shot_name_{position}",
                        help="This will be the base image for the generation.",
                    )
                    shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                    frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                    list_of_timings = [i + 1 for i in range(len(frame_list))]
                    timing = st.selectbox(
                        "Frame #:",
                        options=list_of_timings,
                        key=f"inpainting_frame_number_{position}",
                        help="This will be the base image for the generation.",
                    )
                    st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)
                    if timing:
                        if st.button("Select as base image", key="inpainting_base_image_2_{position}"):
                            st.session_state["uploaded_image"] = frame_list[timing - 1].primary_image.location

    with canvas_width:
        if st.session_state["uploaded_image"]:
            inpainting_element(options_width, st.session_state["uploaded_image"], position)
        else:
            st.info("<- Please select an image")


def replace_with_image(stage, output_file, current_frame_uuid, promote=False):
    data_repo = DataRepo()

    if stage == WorkflowStageType.SOURCE.value:
        data_repo.update_specific_timing(
            current_frame_uuid, source_image_id=output_file.uuid, update_in_place=True
        )
    elif stage == WorkflowStageType.STYLED.value:
        number_of_image_variants = add_image_variant(output_file.uuid, current_frame_uuid)
        if promote:
            promote_image_variant(current_frame_uuid, number_of_image_variants - 1)

    st.rerun()


def generate_mask(vertex_coords, canvas_width, canvas_height, upscale_factor=1):
    upscale_width = canvas_width * upscale_factor
    upscale_height = canvas_height * upscale_factor
    canvas = Image.new("RGB", (upscale_width, upscale_height), "white")
    draw = ImageDraw.Draw(canvas)
    upscaled_vertex_coords = [(x * upscale_factor, y * upscale_factor) for x, y in vertex_coords]
    draw.polygon(upscaled_vertex_coords, fill="black", outline="black")
    # canvas.save("original_mask.png")
    canvas = canvas.resize((canvas_width, canvas_height), resample=Image.LANCZOS)

    # width, height = canvas.size
    # for y in range(height):
    #     for x in range(width):
    #         r, g, b = canvas.getpixel((x, y))
    #         if r + g + b > 10:
    #             canvas.putpixel((x, y), (255, 255, 255))

    return canvas
