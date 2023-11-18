from io import BytesIO
import uuid
import json
import time
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ui_components.constants import WorkflowStageType
from utils.data_repo.data_repo import DataRepo
from ui_components.methods.common_methods import extract_canny_lines
from shared.constants import InternalFileType
from ui_components.methods.file_methods import save_or_host_file


from utils import st_memory


def drawing_element(timing_details,project_settings,project_uuid,stage=WorkflowStageType.STYLED.value):
    
    drawing_stage = "Styled Image"
    
    stage=WorkflowStageType.STYLED.value
    
    image_path = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
    data_repo = DataRepo()

    canvas1, canvas2 = st.columns([1, 1.5])
    timing = data_repo.get_timing_from_uuid(
        st.session_state['current_frame_uuid'])

    with canvas1:
        width = int(project_settings.width)
        height = int(project_settings.height)

        if timing.source_image and timing.source_image.location != "":
            if timing.source_image.location.startswith("http"):
                canvas_image = r.get(
                    timing.source_image.location)
                canvas_image = Image.open(
                    BytesIO(canvas_image.content))
            else:
                canvas_image = Image.open(
                    image_path)
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
        

        st.markdown("***")
        
                                            
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

            image_path = timing_details[st.session_state['current_frame_index'] - 1].primary_image_location
                        
            canny_image = extract_canny_lines(
                    image_path, project_uuid, low_threshold, high_threshold)
            
            st.session_state['canny_image'] = canny_image.uuid

        if st.session_state['canny_image']:
            canny_image = data_repo.get_file_from_uuid(st.session_state['canny_image'])
            
            canny_action_1, canny_action_2 = st.columns([2, 1])
            with canny_action_1:
                st.image(canny_image.location)
                                                                            
                if st.button(f"Make Into Guidance Image"):
                    # data_repo.update_specific_timing(st.session_state['current_frame_uuid'], source_image_id=st.session_state['canny_image'])
                    st.session_state['reset_canvas'] = True
                    st.session_state['canny_image'] = None
                    st.rerun()

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
                key="full_app_draw",
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
            st.rerun()
        if canvas_result is not None:            
            if canvas_result.json_data is not None and not canvas_result.json_data.get('objects'):
                st.button("Save New Image", key="save_canvas", disabled=True, help="Draw something first")
            else:                
                if st.button("Save New Image", key="save_canvas_active",type="primary"):
                    if canvas_result.image_data is not None:

                        if timing.primary_image_location:
                            if timing.primary_image_location.startswith("http"):
                                canny_image = r.get(
                                    timing.primary_image_location)
                                canny_image = Image.open(
                                    BytesIO(canny_image.content))
                            else:
                                canny_image = Image.open(
                                    timing.primary_image_location)
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
                        file_location = f"videos/{timing.shot.project.uuid}/assets/resources/masks/{unique_file_name}"
                        hosted_url = save_or_host_file(new_canny_image, file_location)
                        file_data = {
                            "name": str(uuid.uuid4()) + ".png",
                            "type": InternalFileType.IMAGE.value,
                            "project_id": project_uuid
                        }

                        if hosted_url:
                            file_data.update({'hosted_url': hosted_url})
                        else:
                            file_data.update({'local_path': file_location})

                        canny_image = data_repo.create_file(
                            **file_data)

                        data_repo.update_specific_timing(
                            st.session_state['current_frame_uuid'], primary_image_id=canny_image.uuid)
                       
                        st.success("New Canny Image Saved")
                        st.session_state['reset_canvas'] = True
                        time.sleep(1)
                        st.rerun()