import streamlit as st
from streamlit_image_comparison import image_comparison
import time
import pandas as pd
import os
from PIL import Image
import requests as r
from streamlit_drawable_canvas import st_canvas
from repository.local_repo.csv_repo import get_app_settings, get_project_settings,update_specific_timing_value,update_project_setting
from ui_components.common_methods import create_gif_preview, delete_frame, get_model_details, get_timing_details, promote_image_variant, trigger_restyling_process,add_image_variant,prompt_interpolation_model,update_speed_of_video_clip,create_timings_row_at_frame_number,extract_canny_lines,get_duration_from_video,get_audio_bytes_for_slice,add_audio_to_video_slice,convert_to_minutes_and_seconds,styling_sidebar,get_primary_variant_location,create_full_preview_video,back_and_forward_buttons,create_single_preview_video,rotate_image,zoom_image
from utils.file_upload.s3 import upload_image
import uuid
import datetime
from pydub import AudioSegment
from io import BytesIO
import shutil
from streamlit_option_menu import option_menu
from moviepy.editor import concatenate_videoclips
import moviepy.editor





def frame_styling_page(mainheader2, project_name):

    
    timing_details = get_timing_details(project_name)

    if len(timing_details) == 0:
        if st.button("Create timings row"):
            create_timings_row_at_frame_number(project_name, 0)
            update_specific_timing_value(project_name, 0, "frame_time",0.0)
            st.experimental_rerun()
    else:
        
    
        if "project_settings" not in st.session_state:
            st.session_state['project_settings'] = get_project_settings(project_name)

        if 'index_of_last_model' not in st.session_state:
            st.session_state['index_of_last_model'] = 0
                                                    
        if "strength" not in st.session_state:                    
            st.session_state['strength'] = st.session_state['project_settings']["last_strength"]
            st.session_state['prompt_value'] = st.session_state['project_settings']["last_prompt"]
            st.session_state['model'] = st.session_state['project_settings']["last_model"]
            st.session_state['custom_pipeline'] = st.session_state['project_settings']["last_custom_pipeline"]
            st.session_state['negative_prompt_value'] = st.session_state['project_settings']["last_negative_prompt"]
            st.session_state['guidance_scale'] = st.session_state['project_settings']["last_guidance_scale"]
            st.session_state['seed'] = st.session_state['project_settings']["last_seed"]
            st.session_state['num_inference_steps'] = st.session_state['project_settings']["last_num_inference_steps"]
            st.session_state['which_stage_to_run_on'] = st.session_state['project_settings']["last_which_stage_to_run_on"]
            st.session_state['show_comparison'] = "Don't show"
                                
        if "which_image" not in st.session_state:
            st.session_state['which_image'] = 0
                                    
        if 'frame_styling_view_type' not in st.session_state:
            st.session_state['frame_styling_view_type'] = "List View"
            st.session_state['frame_styling_view_type_index'] = 0


        with st.sidebar:

            st.session_state['which_image'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(timing_details)-1, value=st.session_state['which_image_value'], step=1, key="which_image_selector")
            if st.session_state['which_image_value'] != st.session_state['which_image']:
                st.session_state['which_image_value'] = st.session_state['which_image']
                st.session_state['reset_canvas'] = True
                st.experimental_rerun()       

            with st.expander("Notes:"):
                    
                notes = st.text_area("Frame Notes:", value=timing_details[st.session_state['which_image']]["notes"], height=100, key="notes")

            if notes != timing_details[st.session_state['which_image']]["notes"]:
                timing_details[st.session_state['which_image']]["notes"] = notes
                update_specific_timing_value(project_name, st.session_state['which_image'], "notes", notes)
                st.experimental_rerun()
            st.markdown("***")
            

        if timing_details == []:

            st.info("You need to select and load key frames first in the Key Frame Selection section.")                            
        
        else:

            top1, top2, top3 = st.columns([4,1,3])
            with top1:
                view_types = ["List View","Individual View"]
                st.session_state['frame_styling_view_type'] = st.radio("View type:", view_types, key="which_view_type", horizontal=True, index=st.session_state['frame_styling_view_type_index'])                        
                if view_types.index(st.session_state['frame_styling_view_type']) != st.session_state['frame_styling_view_type_index']:
                    st.session_state['frame_styling_view_type_index'] = view_types.index(st.session_state['frame_styling_view_type'])
                    st.experimental_rerun()
                                                                
            with top2:
                st.write("")

            project_settings = get_project_settings(project_name)

            if st.session_state['frame_styling_view_type'] == "Individual View":

                if "section_index" not in st.session_state:
                    st.session_state['section_index'] = 0

                sections = ["Guidance", "Styling", "Motion"]
                
                st.session_state['section'] = option_menu(None, sections, icons=['pencil', 'palette', "hourglass", 'stopwatch'], menu_icon="cast", default_index=st.session_state['section_index'], orientation="horizontal")

                if st.session_state['section_index'] != sections.index(st.session_state['section']):
                    st.session_state['section_index'] = sections.index(st.session_state['section'])
                    st.experimental_rerun()
                section = st.session_state['section']

                st.subheader(section)

                                                    
                                            
                
                if section == "Guidance":   
                    
                    guidance_types = ["Drawing", "Images", "Video"]
                    if 'how to guide_index' not in st.session_state:
                        if project_settings["guidance_type"] == "":
                            st.session_state['how_to_guide_index'] = 0
                        else:
                            st.session_state['how_to_guide_index'] = guidance_types.index(project_settings["guidance_type"])
                    how_to_guide = st.radio("How to guide:", guidance_types, key="how_to_guide", horizontal=True, index=st.session_state['how_to_guide_index'])
                    if guidance_types.index(how_to_guide) != st.session_state['how_to_guide_index']:
                        st.session_state['how_to_guide_index'] = guidance_types.index(how_to_guide)
                        update_project_setting("guidance_type", how_to_guide,project_name)                                    
                        st.experimental_rerun()

                    if how_to_guide == "Drawing":

                        canvas1, canvas2 = st.columns([1.25,3])

                        with canvas1:
                                                                                                        
                    
                            width = int(project_settings["width"])
                            height = int(project_settings["height"])
                            if timing_details[st.session_state['which_image']]["source_image"] != "":
                                if timing_details[st.session_state['which_image']]["source_image"] .startswith("http"):
                                    canvas_image = r.get(timing_details[st.session_state['which_image']]["source_image"] )
                                    canvas_image = Image.open(BytesIO(canvas_image.content))
                                else:
                                    canvas_image = Image.open(timing_details[st.session_state['which_image']]["source_image"] )             
                            else:
                                canvas_image = Image.new("RGB", (width, height), "white")
                            if 'drawing_input' not in st.session_state:
                                    st.session_state['drawing_input'] = 'Magic shapes 🪄'
                            col1, col2 = st.columns([6,3])
                                            
                            with col1:
                                st.session_state['drawing_input'] = st.radio(
                                    "Drawing tool:",
                                    ("Draw lines ✏️","Erase Lines ❌","Make shapes 🪄","Move shapes 🏋🏾‍♂️","Make Lines ║"), horizontal=True,
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
                                if st.session_state['drawing_input']  == "Draw lines ✏️" or st.session_state['drawing_input']  == "Make Lines ║":         
                                    stroke_width = st.slider("Stroke width: ", 1, 50, 2)
                                elif st.session_state['drawing_input']  == "Erase Lines ❌":
                                    stroke_width = st.slider("Stroke width: ", 1, 100, 25)
                                else:
                                    stroke_width = 3

                            
                            

                            if st.button("Clear Canny Image"):
                                update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", "")                    
                                st.session_state['reset_canvas'] = True                    
                                st.experimental_rerun()
                            
                            st.markdown("***")
                            back_and_forward_buttons(timing_details)
                            st.markdown("***")
                            what_degree = st.number_input("Rotate image by: ", 0, 360, 0)
                            what_zoom = st.number_input("Zoom image by: ", 0.1, 5.0, 1.0)
                            if st.button("Rotate Image"):                                
                                output_image = rotate_image(get_primary_variant_location(timing_details, st.session_state['which_image']),what_degree)
                                output_image.save("temp.png")
                                if what_zoom != 1.0:
                                    output_image = zoom_image("temp.png", what_zoom)
                                st.image(output_image, caption="Rotated image", use_column_width=True)
                        
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
                                    im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
                            else:
                                st.session_state['reset_canvas'] = False
                                canvas_result = st_canvas()       
                                time.sleep(0.1)         
                                st.experimental_rerun()

                        

                        if st.button("Save New Canny Image"):
                            if canvas_result.image_data is not None:
                                # overlay the canvas image on top of the canny image and save the result
                                # if canny image is from a url, then we need to download it first
                                if timing_details[st.session_state['which_image']]["source_image"] != "":
                                    if timing_details[st.session_state['which_image']]["source_image"].startswith("http"):
                                        canny_image = r.get(timing_details[st.session_state['which_image']]["source_image"])
                                        canny_image = Image.open(BytesIO(canny_image.content))
                                    else:
                                        canny_image = Image.open(timing_details[st.session_state['which_image']]["source_image"])
                                else:
                                    canny_image = Image.new("RGB", (width, height), "white")
                                canny_image = canny_image.convert("RGBA")
                                # canvas_image = canvas_image.convert("RGBA")                                            
                                canvas_image = im
                                canvas_image = canvas_image.convert("RGBA")
                                new_canny_image = Image.alpha_composite(canny_image, canvas_image)
                                new_canny_image = new_canny_image.convert("RGB")
                                unique_file_name = str(uuid.uuid4()) + ".png"
                                file_location = f"videos/{project_name}/assets/resources/masks/{unique_file_name}"
                                new_canny_image.save(file_location)
                                update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", file_location)
                                st.success("New Canny Image Saved")
                                st.session_state['reset_canvas'] = True
                                time.sleep(1)
                                st.experimental_rerun()  

                        st.markdown("***")
                        
                        
                        
                        canny1, canny2, canny3 = st.columns([1,1,1.1])
                        with canny1:
                            st.markdown("#### Use Image From Other Frame")
                            st.markdown("This will use a canny image from another frame. This will take a few seconds.") 
                            
                            if st.session_state['which_image'] == 0:
                                value = 0
                            else:
                                value = st.session_state['which_image'] - 1
                            which_number_image_for_canny = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(timing_details)-1, value=value, step=1,key="which_number_image_for_canny")
                            if st.button("Use Source Image From Other Frame"):
                                if timing_details[which_number_image_for_canny]["source_image"] != "":                             
                                    update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", timing_details[which_number_image_for_canny]["source_image"])                                                
                                    st.experimental_rerun()
                            if timing_details[which_number_image_for_canny]["source_image"] != "":
                                st.image(timing_details[which_number_image_for_canny]["source_image"]) 
                            else:
                                st.error("No Source Image Found")                 
                        with canny2:                                                            
                            st.markdown("#### Upload Source Image")
                            st.markdown("This will upload a canny image from your computer. This will take a few seconds.")
                            uploaded_file = st.file_uploader("Choose a file")
                            if st.button("Upload Source Image"):                                
                                with open(os.path.join(f"videos/{project_name}/assets/resources/masks",uploaded_file.name),"wb") as f:
                                    f.write(uploaded_file.getbuffer())                                                                                                                                                      
                                    st.success("Your file is uploaded")
                                    update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", f"videos/{project_name}/assets/resources/masks/{uploaded_file.name}")                               
                                    time.sleep(1.5)
                                    st.experimental_rerun()  
                        with canny3:
                            st.markdown("#### Extract Canny From image")
                            st.markdown("This will extract a canny image from the current image. This will take a few seconds.")
                            source_of_image = st.radio("Which image would you like to use?", ["Existing Frame", "Uploaded Image"])
                            if source_of_image == "Existing Frame":
                                which_frame = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(timing_details)-1, value=st.session_state['which_image'], step=1)
                                if timing_details[which_frame]["alternative_images"] != "":
                                    variants = timing_details[which_frame]["alternative_images"]                        
                                    current_variant = int(timing_details[which_frame]["primary_image"])     
                                    image_path = variants[current_variant]
                                    st.image(image_path)
                                else:
                                    st.error("No Image Found")

                            elif source_of_image == "Uploaded Image":
                                uploaded_image = st.file_uploader("Choose a file", key="uploaded_image")   
                                if uploaded_image is not None:
                                    # download image as temp.png
                                    with open("temp.png","wb") as f:
                                        f.write(uploaded_image.getbuffer())                                                                                                                                                      
                                        st.success("Your file is uploaded")
                                        uploaded_image = "temp.png"
                        

                            threshold1, threshold2 = st.columns([1,1])
                            with threshold1:
                                low_threshold = st.number_input("Low Threshold", min_value=0, max_value=255, value=100, step=1)
                            with threshold2:                    
                                high_threshold = st.number_input("High Threshold", min_value=0, max_value=255, value=200, step=1)
                                                                                                                                                        
                            if st.button("Extract Canny From image"):
                                if source_of_image == "Existing Frame":
                                    canny_image = extract_canny_lines(image_path, project_name,low_threshold, high_threshold)
                                elif source_of_image == "Uploaded Image":
                                    canny_image = extract_canny_lines(uploaded_image, project_name,low_threshold, high_threshold)
                                st.image(canny_image)
                                if st.button("Save Canny Image"):
                                    update_specific_timing_value(project_name, int(st.session_state['which_image']), "source_image", canny_image)
                                    st.session_state['reset_canvas'] = True                        
                                    st.experimental_rerun()
                # if current item is 0 

                    elif how_to_guide == "Images":
                        if timing_details[st.session_state['which_image']]["source_image"] != "":
                            st.image(timing_details[st.session_state['which_image']]["source_image"])
                        else:
                            st.error("No Source Image Found")


                        canny1, canny2, canny3 = st.columns([1,1,1.1])
                        with canny1:
                            st.markdown("#### Use Canny Image From Other Frame")
                            st.markdown("This will use a canny image from another frame. This will take a few seconds.") 
                            
                            if st.session_state['which_image'] == 0:
                                value = 0
                            else:
                                value = st.session_state['which_image'] - 1
                            which_number_image_for_canny = st.number_input("Which frame would you like to use?", min_value=0, max_value=len(timing_details)-1, value=value, step=1,key="which_number_image_for_canny")
                            if st.button("Use Source Image From Other Frame"):
                                if timing_details[which_number_image_for_canny]["source_image"] != "":                             
                                    update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", timing_details[which_number_image_for_canny]["source_image"])                                                
                                    st.experimental_rerun()
                            if timing_details[which_number_image_for_canny]["source_image"] != "":
                                st.image(timing_details[which_number_image_for_canny]["source_image"]) 
                            else:
                                st.error("No Source Image Found")                 
                        with canny2:                                                            
                            st.markdown("#### Upload Source Image")
                            st.markdown("This will upload a canny image from your computer. This will take a few seconds.")
                            uploaded_file = st.file_uploader("Choose a file")
                            if st.button("Upload Source Image"):                                
                                with open(os.path.join(f"videos/{project_name}/assets/resources/masks",uploaded_file.name),"wb") as f:
                                    f.write(uploaded_file.getbuffer())                                                                                                                                                      
                                    st.success("Your file is uploaded")
                                    update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", f"videos/{project_name}/assets/resources/masks/{uploaded_file.name}")                               
                                    time.sleep(1.5)
                                    st.experimental_rerun()  

                elif section == "Motion":
                                                                                

                    timing1, timing2 = st.columns([1,1])

                    with timing1:

                        if st.session_state['which_image'] == 0:
                            previous_frame_time = None
                            st.info("No Previous Frame Time")
                            current_frame_time = st.slider(f"Current Frame Time = {timing_details[st.session_state['which_image']]['frame_time']}",  value=timing_details[st.session_state['which_image']]['frame_time'], step=0.01,disabled=True)
                            next_frame_max_value = current_frame_time + 10.0                            
                            
                        else:
                            if st.session_state['which_image'] != 1:
                                previous_frame_time = st.slider(f"Previous Frame Time = {timing_details[st.session_state['which_image']-1]['frame_time']}", min_value=timing_details[st.session_state['which_image']-2]['frame_time'], max_value=timing_details[st.session_state['which_image']]['frame_time'], value=timing_details[st.session_state['which_image']-1]['frame_time'], step=0.01)
                            else:
                                previous_frame_time = st.slider(f"Previous Frame Time = {timing_details[st.session_state['which_image']-1]['frame_time']}", value = timing_details[st.session_state['which_image']-1]['frame_time'], step=0.01,disabled=True)  

                            # if it's the last frame then set the max value to the last frame time + 10
                            if st.session_state['which_image'] == len(timing_details)-1:
                                current_frame_max_value = timing_details[st.session_state['which_image']]['frame_time'] + 10.0
                            else:
                                current_frame_max_value = timing_details[st.session_state['which_image']+1]['frame_time']
                 
                            current_frame_time = st.slider(f"Current Frame Time = {timing_details[st.session_state['which_image']]['frame_time']}", 
                                                        min_value=timing_details[st.session_state['which_image']-1]['frame_time'], max_value=current_frame_max_value, value=timing_details[st.session_state['which_image']]['frame_time'], step=0.01)
                        
                        if st.session_state['which_image'] != len(timing_details)-1:
                            # if it's the second last frame then set the max value to the last frame time
                            if st.session_state['which_image'] == len(timing_details)-2:
                                next_frame_max_value = current_frame_time + 10.0
                            else:
                                next_frame_max_value = timing_details[st.session_state['which_image']+2]['frame_time']

                            # if it's the not the last frame, show the next frame time slider
                            if st.session_state['which_image'] != len(timing_details)-1:
                                next_frame_time = st.slider(f"Next Frame Time = {timing_details[st.session_state['which_image']+1]['frame_time']}", min_value=timing_details[st.session_state['which_image']]['frame_time'], max_value=next_frame_max_value, value=timing_details[st.session_state['which_image']+1]['frame_time'], step=0.01)
                            
                        else:
                            st.info("No Next Frame Time")
                            next_frame_time = None
                        
                        

                        
                        # if there are any differences between the saved times then show a warning
                        
                        if st.button("Update Frame Time"):
                            if previous_frame_time is not None:
                                update_specific_timing_value(project_name, st.session_state['which_image']-1, "frame_time", previous_frame_time)                            
                            update_specific_timing_value(project_name, st.session_state['which_image'], "frame_time", current_frame_time)
                            if next_frame_time is not None:
                                update_specific_timing_value(project_name, st.session_state['which_image']+1, "frame_time", next_frame_time)
                            st.success("Frame time updated")
                            time.sleep(0.3)
                            st.experimental_rerun()

                    with timing2:                    
            
                        # if st.button("Preview audio at this time"):
                        #  audio_bytes = get_audio_bytes_for_slice(project_name, st.session_state['which_image'])
                        # st.audio(audio_bytes, format='audio/wav')
                                                        
                        variants = timing_details[st.session_state['which_image']]["alternative_images"]
                        if timing_details[st.session_state['which_image']]["preview_video"] != "":
                            st.video(timing_details[st.session_state['which_image']]['preview_video'])                                 
                        else:
                            st.error("No preview video available for this frame")
                        if variants != [] and variants != None and variants != "":
                            if st.button("Generate New Preview Video"):                                     
                                if st.session_state['which_image'] == 0:
                                    preview_video = create_single_preview_video(st.session_state['which_image'],project_name)                                                                                                            
                                elif st.session_state['which_image'] == len(timing_details)-1:
                                    preview_video = create_single_preview_video(st.session_state['which_image']-1,project_name)
                                else:                       
                                    preview_video = create_full_preview_video(project_name, st.session_state['which_image'])                                    
                                update_specific_timing_value(project_name, st.session_state['which_image'], "preview_video", preview_video)                                    
                                st.experimental_rerun()   
                        back_and_forward_buttons(timing_details) 
                    
                    with st.expander("Animation style"):

                        animation1,animation2 = st.columns([1.5,1])

                        with animation1:

                            if project_settings["default_animation_style"] == "Direct Morphing":
                                index_of_animation_style = 1
                            else:
                                index_of_animation_style = 0

                            animation_style = st.radio("Which animation style would you like to use for this frame?", ["Interpolation", "Direct Morphing"], index=index_of_animation_style)

                            animationbutton1, animationbutton2 = st.columns([1,1])

                            with animationbutton1:

                                if animation_style != timing_details[st.session_state['which_image']]["animation_style"] and animation_style != project_settings["default_animation_style"]:

                                    if st.button("Update this slides animation style"):
                                        update_specific_timing_value(project_name, st.session_state['which_image'], "animation_style", animation_style)                            
                                        st.success("Animation style updated")                                
                                        update_specific_timing_value(project_name, st.session_state['which_image'], "interpolated_video", "")
                                        if project_settings["default_animation_style"] == "":
                                            update_project_setting("default_animation_style", animation_style,project_name)
                                        time.sleep(0.3)
                                        st.experimental_rerun()
                                else:
                                    st.info(f"{animation_style} is already the animation style for this frame.")
                            
                            with animationbutton2:
                                if animation_style != project_settings["default_animation_style"]:
                                    if st.button(f"Change default animation style to {animation_style}", help="This will change the default animation style - but won't affect current frames."):
                                        update_project_setting("default_animation_style", animation_style,project_name)    
                                    
                        with animation2:
                            
                            if animation_style == "Interpolation":
                                st.info("This will fill the gaps between the current frame and the next frame with interpolated frames. This will make the animation smoother but will take longer to render.")
                            elif animation_style == "Direct Morphing":
                                st.info("This will morph the current frame directly into the next frame. This will make the animation less smooth but can be used to nice effect.")

                            
                                                
                                
                                


                    
                    with st.expander("Clip speed adjustment"):
                        
                    

                        clip_data = []
                        start_pct = 0.0
                        total_duration = 0.0
                        st.subheader("Speed Adjustment")

                        while start_pct < 1.0:
                            st.info(f"##### Section {len(clip_data) + 1}")
                            end_pct = st.slider(f"What percentage of the original clip should section {len(clip_data) + 1} go until?", min_value=start_pct, max_value=1.0, value=1.0, step=0.01)

                            if end_pct == 1.0:
                                remaining_duration = 1.0 - total_duration
                                remaining_pct = 1.0 - start_pct
                                speed_change = remaining_pct / remaining_duration
                                st.write(f"Speed change for the last section will be set to **{speed_change:.2f}x** to maintain the original video length.")
                            else:
                                speed_change = st.slider(f"What speed change should be applied to section {len(clip_data) + 1}?", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

                            clip_data.append({
                                "start_pct": start_pct,
                                "end_pct": end_pct,
                                "speed_change": speed_change
                            })

                            original_duration = end_pct - start_pct
                            final_duration = original_duration / (speed_change + 1e-6)
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

                        
                                                                                                                
                                    
                                                                        
                                                        
                
                elif section == "Styling":

                    with top3:
                        comparison_values = ["None"]
                        if st.session_state['which_image'] != 0 and timing_details[st.session_state['which_image']]["source_image"] != "":                        
                            comparison_values.append("Source Frame")
                        if timing_details[st.session_state['which_image']-1]["alternative_images"] != "":
                            comparison_values.append("Previous Frame")                        
                        if len(timing_details) > st.session_state['which_image']+1:
                            if timing_details[st.session_state['which_image']+1]["alternative_images"] != "":
                                comparison_values.append("Next Frame")

                        st.session_state['show_comparison'] = st.radio("Show comparison to:", options=comparison_values, horizontal=True)
                        
                                    

                                        
                                                        
                    mainimages1, mainimages2 = st.columns([1.5,1])

                    
                    variants = timing_details[st.session_state['which_image']]["alternative_images"]

                    if variants != [] and variants != None and variants != "":
                                    
                        primary_variant_location = get_primary_variant_location(timing_details, st.session_state['which_image'])
                    

                    with mainimages1:
                    
                        if st.session_state['show_comparison'] == "None":
                            project_settings = get_project_settings(project_name)
                            if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                                st.image(primary_variant_location, use_column_width=True)   
                            else:
                                st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True)   
                        elif st.session_state['show_comparison'] == "Source Frame":
                            if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                                img2=primary_variant_location
                            else:
                                img2='https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'          
                            image_comparison(starting_position=50,
                                img1=timing_details[st.session_state['which_image']]["source_image"],
                                img2=img2,make_responsive=False)
                        elif st.session_state['show_comparison'] == "Previous Frame":
                            previous_image = get_primary_variant_location(timing_details, st.session_state['which_image']-1)
                            image_comparison(starting_position=50,
                                img1=previous_image,
                                img2=primary_variant_location,make_responsive=False)
                        elif st.session_state['show_comparison'] == "Next Frame":
                            next_image = get_primary_variant_location(timing_details, st.session_state['which_image']+1)
                            image_comparison(starting_position=50,
                                img1=primary_variant_location,
                                img2=next_image,make_responsive=False)

                        
                        elif st.session_state['show_comparison'] == "Previous Frame":
                            st.write("")
                            
                        detail1, detail2, detail3, detail4 = st.columns([2.5,2.5,3.5,2])

                        with detail1:
                            individual_number_of_variants = st.number_input(f"How many variants?", min_value=1, max_value=10, value=1, key=f"number_of_variants_{st.session_state['which_image']}")
                            
                            
                        with detail2:
                            st.write("")
                            st.write("")
                            
                            if st.button(f"Generate variants", key=f"new_variations_{st.session_state['which_image']}",help="This will generate new variants based on the settings to the left."):
                                for i in range(0, individual_number_of_variants):
                                    index_of_current_item = st.session_state['which_image']
                                    trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],st.session_state['custom_models'],st.session_state['adapter_type'], True) 
                                st.experimental_rerun()
                        with detail3:
                            st.write("")
                            st.write("")                    
                            if st.button(f"Re-run w/ saved settings", key=f"re_run_on_this_frame_{st.session_state['which_image']}",help="This will re-run the restyling process on this frame."):
                                index_of_current_item = st.session_state['which_image']
                                trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],st.session_state['custom_models'],st.session_state['adapter_type'],False)
                                st.experimental_rerun()
                        with detail4:
                            st.write("")
                            
                            st.write("")
                            
                    with mainimages2:

                        if timing_details[st.session_state['which_image']]["alternative_images"] != "":            

                            number_of_variants = len(variants)                                                                           

                            back_and_forward_buttons(timing_details)                                                        
                            current_variant = int(timing_details[st.session_state['which_image']]["primary_image"])             
                            which_variant = st.radio(f'Main variant = {current_variant}', range(number_of_variants), index=current_variant, horizontal = True, key = f"Main variant for {st.session_state['which_image']}")                        
                            st.image(variants[which_variant], use_column_width=True)

                            if which_variant == current_variant:   
                                st.write("")                                   
                                st.success("Main variant")
                                st.write("")                                                                                                        
                            else:
                                st.write("")
                                if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['which_image']}", help="Promote this variant to the primary image"):
                                    promote_image_variant(st.session_state['which_image'], project_name, which_variant)
                                    time.sleep(0.5)
                                    st.experimental_rerun()  
                                                        
                    with st.expander("Compare to previous and next images", expanded=True):                 
                        
                        img1, img2 = st.columns(2)
                        with img1:
                            # if it's the first image, don't show a previous image
                            if st.session_state['which_image'] != 0:
                                variants = timing_details[st.session_state['which_image']-1]["alternative_images"]
                                if variants != [] and variants != None and variants != "":
                                    previous_image = get_primary_variant_location(timing_details, st.session_state['which_image']-1)                        
                                    st.image(previous_image, use_column_width=True, caption=f"Previous image")                        
                                else:
                                    st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True, caption=f"Previous image")
                            else:
                                st.write("")
                        with img2:
                            # if it's the last image, don't show a next image
                            if st.session_state['which_image'] != len(timing_details)-1:
                                variants = timing_details[st.session_state['which_image']+1]["alternative_images"]
                                if variants != [] and variants != None and variants != "":                            
                                    next_image = get_primary_variant_location(timing_details, st.session_state['which_image']+1)                    
                                    st.image(next_image, use_column_width=True, caption=f"Next image")
                                else:
                                    st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True, caption=f"Next image")
                            else:
                                st.write("")




                    with st.expander("🛠️ Saved Settings"):
                        st.info("These are the settings that will be used when you click 'Re-run w/ saved settings'.")
                        updated_prompt = st.text_area("Prompt:", value = timing_details[st.session_state['which_image']]["prompt"], height=100)
                        if st.button("Save prompt"):
                            timing_details[st.session_state['which_image']]["prompt"] = updated_prompt
                            update_specific_timing_value(project_name, st.session_state['which_image'], "prompt", updated_prompt)
                            st.experimental_rerun()
                        
                    

                        

                    

                    with st.expander("Replace Frame"):
                    
                        replace_with = st.radio("Replace with:", ["Uploaded Frame","Previous Frame"], horizontal=True)
                        replace1, replace2, replace3 = st.columns([2,1,1])                    

                        if replace_with == "Previous Frame":  
                            with replace1:
                                which_stage_to_use_for_replacement = st.radio("Select stage to use:", ["Styled Key Frame","Unedited Key Frame"],key="which_stage_to_use_for_replacement", horizontal=True)
                                which_image_to_use_for_replacement = st.number_input("Select image to use:", min_value=0, max_value=len(timing_details)-1, value=0, key="which_image_to_use_for_replacement")
                                if which_stage_to_use_for_replacement == "Unedited Key Frame":                                    
                                    background_image = timing_details[which_image_to_use_for_replacement]["source_image"]                            
                                elif which_stage_to_use_for_replacement == "Styled Key Frame":
                                    variants = timing_details[which_image_to_use_for_replacement]["alternative_images"]
                                    primary_image = timing_details[which_image_to_use_for_replacement]["primary_image"]             
                                    background_image = variants[primary_image]
                                if st.button("Replace with selected frame",disabled=False):
                                    if st.session_state['which_stage'] == "Unedited Key Frame":
                                        update_specific_timing_value(project_name, st.session_state['which_image'], "source_image",background_image)                                                                                   
                                    elif st.session_state['which_stage'] == "Styled Key Frame":
                                        number_of_image_variants = add_image_variant(background_image, st.session_state['which_image'], project_name, timing_details)
                                        promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1) 
                                    st.success("Replaced")
                                    time.sleep(1)     
                                    st.experimental_rerun()
                            with replace2:
                                st.image(background_image, width=300)       
                                                                                                                                                                    
                        elif replace_with == "Uploaded Frame":
                            with replace1:
                                replacement_frame = st.file_uploader("Upload a replacement frame here", type="png", accept_multiple_files=False, key="replacement_frame")                                                
                            with replace2:                                                        
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                if st.button("Replace frame",disabled=False):
                                    images_for_model = []                    
                                    with open(os.path.join(f"videos/{project_name}/",replacement_frame.name),"wb") as f: 
                                        f.write(replacement_frame.getbuffer())     
                                    uploaded_image = upload_image(f"videos/{project_name}/{replacement_frame.name}")
                                    
                                    number_of_image_variants = add_image_variant(uploaded_image, st.session_state['which_image'], project_name, timing_details)
                                    promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1) 
                                    # delete the uploaded file
                                    os.remove(f"videos/{project_name}/{replacement_frame.name}")
                                    st.success("Replaced")
                                    time.sleep(1)     
                                    st.experimental_rerun()  


                st.markdown("***")
                extra_settings_1, extra_settings_2 = st.columns([1,1])

                with extra_settings_1:

                    with st.expander("Add Key Frame", expanded=True):

                        how_long_after = st.slider("How long after?", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                    
                        if st.button(f"Add key frame after #{st.session_state['which_image']}"):
                            index_of_current_item = st.session_state['which_image']
                            
                            timing_details = get_timing_details(project_name)
                            # if it's the last frame, then add a second to the frame time. If not, then add half the time between the current frame and the next frame.
                            if len(timing_details) == 1:
                                key_frame_time = 0.0
                            elif index_of_current_item == len(timing_details) - 1:
                                key_frame_time = float(timing_details[index_of_current_item]["frame_time"]) + 1.0
                            else:
                                st.write(timing_details[index_of_current_item]["frame_time"])
                                st.write(index_of_current_item)
                                st.write(timing_details[index_of_current_item + 1]["frame_time"])
                                st.write(index_of_current_item + 1)
                                key_frame_time = (float(timing_details[index_of_current_item]["frame_time"]) + float(timing_details[index_of_current_item + 1]["frame_time"])) / 2.0
                            create_timings_row_at_frame_number(project_name, index_of_current_item +1)
                            update_specific_timing_value(project_name, st.session_state['which_image'] + 1, "frame_time", st.session_state['which_image']['frame_time'] + how_long_after)
                            
                            timing_details = get_timing_details(project_name)                    
                            st.session_state['which_image_value'] = st.session_state['which_image_value'] + 1                
                            st.experimental_rerun()
                with extra_settings_2:
                    
                    with st.expander("Delete Key Frame", expanded=True):
                        confirm_delete = st.checkbox("Confirm deletion")
                        if confirm_delete == True:
                            if st.button("Delete key frame"):
                                index_of_current_item = st.session_state['which_image']
                                delete_frame(project_name, index_of_current_item)                
                                timing_details = get_timing_details(project_name)
                                st.experimental_rerun()
                        else:
                            st.button("Delete key frame", disabled=True)



            elif st.session_state['frame_styling_view_type'] == "List View":
                for i in range(0, len(timing_details)):
                    index_of_current_item = i
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.subheader(f"Frame {i}")
                    col2.empty()
                    with col3:
                        if st.button("Delete this keyframe", key=f'{index_of_current_item}'):
                            delete_frame(project_name, index_of_current_item)
                            timing_details = get_timing_details(project_name)
                            st.experimental_rerun()           
                                        
                    if timing_details[i]["alternative_images"] != "":
                        variants = timing_details[i]["alternative_images"]
                        current_variant = int(timing_details[i]["primary_image"])    
                        st.image(variants[current_variant])                            
                    else:
                        st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True) 
                    

                    detail1, detail2, detail3, detail4 = st.columns([2,2,1,3])

                    with detail1:
                        individual_number_of_variants = st.number_input(f"How many variants?", min_value=1, max_value=10, value=1, key=f"number_of_variants_{index_of_current_item}")
                        

                        
                    with detail2:
                        st.write("")
                        st.write("")
                        if st.button(f"Generate variants", key=f"new_variations_{index_of_current_item}",help="This will generate new variants based on the settings to the left."):
                            for a in range(0, individual_number_of_variants):
                                index_of_current_item = i
                                trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],st.session_state['custom_models'],st.session_state['adapter_type'], True)                             
                            st.experimental_rerun()
                        
                        
                    with detail3:
                        st.write("")
                    with detail4:
                        if st.button(f"Jump to single frame view for #{index_of_current_item}", help="This will switch to a Single Frame view type and open this individual image."):
                            st.session_state['which_image_value'] = index_of_current_item
                            st.session_state['frame_styling_view_type'] = "Individual View"
                            st.session_state['frame_styling_view_type_index'] = 1                                    
                            st.experimental_rerun() 
                
            
        
                                                

                                                
                    
            with st.sidebar:
                styling_sidebar(project_name, timing_details)
            



            
        

            '''
            st.sidebar.header("Restyle Frames")   

            
            if 'index_of_which_stage_to_run_on' not in st.session_state:                        
                st.session_state['index_of_which_stage_to_run_on'] = 0
            st.write(f"Current frame: {st.session_state['which_image']}")
            smallbutton1, smallbutton2,smallbutton3, smallbutton4 = st.sidebar.columns([1,1,1,4])
            with smallbutton1:
                # if it's not the first image
                if st.session_state['which_image'] != 0:
                    if st.button(f"{st.session_state['which_image']-1} ⏪", key=f"Previous Image for {st.session_state['which_image']}"):
                        st.session_state['which_image_value'] = st.session_state['which_image_value'] - 1
                        st.experimental_rerun()
                # number of frame
                
            with smallbutton2:
                st.button(f"{st.session_state['which_image']} 📍",disabled=True)
            with smallbutton3:
                # if it's not the last image
                if st.session_state['which_image'] != len(timing_details)-1:
                    if st.button(f"{st.session_state['which_image']+1} ⏩", key=f"Next Image for {st.session_state['which_image']}"):
                        st.session_state['which_image_value'] = st.session_state['which_image_value'] + 1
                        st.experimental_rerun()
                
            stages = ["Extracted Key Frames", "Current Main Variants"]
            st.session_state['which_stage_to_run_on'] = st.sidebar.radio("What stage of images would you like to run styling on?", options=stages, horizontal=True, index =st.session_state['index_of_which_stage_to_run_on'] , help="Extracted frames means the original frames from the video.")                                                                                     
            if stages.index(st.session_state['which_stage_to_run_on']) != st.session_state['index_of_which_stage_to_run_on']:
                st.session_state['index_of_which_stage_to_run_on'] = stages.index(st.session_state['which_stage_to_run_on'])
                st.experimental_rerun()

            custom_pipelines = ["None","Mystique"]                   
            if 'index_of_last_custom_pipeline' not in st.session_state:
                st.session_state['index_of_last_custom_pipeline'] = 0        
            st.session_state['custom_pipeline'] = st.sidebar.selectbox(f"Custom Pipeline:", custom_pipelines, index=st.session_state['index_of_last_custom_pipeline'])
            if custom_pipelines.index(st.session_state['custom_pipeline']) != st.session_state['index_of_last_custom_pipeline']:
                st.session_state['index_of_last_custom_pipeline'] = custom_pipelines.index(st.session_state['custom_pipeline'])
                st.experimental_rerun()

            if st.session_state['custom_pipeline'] == "Mystique":
                if st.session_state['index_of_last_model'] > 1:
                    st.session_state['index_of_last_model'] = 0       
                    st.experimental_rerun()           
                with st.sidebar.expander("Mystique is a custom pipeline that uses a multiple models to generate a consistent character and style transformation."):
                    st.markdown("## How to use the Mystique pipeline")                
                    st.markdown("1. Create a fine-tined model in the Custom Model section of the app - we recommend Dreambooth for character transformations.")
                    st.markdown("2. It's best to include a detailed prompt. We recommend taking an example input image and running it through the Prompt Finder")
                    st.markdown("3. Use [expression], [location], [mouth], and [looking] tags to vary the expression and location of the character dynamically if that changes throughout the clip. Varying this in the prompt will make the character look more natural - especially useful if the character is speaking.")
                    st.markdown("4. In our experience, the best strength for coherent character transformations is 0.25-0.3 - any more than this and details like eye position change.")  
                models = ["LoRA","Dreambooth"]                                     
                st.session_state['model'] = st.sidebar.selectbox(f"Which type of model is trained on your character?", models, index=st.session_state['index_of_last_model'])                    
                if st.session_state['index_of_last_model'] != models.index(st.session_state['model']):
                    st.session_state['index_of_last_model'] = models.index(st.session_state['model'])
                    st.experimental_rerun()                          
            else:
                models = ['controlnet','stable-diffusion-img2img-v2.1', 'depth2img', 'pix2pix', 'Dreambooth', 'LoRA','StyleGAN-NADA']            
                st.session_state['model'] = st.sidebar.selectbox(f"Which model would you like to use?", models, index=st.session_state['index_of_last_model'])                    
                if st.session_state['index_of_last_model'] != models.index(st.session_state['model']):
                    st.session_state['index_of_last_model'] = models.index(st.session_state['model'])
                    st.experimental_rerun() 
                    
            
            if st.session_state['model'] == "controlnet":   
                controlnet_adapter_types = ["scribble","normal", "canny", "hed", "seg", "hough", "depth2img", "pose"]
                if 'index_of_controlnet_adapter_type' not in st.session_state:
                    st.session_state['index_of_controlnet_adapter_type'] = 0
                st.session_state['adapter_type'] = st.sidebar.selectbox(f"Adapter Type",controlnet_adapter_types, index=st.session_state['index_of_controlnet_adapter_type'])
                if st.session_state['index_of_controlnet_adapter_type'] != controlnet_adapter_types.index(st.session_state['adapter_type']):
                    st.session_state['index_of_controlnet_adapter_type'] = controlnet_adapter_types.index(st.session_state['adapter_type'])
                    st.experimental_rerun()
                custom_models = []    
                
            elif st.session_state['model'] == "LoRA": 
                if 'index_of_lora_model_1' not in st.session_state:
                    st.session_state['index_of_lora_model_1'] = 0
                    st.session_state['index_of_lora_model_2'] = 0
                    st.session_state['index_of_lora_model_3'] = 0
                df = pd.read_csv('models.csv')
                filtered_df = df[df.iloc[:, 5] == 'LoRA']
                lora_model_list = filtered_df.iloc[:, 0].tolist()
                lora_model_list.insert(0, '')
                st.session_state['lora_model_1'] = st.sidebar.selectbox(f"LoRA Model 1", lora_model_list, index=st.session_state['index_of_lora_model_1'])
                if st.session_state['index_of_lora_model_1'] != lora_model_list.index(st.session_state['lora_model_1']):
                    st.session_state['index_of_lora_model_1'] = lora_model_list.index(st.session_state['lora_model_1'])
                    st.experimental_rerun()
                st.session_state['lora_model_2'] = st.sidebar.selectbox(f"LoRA Model 2", lora_model_list, index=st.session_state['index_of_lora_model_2'])
                if st.session_state['index_of_lora_model_2'] != lora_model_list.index(st.session_state['lora_model_2']):
                    st.session_state['index_of_lora_model_2'] = lora_model_list.index(st.session_state['lora_model_2'])
                    st.experimental_rerun()
                st.session_state['lora_model_3'] = st.sidebar.selectbox(f"LoRA Model 3", lora_model_list, index=st.session_state['index_of_lora_model_3'])
                if st.session_state['index_of_lora_model_3'] != lora_model_list.index(st.session_state['lora_model_3']):
                    st.session_state['index_of_lora_model_3'] = lora_model_list.index(st.session_state['lora_model_3'])                     
                    st.experimental_rerun()
                custom_models = [st.session_state['lora_model_1'], st.session_state['lora_model_2'], st.session_state['lora_model_3']]                    
                st.sidebar.info("You can reference each model in your prompt using the following keywords: <1>, <2>, <3> - for example '<1> in the style of <2>.")
                lora_adapter_types = ['sketch', 'seg', 'keypose', 'depth', None]
                if "index_of_lora_adapter_type" not in st.session_state:
                    st.session_state['index_of_lora_adapter_type'] = 0
                st.session_state['adapter_type'] = st.sidebar.selectbox(f"Adapter Type:", lora_adapter_types, help="This is the method through the model will infer the shape of the object. ", index=st.session_state['index_of_lora_adapter_type'])
                if st.session_state['index_of_lora_adapter_type'] != lora_adapter_types.index(st.session_state['adapter_type']):
                    st.session_state['index_of_lora_adapter_type'] = lora_adapter_types.index(st.session_state['adapter_type'])
            elif st.session_state['model'] == "Dreambooth":
                df = pd.read_csv('models.csv')
                filtered_df = df[df.iloc[:, 5] == 'Dreambooth']
                dreambooth_model_list = filtered_df.iloc[:, 0].tolist()
                if 'index_of_dreambooth_model' not in st.session_state:
                    st.session_state['index_of_dreambooth_model'] = 0
                custom_models = st.sidebar.selectbox(f"Dreambooth Model", dreambooth_model_list, index=st.session_state['index_of_dreambooth_model'])
                if st.session_state['index_of_dreambooth_model'] != dreambooth_model_list.index(custom_models):
                    st.session_state['index_of_dreambooth_model'] = dreambooth_model_list.index(custom_models)                                    
            else:
                custom_models = []
                st.session_state['adapter_type'] = "N"
            
            if st.session_state['model'] == "StyleGAN-NADA":
                st.sidebar.warning("StyleGAN-NADA is a custom model that uses StyleGAN to generate a consistent character and style transformation. It only works for square images.")
                st.session_state['prompt'] = st.sidebar.selectbox("What style would you like to apply to the character?", ['base', 'mona_lisa', 'modigliani', 'cubism', 'elf', 'sketch_hq', 'thomas', 'thanos', 'simpson', 'witcher', 'edvard_munch', 'ukiyoe', 'botero', 'shrek', 'joker', 'pixar', 'zombie', 'werewolf', 'groot', 'ssj', 'rick_morty_cartoon', 'anime', 'white_walker', 'zuckerberg', 'disney_princess', 'all', 'list'])
                st.session_state['strength'] = 0.5
                st.session_state['guidance_scale'] = 7.5
                st.session_state['seed'] = int(0)
                st.session_state['num_inference_steps'] = int(50)
                            
            else:
                st.session_state['prompt'] = st.sidebar.text_area(f"Prompt", label_visibility="visible", value=st.session_state['prompt_value'],height=150)
                if st.session_state['prompt'] != st.session_state['prompt_value']:
                    st.session_state['prompt_value'] = st.session_state['prompt']
                    st.experimental_rerun()
                with st.sidebar.expander("💡 Learn about dynamic prompting"):
                    st.markdown("## Why and how to use dynamic prompting")
                    st.markdown("Why:")
                    st.markdown("Dynamic prompting allows you to automatically vary the prompt throughout the clip based on changing features in the source image. This makes the output match the input more closely and makes character transformations look more natural.")
                    st.markdown("How:")
                    st.markdown("You can include the following tags in the prompt to vary the prompt dynamically: [expression], [location], [mouth], and [looking]")
                if st.session_state['model'] == "Dreambooth":
                    model_details = get_model_details(custom_models)
                    st.sidebar.info(f"Must include '{model_details['keyword']}' to run this model")   
                    if model_details['controller_type'] != "":                    
                        st.session_state['adapter_type']  = st.sidebar.selectbox(f"Would you like to use the {model_details['controller_type']} controller?", ['Yes', 'No'])
                    else:
                        st.session_state['adapter_type']  = "No"

                else:
                    if st.session_state['model'] == "pix2pix":
                        st.sidebar.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")                    
                st.session_state['strength'] = st.sidebar.number_input(f"Strength", value=float(st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)
                
                with st.sidebar.expander("Advanced settings 😏"):
                    st.session_state['negative_prompt'] = st.text_area(f"Negative prompt", value=st.session_state['negative_prompt_value'], label_visibility="visible")
                    if st.session_state['negative_prompt'] != st.session_state['negative_prompt_value']:
                        st.session_state['negative_prompt_value'] = st.session_state['negative_prompt']
                        st.experimental_rerun()
                    st.session_state['guidance_scale'] = st.number_input(f"Guidance scale", value=float(st.session_state['guidance_scale']))
                    st.session_state['seed'] = st.number_input(f"Seed", value=int(st.session_state['seed']))
                    st.session_state['num_inference_steps'] = st.number_input(f"Inference steps", value=int(st.session_state['num_inference_steps']))
                                
            batch_run_range = st.sidebar.slider("Select range:", 1, 0, (0, len(timing_details)-1))  
            
            st.session_state["promote_new_generation"] = True                    
            st.session_state["promote_new_generation"] = st.sidebar.checkbox("Promote new generation to main variant", value=True, key="promote_new_generation_to_main_variant")
            st.session_state["use_new_settings"] = st.sidebar.checkbox("Use new settings for batch query", value=True, key="keep_existing_settings", help="If unchecked, the new settings will be applied to the existing variants.")

            app_settings = get_app_settings()

            if 'restyle_button' not in st.session_state:
                st.session_state['restyle_button'] = ''
                st.session_state['item_to_restyle'] = ''                

            btn1, btn2 = st.sidebar.columns(2)

            with btn1:
                batch_number_of_variants = st.number_input("How many variants?", value=1, min_value=1, max_value=10, step=1, key="number_of_variants")
            

            with btn2:

                st.write("")
                st.write("")
                if st.button(f'Batch restyle') or st.session_state['restyle_button'] == 'yes':
                                    
                    if st.session_state['restyle_button'] == 'yes':
                        range_start = int(st.session_state['item_to_restyle'])
                        range_end = range_start + 1
                        st.session_state['restyle_button'] = ''
                        st.session_state['item_to_restyle'] = ''

                    for i in range(batch_run_range[1]+1):
                        for number in range(0, batch_number_of_variants):
                            index_of_current_item = i
                            trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type'],st.session_state["use_new_settings"])
                    st.experimental_rerun()

            '''

            
            
                    
                    
                                    

            
                        

            
    