from io import BytesIO
import os
import time
import streamlit as st
from PIL import Image
import requests as r
from streamlit_drawable_canvas import st_canvas
from streamlit_image_comparison import image_comparison

from repository.local_repo.csv_repo import get_project_settings
from ui_components.common_methods import add_image_variant, create_or_update_mask, execute_image_edit, extract_frame, get_timing_details, promote_image_variant, update_source_image
from utils.file_upload.s3 import upload_image

def frame_editing_page(project_name):
    # if 0_extract folder is empty, show error

    if len(timing_details) == 0:
        st.info("You need to add  key frames first in the Key Frame Selection section.")

    else:

        timing_details = get_timing_details(project_name)
        project_settings = get_project_settings(project_name)

        #initiative value
        if "which_image" not in st.session_state:
            st.session_state['which_image'] = 0
        
        def reset_new_image():
            st.session_state['edited_image'] = ""

        if "which_stage" not in st.session_state:
            st.session_state['which_stage'] = "Unedited Key Frame"
            st.session_state['which_stage_index'] = 0
    
            
        f1, f2, f3 = st.columns([1,2,1])
        with f1:
            st.session_state['which_image'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(timing_details)-1, on_change=reset_new_image, value=st.session_state['which_image_value'])
            if st.session_state['which_image_value'] != st.session_state['which_image']:
                st.session_state['which_image_value'] = st.session_state['which_image']
                st.experimental_rerun()
        with f2:                
            st.session_state['which_stage'] = st.radio('Select stage:', ["Unedited Key Frame", "Styled Key Frame"], horizontal=True, on_change=reset_new_image, index=st.session_state['which_stage_index'])
            if st.session_state['which_stage'] == "Styled Key Frame" and st.session_state['which_stage_index'] == 0:
                st.session_state['which_stage_index'] = 1
                st.experimental_rerun()

        with f3:                
            if st.session_state['which_stage'] == "Unedited Key Frame":     
                st.write("")                     
                if st.button("Reset Key Frame", help="This will reset the base key frame to the original unedited version. This will not affect the video."):
                    extract_frame(int(st.session_state['which_image']), project_name, project_settings["input_video"], timing_details[st.session_state['which_image']]["frame_number"],timing_details)                            
                    st.experimental_rerun()
                            
        if "edited_image" not in st.session_state:
            st.session_state.edited_image = ""                        
        
        if st.session_state['which_stage'] == "Styled Key Frame" and timing_details[st.session_state['which_image']]["alternative_images"] == "":
            st.info("You need to add a style first in the Style Selection section.")
        else:

            if st.session_state['which_stage'] == "Unedited Key Frame":
                editing_image = timing_details[st.session_state['which_image']]["source_image"]
            elif st.session_state['which_stage'] == "Styled Key Frame":                                             
                variants = timing_details[st.session_state['which_image']]["alternative_images"]
                primary_image = timing_details[st.session_state['which_image']]["primary_image"]             
                editing_image = variants[primary_image]
    
            width = int(project_settings["width"])
            height = int(project_settings["height"])

            
            st.sidebar.markdown("### Select Area To Edit:") 
            if 'index_of_type_of_mask_selection' not in st.session_state:
                st.session_state['index_of_type_of_mask_selection'] = 0
            mask_selection_options = ["Automated Background Selection", "Automated Layer Selection", "Manual Background Selection","Re-Use Previous Mask"]
            type_of_mask_selection = st.sidebar.radio("How would you like to select what to edit?", mask_selection_options, horizontal=True, index=st.session_state['index_of_type_of_mask_selection'])                                                                      
            if st.session_state['index_of_type_of_mask_selection'] != mask_selection_options.index(type_of_mask_selection):
                st.session_state['index_of_type_of_mask_selection'] = mask_selection_options.index(type_of_mask_selection)
                st.experimental_rerun()
        
            if "which_layer" not in st.session_state:
                st.session_state['which_layer'] = "Background"

            if type_of_mask_selection == "Automated Layer Selection":
                st.session_state['which_layer'] = st.sidebar.selectbox("Which layer would you like to replace?", ["Background", "Middleground", "Foreground"])


            if type_of_mask_selection == "Manual Background Selection":
                if st.session_state['edited_image'] == "":                                
                    if editing_image.startswith("http"):
                        canvas_image = r.get(editing_image)
                        canvas_image = Image.open(BytesIO(canvas_image.content))
                    else:
                        canvas_image = Image.open(editing_image)
                    if 'drawing_input' not in st.session_state:
                        st.session_state['drawing_input'] = 'Magic shapes 🪄'
                    col1, col2 = st.columns([6,3])
                                    
                    with col1:
                        st.session_state['drawing_input'] = st.sidebar.radio(
                            "Drawing tool:",
                            ("Make shapes 🪄","Move shapes 🏋🏾‍♂️", "Draw lines ✏️"), horizontal=True,
                        )
                    
                    if st.session_state['drawing_input'] == "Move shapes 🏋🏾‍♂️":
                        drawing_mode = "transform"
                        st.sidebar.info("To delete something, just move it outside of the image! 🥴")
                    elif st.session_state['drawing_input'] == "Make shapes 🪄":
                        drawing_mode = "polygon"
                        st.sidebar.info("To end a shape, right click!")
                    elif st.session_state['drawing_input'] == "Draw lines ✏️":
                        drawing_mode = "freedraw"
                        st.sidebar.info("To draw, draw! ")
                    
                    with col2:    
                        if drawing_mode == "freedraw":           
                            stroke_width = st.slider("Stroke width: ", 1, 25, 12)
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
                        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
                        create_or_update_mask(project_name, st.session_state['which_image'], im)
                else:
                    image_comparison(
                        img1=editing_image,
                        img2=st.session_state['edited_image'], starting_position=5, label1="Original", label2="Edited")  
                    if st.button("Reset Canvas"):
                        st.session_state['edited_image'] = ""
                        st.experimental_rerun()
            
            elif type_of_mask_selection == "Automated Background Selection" or type_of_mask_selection == "Automated Layer Selection" or type_of_mask_selection == "Re-Use Previous Mask":
                if type_of_mask_selection == "Re-Use Previous Mask" and timing_details[st.session_state['which_image']]["mask"] == "":
                    st.sidebar.info("You don't have a previous mask to re-use.")
                if st.session_state['edited_image'] == "":
                    st.image(editing_image, use_column_width=True)
                else:
                    image_comparison(
                        img1=editing_image,
                        img2=st.session_state['edited_image'], starting_position=5, label1="Original", label2="Edited") 
                    if st.button("Reset Canvas"):
                        st.session_state['edited_image'] = ""
                        st.experimental_rerun()                    
                    
                                        

            st.sidebar.markdown("### Edit Individual Image:") 

            if "type_of_mask_replacement" not in st.session_state:
                st.session_state["type_of_mask_replacement"] = "Replace With Image"
                st.session_state["index_of_type_of_mask_replacement"] = 0
            
            types_of_mask_replacement = ["Replace With Image", "Inpainting"]
            st.session_state["type_of_mask_replacement"] = st.sidebar.radio("Select type of edit", types_of_mask_replacement, horizontal=True, index=st.session_state["index_of_type_of_mask_replacement"])    


            if st.session_state["index_of_type_of_mask_replacement"] != types_of_mask_replacement.index(st.session_state["type_of_mask_replacement"]):
                st.session_state["index_of_type_of_mask_replacement"] = types_of_mask_replacement.index(st.session_state["type_of_mask_replacement"])
                st.experimental_rerun()

            if st.session_state["type_of_mask_replacement"] == "Replace With Image":
                prompt = ""
                negative_prompt = ""
                background_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/backgrounds') if f.endswith('.png')]                 
                background_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/backgrounds') if f.endswith('.png')]                 
                sources_of_images = ["Uploaded", "From Other Frame"]
                if 'index_of_source_of_image' not in st.session_state:
                    st.session_state['index_of_source_of_image'] = 0
                source_of_image = st.sidebar.radio("Select type of image", sources_of_images,horizontal=True, index=st.session_state['index_of_source_of_image'])

                if st.session_state['index_of_source_of_image'] != sources_of_images.index(source_of_image):
                    st.session_state['index_of_source_of_image'] = sources_of_images.index(source_of_image)
                    st.experimental_rerun()

                if source_of_image == "Uploaded":                                
                    btn1, btn2 = st.sidebar.columns([1,1])
                    with btn1:
                        uploaded_files = st.file_uploader("Add more background images here", accept_multiple_files=True)                    
                        if st.button("Upload Backgrounds"):                            
                            for uploaded_file in uploaded_files:
                                with open(os.path.join(f"videos/{project_name}/assets/resources/backgrounds",uploaded_file.name),"wb") as f: 
                                    f.write(uploaded_file.getbuffer())                                                                                                                                                      
                                    st.success("Your backgrounds are uploaded file - they should appear in the dropdown.")                     
                                    background_list.append(uploaded_file.name)
                                    time.sleep(1.5)
                                    st.experimental_rerun()                                
                    with btn2:
                        background_image = st.sidebar.selectbox("Range background", background_list)
                        if background_list != []:
                            st.image(f"videos/{project_name}/assets/resources/backgrounds/{background_image}", use_column_width=True)
                elif source_of_image == "From Other Frame":
                    btn1, btn2 = st.sidebar.columns([1,1])
                    with btn1:
                        which_stage_to_use = st.radio("Select stage to use:", ["Unedited Key Frame", "Styled Key Frame"])
                        which_image_to_use = st.number_input("Select image to use:", min_value=0, max_value=len(timing_details)-1, value=0)
                        if which_stage_to_use == "Unedited Key Frame":                                    
                            background_image = timing_details[which_image_to_use]["source_image"]
                        elif which_stage_to_use == "Styled Key Frame":
                            variants = timing_details[which_image_to_use]["alternative_images"]
                            primary_image = timing_details[which_image_to_use]["primary_image"]             
                            background_image = variants[primary_image]
                    with btn2:
                        st.image(background_image, use_column_width=True)
                    

            elif st.session_state["type_of_mask_replacement"] == "Inpainting":
                btn1, btn2 = st.sidebar.columns([1,1])
                with btn1:
                    prompt = st.text_input("Prompt:", help="Describe the whole image, but focus on the details you want changed!")
                with btn2:
                    negative_prompt = st.text_input("Negative Prompt:", help="Enter any things you want to make the model avoid!")

            edit1, edit2 = st.sidebar.columns(2)

            with edit1:
                if st.button(f'Run Edit On Current Image'):
                    if st.session_state["type_of_mask_replacement"] == "Inpainting":
                        st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, "", editing_image, prompt, negative_prompt,width, height,st.session_state['which_layer'], st.session_state['which_image'])
                    elif st.session_state["type_of_mask_replacement"] == "Replace With Image":
                        st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, background_image, editing_image, "", "",width, height,st.session_state['which_layer'], st.session_state['which_image'])
                    st.experimental_rerun()
            with edit2:
                if st.session_state['edited_image'] != "":                                     
                    if st.button("Promote Last Edit", type="primary"):
                        if st.session_state['which_stage'] == "Unedited Key Frame":                        
                            update_source_image(project_name, st.session_state['which_image'], st.session_state['edited_image'])
                        elif st.session_state['which_stage'] == "Styled Key Frame":
                            number_of_image_variants = add_image_variant(st.session_state['edited_image'], st.session_state['which_image'], project_name, timing_details)
                            promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1)
                        st.session_state['edited_image'] = ""
                        st.success("Image promoted!")
                else:
                    if st.button("Run Edit & Promote"):
                        if st.session_state["type_of_mask_replacement"] == "Inpainting":
                            st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, "", editing_image, prompt, negative_prompt,width, height,st.session_state['which_layer'], st.session_state['which_image'])
                        elif st.session_state["type_of_mask_replacement"] == "Replace With Image":
                            st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, background_image, editing_image, "", "",width, height,st.session_state['which_layer'], st.session_state['which_image'])
                        if st.session_state['which_stage'] == "Unedited Key Frame":                        
                            update_source_image(project_name, st.session_state['which_image'], st.session_state['edited_image'])
                        elif st.session_state['which_stage'] == "Styled Key Frame":
                            number_of_image_variants = add_image_variant(st.session_state['edited_image'], st.session_state['which_image'], project_name, timing_details)
                            promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1)
                        st.session_state['edited_image'] = ""
                        st.success("Image promoted!")
                        st.experimental_rerun()
                        
            with st.expander("Replace Frame"):
                replace1, replace2, replace3 = st.columns([2,1,1])
                with replace1:            
                    replacement_frame = st.file_uploader("Upload a replacement frame here", type="png", accept_multiple_files=False, key="replacement_frame")
                with replace2:
                    st.write("")
                    confirm_replace = st.checkbox(f"I confirm I want to replace {st.session_state['which_stage']} {st.session_state['which_image']} with this frame", key="confirm_replace}")
                with replace3:
                    st.write("")
                    if confirm_replace == True and replacement_frame is not None:
                        if st.button("Replace frame",disabled=False):
                            images_for_model = []                    
                            with open(os.path.join(f"videos/{project_name}/",replacement_frame.name),"wb") as f: 
                                f.write(replacement_frame.getbuffer())     
                            uploaded_image = upload_image(f"videos/{project_name}/{replacement_frame.name}")
                            if st.session_state['which_stage'] == "Unedited Key Frame":
                                update_source_image(project_name, st.session_state['which_image'], uploaded_image)
                            elif st.session_state['which_stage'] == "Styled Key Frame":
                                number_of_image_variants = add_image_variant(uploaded_image, st.session_state['which_image'], project_name, timing_details)
                                promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1) 
                            # delete the uploaded file
                            os.remove(f"videos/{project_name}/{replacement_frame.name}")
                            st.success("Replaced")
                            time.sleep(1)     
                            st.experimental_rerun()
                                                            
                            
                    else:
                        st.button("Replace frame",disabled=True, help="You need to confirm you want to replace the frame and upload a replacement frame first.")
                    

        st.sidebar.markdown("### Batch Run Edits:")   
        st.sidebar.write("This will batch run the settings you have above on a batch of images.")     
        batch_run_range = st.sidebar.slider("Select range:", 1, 0, (0, len(timing_details)-1))                                
        if st.session_state['which_stage'] == "Unedited Key Frame":
            st.sidebar.warning("This will overwrite the source images in the range you select - you can always reset them if you wish.")
        elif st.session_state['which_stage'] == "Styled Key Frame":
            make_primary_variant = st.sidebar.checkbox("Make primary variant", value=True, help="If you want to make the edited image the primary variant, tick this box. If you want to keep the original primary variant, untick this box.")          
        if st.sidebar.button("Batch Run Edit"):
            for i in range(batch_run_range[1]+1):
                if st.session_state["type_of_mask_replacement"] == "Inpainting":
                        background_image = ""
                if st.session_state['which_stage'] == "Unedited Key Frame":
                    editing_image = timing_details[i]["source_image"]                            
                    edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, background_image, editing_image, prompt, negative_prompt, width, height,st.session_state['which_layer'], st.session_state['which_image'])
                elif st.session_state['which_stage'] == "Styled Key Frame":
                    variants = timing_details[i]["alternative_images"]
                    primary_image = timing_details[i]["primary_image"]             
                    editing_image = variants[primary_image]   
                    edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, background_image, editing_image, prompt, negative_prompt, width, height,st.session_state['which_layer'], st.session_state['which_image'])
                    number_of_image_variants = add_image_variant(edited_image, i, project_name, timing_details)                        
                    promote_image_variant(i, project_name, number_of_image_variants-1)
