import streamlit as st
import os
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps,ImageEnhance,ImageFilter, ImageChops
from moviepy.editor import *
from requests_toolbelt.multipart.encoder import MultipartEncoder
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import csv
import pandas as pd
import replicate
import urllib
import requests as r
import imageio
import ffmpeg
import string
import math
import json
import tempfile
import boto3
import time
import zipfile
from math import cos, sin,ceil,radians,gcd
import random
import uuid
from io import BytesIO
import ast
import numpy as np
from repository.local_repo.csv_repo import CSVProcessor, get_app_settings, get_project_settings, update_project_setting, update_specific_timing_value
from pydub import AudioSegment
import shutil
from moviepy.editor import concatenate_videoclips,TextClip,VideoFileClip, vfx
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from utils import st_memory
from urllib.parse import urlparse
 
from typing import Union
from moviepy.video.fx.all import speedx
import moviepy.editor
from streamlit_cropper import st_cropper

def prompt_finder_element(project_name):

    uploaded_file = st.file_uploader("What image would you like to find the prompt for?", type=['png','jpg','jpeg'], key="prompt_file")
    which_model = st.radio("Which model would you like to get a prompt for?", ["Stable Diffusion 1.5", "Stable Diffusion 2"], key="which_model", help="This is to know which model we should optimize the prompt for. 1.5 is usually best if you're in doubt", horizontal=True)
    best_or_fast = st.radio("Would you like to optimize for best quality or fastest speed?", ["Best", "Fast"], key="best_or_fast", help="This is to know whether we should optimize for best quality or fastest speed. Best quality is usually best if you're in doubt", horizontal=True).lower()
    if st.button("Get prompts"):                
        with open(f"videos/{project_name}/assets/resources/prompt_images/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        prompt = prompt_clip_interrogator(f"videos/{project_name}/assets/resources/prompt_images/{uploaded_file.name}", which_model, best_or_fast)                
        if not os.path.exists(f"videos/{project_name}/prompts.csv"):
            with open(f"videos/{project_name}/prompts.csv", "w") as f:
                f.write("prompt,example_image,which_model\n")
        # add the prompt to prompts.csv
        with open(f"videos/{project_name}/prompts.csv", "a") as f:
            f.write(f'"{prompt}",videos/{project_name}/assets/resources/prompt_images/{uploaded_file.name},{which_model}\n')
        st.success("Prompt added successfully!")
        time.sleep(1)
        uploaded_file = ""
        st.experimental_rerun()
    # list all the prompts in prompts.csv
    if os.path.exists(f"videos/{project_name}/prompts.csv"):
        
        df = pd.read_csv(f"videos/{project_name}/prompts.csv",na_filter=False)
        prompts = df.to_dict('records')
        
        prompts.reverse()
                    
        col1, col2 = st.columns([1.5,1])
        with col1:
            st.markdown("### Prompt")
        with col2:
            st.markdown("### Example Image")
        with open(f"videos/{project_name}/prompts.csv", "r") as f:
            for i in prompts:
                index_of_current_item = prompts.index(i)                  
                col1, col2 = st.columns([1.5,1])
                with col1:
                    st.write(prompts[index_of_current_item]["prompt"])                        
                with col2:                            
                    st.image(prompts[index_of_current_item]["example_image"], use_column_width=True)
                st.markdown("***")


def save_new_image(img: Union[Image.Image, str, np.ndarray]) -> str:
    file_name = str(uuid.uuid4()) + ".png"
    file_path = os.path.join("temp", file_name)

    # Check if img is a PIL image
    if isinstance(img, Image.Image):
        img.save(file_path)

    # Check if img is a URL
    elif isinstance(img, str) and bool(urlparse(img).netloc):
        response = r.get(img)
        img = Image.open(BytesIO(response.content))
        img.save(file_path)

    # Check if img is a local file
    elif isinstance(img, str):
        img = Image.open(img)
        img.save(file_path)

    # Check if img is a numpy ndarray
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        img.save(file_path)

    else:
        raise ValueError("Invalid image input. Must be a PIL image, a URL string, a local file path string or a numpy ndarray.")

    return file_path

def save_pillow_image(image, project_name, stage, promote=False):
    file_name = str(uuid.uuid4()) + ".png"
    if stage == "Source":
        save_location = f"videos/{project_name}/assets/frames/1_selected/{file_name}"
        image.save(save_location)
        update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", save_location)        
    elif stage == "Styled":
        save_location = f"videos/{project_name}/assets/frames/2_character_pipeline_completed/{file_name}"
        image.save(save_location)
        number_of_image_variants = add_image_variant(save_location, st.session_state['which_image'], project_name, timing_details)
        if promote:
            promote_image_variant(st.session_state['which_image'], project_name,number_of_image_variants - 1)         

def resize_and_rotate_element(stage,timing_details, project_name):

    if "rotated_image" not in st.session_state:
        st.session_state['rotated_image'] = ""
                                
    with st.expander("Zoom Image"):
        select1, select2 = st.columns([2,1])
        with select1:
            what_degree = st.number_input("Rotate image by: ", 0.0, 360.0, 0.0)
            what_zoom = st.number_input("Zoom image by: ", 0.1, 5.0, 1.0)
        with select2:
            fill_with = st.radio("Fill blank space with: ", ["Blur", None])
        if st.button("Rotate Image"): 
            if stage == "Source":
                input_image = timing_details[st.session_state['which_image']]["source_image"]                                
            elif stage == "Styled":
                input_image = get_primary_variant_location(timing_details, st.session_state['which_image'])
            if what_degree != 0:                           
                st.session_state['rotated_image'] = rotate_image(input_image,what_degree)
                st.session_state['rotated_image'].save("temp.png")
            else:
                st.session_state['rotated_image'] = input_image
                if st.session_state['rotated_image'].startswith("http"):
                    st.session_state['rotated_image'] = r.get(st.session_state['rotated_image'])
                    st.session_state['rotated_image'] = Image.open(BytesIO(st.session_state['rotated_image'].content))
                else:
                    st.session_state['rotated_image'] = Image.open(st.session_state['rotated_image'])
                st.session_state['rotated_image'].save("temp.png")
            if what_zoom != 1.0:
                st.session_state['rotated_image'] = zoom_image("temp.png", what_zoom,fill_with)
        if st.session_state['rotated_image'] != "":
            st.image(st.session_state['rotated_image'], caption="Rotated image", width=300)
            btn1, btn2 = st.columns(2)
            with btn1:
                if st.button("Save image",type="primary"):
                    file_name = str(uuid.uuid4()) + ".png"                    
                    if stage == "Source":                        
                        time.sleep(1)
                        save_location = f"videos/{project_name}/assets/frames/1_selected/{file_name}"                                          
                        st.session_state['rotated_image'].save(save_location)
                        update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", save_location)
                        st.session_state['rotated_image'] = ""
                        st.experimental_rerun()
                        
                    elif stage == "Styled":
                        save_location = f"videos/{project_name}/assets/frames/2_character_pipeline_completed/{file_name}"
                        st.session_state['rotated_image'].save(save_location)
                        number_of_image_variants = add_image_variant(save_location, st.session_state['which_image'], project_name, timing_details)
                        promote_image_variant(st.session_state['which_image'], project_name,number_of_image_variants - 1) 
                        st.session_state['rotated_image'] = ""
                        st.experimental_rerun()
            with btn2:
                if st.button("Clear Current Image"):
                    st.session_state['rotated_image'] = ""
                    st.experimental_rerun()
                            

def create_individual_clip(index_of_item, project_name):

    timing_details = get_timing_details(project_name)

    if timing_details[index_of_item]["animation_style"] == "":
        project_settings = get_project_settings(project_name)
        animation_style = project_settings["default_animation_style"]
    else:
        animation_style = timing_details[index_of_item]["animation_style"]

    if animation_style == "Interpolation":        
        output_video = prompt_interpolation_model(index_of_item, project_name)
        
    elif animation_style == "Direct Morphing":        
        output_video = create_video_without_interpolation(index_of_item, project_name)

    return output_video

        
def prompt_interpolation_model(index_of_current_item, project_name):
    timing_details = get_timing_details(project_name)
    img1 = get_primary_variant_location(timing_details, index_of_current_item)
    img2 = get_primary_variant_location(timing_details, index_of_current_item+1)
    interpolation_steps = int(float(timing_details[index_of_current_item]["interpolation_steps"]))
    app_settings = get_app_settings()
    replicate_api_key = app_settings["replicate_com_api_key"]
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    model = replicate.models.get("google-research/frame-interpolation")
    version = model.versions.get("4f88a16a13673a8b589c18866e540556170a5bcb2ccdc12de556e800e9456d3d")

    if not img1.startswith("http"):
        img1 = open(img1, "rb")

    if not img2.startswith("http"):
        img2 = open(img2, "rb")

    output = version.predict(frame1=img1, frame2=img2,times_to_interpolate=interpolation_steps)
    file_name = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=16)) + ".mp4"

    video_location = "videos/" + project_name + \
        "/assets/videos/0_raw/" + str(file_name)
    try:
        urllib.request.urlretrieve(output, video_location)

    except Exception as e:
        print(e)

    clip = VideoFileClip(video_location)

    return video_location

def create_video_without_interpolation(index_of_item, project_name):

    timing_details = get_timing_details(project_name)

    image_path_or_url = get_primary_variant_location(timing_details, index_of_item)

    video_location = "videos/" + project_name + "/assets/videos/0_raw/" + \
                     ''.join(random.choices(string.ascii_lowercase + string.digits, k=16)) + ".mp4"
    os.makedirs(os.path.dirname(video_location), exist_ok=True)

    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        response = r.get(image_path_or_url)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path_or_url)

    if image is None:
        raise ValueError("Could not read the image. Please provide a valid image path or URL.")

    height, width, _ = image.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = int(1 / 0.1)
    video_writer = cv2.VideoWriter(video_location, fourcc, fps, (width, height))

    for _ in range(fps):
        video_writer.write(image)

    video_writer.release()

    return video_location

def get_pillow_image(image_location):
    if image_location.startswith("http://") or image_location.startswith("https://"):
        response = r.get(image_location)
        image = Image.open(BytesIO(response.content))        
    else:
        image = Image.open(image_location)

    return image
        




            


def create_alpha_mask(size, edge_blur_radius):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)

    width, height = size
    draw.rectangle((edge_blur_radius, edge_blur_radius, width - edge_blur_radius, height - edge_blur_radius), fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_blur_radius))
    return mask

def zoom_image(location, zoom_factor, fill_with=None):
    blur_radius = 5
    edge_blur_radius = 15

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0")

    # Check if the provided location is a URL
    if location.startswith('http') or location.startswith('https'):
        response = requests.get(location)
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(f"File not found: {location}")
        image = Image.open(location)

    # Calculate new dimensions based on zoom factor
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    if zoom_factor < 1:
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        if fill_with == "Blur":
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Resize the blurred image to match the original dimensions
            blurred_background = blurred_image.resize((width, height), Image.ANTIALIAS)

            # Create an alpha mask for blending
            alpha_mask = create_alpha_mask(resized_image.size, edge_blur_radius)

            # Blend the resized image with the blurred background using the alpha mask
            blended_image = Image.composite(resized_image, blurred_background.crop((0, 0, new_width, new_height)), alpha_mask)

            # Calculate the position to paste the blended image at the center of the blurred background
            paste_left = (blurred_background.width - blended_image.width) // 2
            paste_top = (blurred_background.height - blended_image.height) // 2

            # Create a new blank image with the size of the blurred background
            final_image = Image.new('RGBA', blurred_background.size)

            # Paste the blurred background onto the final image
            final_image.paste(blurred_background, (0, 0))

            # Paste the blended image onto the final image using the alpha mask
            final_image.paste(blended_image, (paste_left, paste_top), mask=alpha_mask)

            return final_image
                        

        elif fill_with == "Inpainting":
            print("Coming soon")
            return resized_image

        elif fill_with is None:
            # Create an empty background with the original dimensions
            background = Image.new('RGBA', (width, height))

            # Calculate the position to paste the resized image at the center of the background
            paste_left = (background.width - resized_image.width) // 2
            paste_top = (background.height - resized_image.height) // 2

            # Paste the resized image onto the background
            background.paste(resized_image, (paste_left, paste_top))

            return background

        else:
            raise ValueError("Invalid fill_with value. Accepted values are 'Blur', 'Inpainting', and None.")

    else:
        # If zooming in, proceed as before
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        left = (resized_image.width - width) / 2
        top = (resized_image.height - height) / 2
        right = (resized_image.width + width) / 2
        bottom = (resized_image.height + height) / 2

        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image
    
def apply_image_transformations(image, zoom_level, rotation_angle, x_shift, y_shift):
    width, height = image.size

    # Calculate the diagonal for the rotation
    diagonal = math.ceil(math.sqrt(width**2 + height**2))

    # Create a new image with black background for rotation
    rotation_bg = Image.new("RGB", (diagonal, diagonal), "black")
    rotation_offset = ((diagonal - width) // 2, (diagonal - height) // 2)
    rotation_bg.paste(image, rotation_offset)

    # Rotation
    rotated_image = rotation_bg.rotate(rotation_angle)

    # Shift
    # Create a new image with black background
    shift_bg = Image.new("RGB", (diagonal, diagonal), "black")
    shift_bg.paste(rotated_image, (x_shift, y_shift))

    # Zoom
    zoomed_width = int(diagonal * (zoom_level / 100))
    zoomed_height = int(diagonal * (zoom_level / 100))
    zoomed_image = shift_bg.resize((zoomed_width, zoomed_height))

    # Crop the zoomed image back to original size
    crop_x1 = (zoomed_width - width) // 2
    crop_y1 = (zoomed_height - height) // 2
    crop_x2 = crop_x1 + width
    crop_y2 = crop_y1 + height
    cropped_image = zoomed_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return cropped_image


def fetch_image_by_stage(timing_details, stage):
    if stage == "Source":
        return timing_details[st.session_state['which_image']]["source_image"]
    elif stage == "Styled":
        return get_primary_variant_location(timing_details, st.session_state['which_image'])
    else:
        return ""


def precision_cropping_element(stage,timing_details, project_name, project_settings):

    def reset_zoom_element():
        st.session_state['zoom_level_input_value'] = 100
        st.session_state['rotation_angle_input_value'] = 0
        st.session_state['x_shift_input_value'] = 0
        st.session_state['y_shift_input_value'] = 0
        st.session_state['zoom_level'] = 100
        st.session_state['rotation_angle'] = 0
        st.session_state['x_shift'] = 0
        st.session_state['y_shift'] = 0                        
        st.experimental_rerun()
    
    input_image = fetch_image_by_stage(timing_details, stage)

    if input_image == "":
        st.error("Please select a source image before cropping")
        return
    else:
        input_image = get_pillow_image(input_image)
        
    col1, col2 = st.columns(2)    
    
    with col1:      

        st.subheader("Precision Cropping:")                

        if st.button("Reset Cropping"):            
            reset_zoom_element()
                                               
        st.session_state['zoom_level'] = st_memory.number_input("Zoom Level (%)", min_value=10, max_value=1000, step=10,key="zoom_level_input", default_value=100,project_name=project_name, project_settings=project_settings)
        st.session_state['rotation_angle'] = st_memory.number_input("Rotation Angle", min_value=-360, max_value=360, step=5, key="rotation_angle_input", default_value=0,project_name=project_name,project_settings=project_settings)
        st.session_state['x_shift'] = st_memory.number_input("Shift Left/Right", min_value=-1000, max_value=1000, step=5, key="x_shift_input", default_value=0,project_name=project_name,project_settings=project_settings)
        st.session_state['y_shift'] = st_memory.number_input("Shift Up/Down", min_value=-1000, max_value=1000,step=5, key="y_shift_input", default_value=0,project_name=project_name,project_settings=project_settings)
    
        st.caption("Input Image:") 
        st.image(input_image, caption="Input Image", width=300)

    with col2:

        st.caption("Output Image:")             
        output_image = apply_image_transformations(input_image, st.session_state['zoom_level'], st.session_state['rotation_angle'], st.session_state['x_shift'], st.session_state['y_shift'])
        st.image(output_image, use_column_width=True)
        if st.button("Save Image"):
            save_pillow_image(output_image,project_name, stage)            
            zoom_details = f"'{st.session_state['zoom_level_input_value']}', '{st.session_state['rotation_angle_input_value']}', '{st.session_state['x_shift_input_value']}', '{st.session_state['y_shift_input_value']}'"
            update_specific_timing_value(project_name, st.session_state['which_image'], "zoom_details", zoom_details)                                                      
            st.success("Image Saved Successfully")                        
        inpaint_in_black_space_element(output_image,project_settings, project_name)
                   




def manual_cropping_element(stage,timing_details,project_name):

    if timing_details[st.session_state['which_image']]["source_image"] == "":
        st.error("Please select a source image before cropping")
        return
    else:
        if stage == "Source":
            input_image = timing_details[st.session_state['which_image']]["source_image"]                                
        elif stage == "Styled":
            input_image = get_primary_variant_location(timing_details, st.session_state['which_image'])
        
        if 'current_working_image_number' not in st.session_state:
            st.session_state['current_working_image_number'] = st.session_state['which_image']
        
        def get_working_image():
            st.session_state['working_image'] = get_pillow_image(input_image)
            st.session_state['working_image'] = ImageOps.expand(st.session_state['working_image'], border=200, fill="black")
            st.session_state['current_working_image_number'] = st.session_state['which_image']
                                  
        
        if 'working_image' not in st.session_state or st.session_state['current_working_image_number'] != st.session_state['which_image']:
            get_working_image()

        options1,options2,option3, option4 = st.columns([3,1,1,1])
        with options1:
            sub_options_1, sub_options_2 = st.columns(2)
            if 'degrees_rotated_to' not in st.session_state:
                st.session_state['degrees_rotated_to'] = 0
            with sub_options_1:
                st.session_state['degree'] = st.slider("Rotate Image", -180, 180, value=st.session_state['degrees_rotated_to'])
                if st.session_state['degrees_rotated_to'] != st.session_state['degree']:                    
                    get_working_image()
                    st.session_state['working_image'] = st.session_state['working_image'].rotate(-st.session_state['degree'], resample=Image.BICUBIC, expand=True)
                    st.session_state['degrees_rotated_to'] = st.session_state['degree']
                    st.experimental_rerun()

            with sub_options_2:
                if st.button("Reset image"):
                    st.session_state['degree'] = 0
                    get_working_image()
                    st.session_state['degrees_rotated_to'] = 0
                    st.experimental_rerun()
                    

        with options2:
            if st.button("Flip horizontally", key="cropbtn1"):                
                st.session_state['working_image'] = st.session_state['working_image'].transpose(Image.FLIP_LEFT_RIGHT)
                
                # save 
            if st.button("Flip vertically", key="cropbtn2"):
                st.session_state['working_image'] = st.session_state['working_image'].transpose(Image.FLIP_TOP_BOTTOM)
        

        with option3:
            brightness_factor = st.slider("Brightness", 0.0, 2.0, 1.0)
            if brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(st.session_state['working_image'])
                st.session_state['working_image'] = enhancer.enhance(brightness_factor)
        with option4:
            contrast_factor = st.slider("Contrast", 0.0, 2.0, 1.0)
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(st.session_state['working_image'])
                st.session_state['working_image'] = enhancer.enhance(contrast_factor)
        
        project_settings = get_project_settings(project_name)

        width = int(project_settings["width"])
        height = int(project_settings["height"])

        gcd_value = gcd(width, height)
        aspect_ratio_width = int(width // gcd_value)
        aspect_ratio_height = int(height // gcd_value)
        aspect_ratio = (aspect_ratio_width, aspect_ratio_height)
        
        img1, img2 = st.columns([3,1.5])

        with img1:

            # use PIL to add 50 pixels of blackspace to the width and height of the image
        
                            
            cropped_img = st_cropper(st.session_state['working_image'], realtime_update=True, box_color="#0000FF", aspect_ratio=aspect_ratio)

        

        with img2:
            st.image(cropped_img, caption="Cropped Image", use_column_width=True,width=200)
        
            cropbtn1, cropbtn2 = st.columns(2)
            with cropbtn1:
                if st.button("Save Cropped Image"):
                    if stage == "Source":
                        # resize the image to the original width and height
                        cropped_img = cropped_img.resize((width, height), Image.ANTIALIAS)
                        # generate a random filename and save it to /temp               
                        file_name = f"temp/{uuid.uuid4()}.png"
                        cropped_img.save(file_name)            
                        st.success("Cropped Image Saved Successfully")
                        update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", file_name)
                        time.sleep(1)
                    st.experimental_rerun()
            with cropbtn2:
                st.warning("Warning: This will overwrite the original image")

            inpaint_in_black_space_element(cropped_img,project_settings, project_name)
                
                                                        

def inpaint_in_black_space_element(cropped_img,project_settings, project_name):
    with st.expander("Inpaint in black space"):
        inpaint_prompt = st.text_area("Prompt", value=project_settings["last_prompt"])
        inpaint_negative_prompt = st.text_input("Negative Prompt", value='edge,branches, frame, fractals, text' +project_settings["last_negative_prompt"])
        if 'inpainted_image' not in st.session_state:
            st.session_state['inpainted_image'] = ""
        if st.button("Inpaint"):
            width = int(project_settings["width"])
            height = int(project_settings["height"])
            saved_cropped_img = cropped_img.resize((width, height), Image.ANTIALIAS)                
            saved_cropped_img.save("temp/cropped.png")
            # Convert image to grayscale
            # Create a new image with the same size as the cropped image
            mask = Image.new('RGB', cropped_img.size)

            # Get the width and height of the image
            width, height = cropped_img.size

            for x in range(width):
                for y in range(height):
                    # Get the RGB values of the pixel
                    r, g, b = cropped_img.getpixel((x, y))

                    # If the pixel is black, set it and its adjacent pixels to black in the new image
                    if r == 0 and g == 0 and b == 0:
                        mask.putpixel((x, y), (0, 0, 0))  # Black
                        for i in range(-2, 3):  # Adjust these values to change the range of adjacent pixels
                            for j in range(-2, 3):
                                # Check that the pixel is within the image boundaries
                                if 0 <= x + i < width and 0 <= y + j < height:
                                    mask.putpixel((x + i, y + j), (0, 0, 0))  # Black
                    # Otherwise, make the pixel white in the new image
                    else:
                        mask.putpixel((x, y), (255, 255, 255))  # White
            # Save the mask image
            mask.save('temp/mask.png')
            
            st.session_state['inpainted_image'] = inpainting(project_name, "temp/cropped.png", inpaint_prompt, inpaint_negative_prompt, st.session_state['which_image'], True, pass_mask=True)
        
        if st.session_state['inpainted_image'] != "":
            st.image(st.session_state['inpainted_image'], caption="Inpainted Image", use_column_width=True,width=200)
        if st.button("Make Source Image"):                        
            update_specific_timing_value(project_name, st.session_state['which_image'], "source_image", st.session_state['inpainted_image'])
            st.session_state['inpainted_image'] = ""
            st.experimental_rerun()


def rotate_image(location, degree):
    if location.startswith('http') or location.startswith('https'):
        response = r.get(location)
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(f"File not found: {location}")
        image = Image.open(location)

    # Rotate the image by the specified degree
    rotated_image = image.rotate(-degree, resample=Image.BICUBIC, expand=False)

    return rotated_image



def create_or_get_single_preview_video(index_of_current_item, project_name):

    timing_details = get_timing_details(project_name)
    project_details = get_project_settings(project_name)

    if timing_details[index_of_current_item]['interpolated_video'] == "":
        update_specific_timing_value(project_name, index_of_current_item, "interpolation_steps", 3)
        interpolated_video = create_individual_clip(index_of_current_item, project_name)
        update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", interpolated_video)
        timing_details = get_timing_details(project_name)


    if timing_details[index_of_current_item]['timing_video'] == "": 

        interpolated_video = timing_details[index_of_current_item]['interpolated_video']
        output_filename = str(uuid.uuid4()) + ".mp4"
        output_file_path = "temp/" + output_filename
        shutil.copy(interpolated_video, output_file_path) 
        preview_video = output_file_path  
        clip = VideoFileClip(preview_video)        
        i = index_of_current_item
        number_text = TextClip(str(i), fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position((number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)
        clip_with_number = CompositeVideoClip([clip, number_background, number_text])        
        os.remove(preview_video)        
        clip_with_number.write_videofile(preview_video)                                                             
        duration_of_clip = calculate_desired_duration_of_individual_clip(timing_details, index_of_current_item)
        update_specific_timing_value(project_name, index_of_current_item, "duration_of_clip", duration_of_clip)        
        location_of_output_video = update_speed_of_video_clip(project_name, preview_video, False, index_of_current_item)
        update_specific_timing_value(project_name, index_of_current_item, "timing_video", location_of_output_video)

    timing_details = get_timing_details(project_name)                                 
        
    if project_details["audio"] != "":
        audio_bytes = get_audio_bytes_for_slice(project_name, index_of_current_item)
        add_audio_to_video_slice(timing_details[index_of_current_item]['timing_video'], audio_bytes)
            

    return timing_details[index_of_current_item]['timing_video']

def single_frame_time_changer(project_name, i, timing_details):
    frame_time = st.number_input("Frame time (secs):", min_value=0.0, max_value=3600.0, value=timing_details[i]["frame_time"], step=0.1, key=f"frame_time_{i}")                                                                   
    if frame_time != timing_details[i]["frame_time"]:
        update_specific_timing_value(project_name, i, "frame_time", frame_time)
        if i != 0:
            update_specific_timing_value(project_name, i-1, "timing_video", "")
        update_specific_timing_value(project_name, i, "timing_video", "")
        # if the frame time of this frame is more than the frame time of the next frame, then we need to update the next frame's frame time, and all the frames after that - shift them by the difference between the new frame time and the old frame time
        # if it's not the last item
        if i < len(timing_details) - 1:
            if frame_time > timing_details[i+1]["frame_time"]:
                for a in range(i+1, len(timing_details)):
                    this_frame_time = timing_details[a]["frame_time"]
                    # shift them by the difference between the new frame time and the old frame time
                    new_frame_time = this_frame_time + (frame_time - timing_details[i]["frame_time"])                            
                    update_specific_timing_value(project_name, a, "frame_time", new_frame_time)
                    update_specific_timing_value(project_name, a, "timing_video", "")
        st.experimental_rerun()


def create_full_preview_video(project_name, index_of_item, speed):
    
    timing_details = get_timing_details(project_name)
    num_timing_details = len(timing_details)
    clips = []
   

    for i in range(index_of_item - 2, index_of_item + 3):
        
        if i < 0 or i >= num_timing_details-1:
            continue

        primary_variant_location = get_primary_variant_location(timing_details, i)

        print(f"primary_variant_location for i={i}: {primary_variant_location}")

        if not primary_variant_location:
            break

        preview_video = create_or_get_single_preview_video(i, project_name)
        
        clip = VideoFileClip(preview_video)

        number_text = TextClip(str(i), fontsize=24, color='white')
        number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(number_text.w + 10, number_text.h + 10))
        number_background = number_background.set_position(('left', 'top')).set_duration(clip.duration)
        number_text = number_text.set_position((number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)
        
        clip_with_number = CompositeVideoClip([clip, number_background, number_text])

        # remove existing preview video
        os.remove(preview_video)
        clip_with_number.write_videofile(preview_video, codec='libx264', bitrate='3000k')
                                
        clips.append(preview_video)
        
            
    video_clips = [VideoFileClip(v) for v in clips]

    combined_clip = concatenate_videoclips(video_clips)

    output_filename = str(uuid.uuid4()) + ".mp4"

    video_location = f"videos/{project_name}/assets/videos/1_final/{output_filename}"

    combined_clip.write_videofile(video_location)

    if speed != 1.0:
        clip = VideoFileClip(video_location)
                
        output_clip = clip.fx(vfx.speedx, speed)
        
        os.remove(video_location)
        
        output_clip.write_videofile(video_location, codec="libx264", preset="fast")        

    return video_location
    




def back_and_forward_buttons(timing_details):
    smallbutton0,smallbutton1, smallbutton2,smallbutton3, smallbutton4 = st.columns([2,2,2,2,2])
    with smallbutton0:
        if st.session_state['which_image'] > 1:
            if st.button(f"{st.session_state['which_image']-2} ⏮️", key=f"Previous Previous Image for {st.session_state['which_image']}"):
                st.session_state['which_image_value'] = st.session_state['which_image_value'] - 2
                st.experimental_rerun()
    with smallbutton1:
        # if it's not the first image
        if st.session_state['which_image'] != 0:
            if st.button(f"{st.session_state['which_image']-1} ⏪", key=f"Previous Image for {st.session_state['which_image']}"):
                st.session_state['which_image_value'] = st.session_state['which_image_value'] - 1
                st.experimental_rerun()        
        
    with smallbutton2:
        st.button(f"{st.session_state['which_image']} 📍",disabled=True)
    with smallbutton3:
        # if it's not the last image
        if st.session_state['which_image'] != len(timing_details)-1:
            if st.button(f"{st.session_state['which_image']+1} ⏩", key=f"Next Image for {st.session_state['which_image']}"):
                st.session_state['which_image_value'] = st.session_state['which_image_value'] + 1
                st.experimental_rerun()
    with smallbutton4:
        if st.session_state['which_image'] < len(timing_details)-2:
            if st.button(f"{st.session_state['which_image']+2} ⏭️", key=f"Next Next Image for {st.session_state['which_image']}"):
                st.session_state['which_image_value'] = st.session_state['which_image_value'] + 2
                st.experimental_rerun()




def styling_element(project_name,timing_details, project_settings, view_type="Single", item_to_show=None):

    if view_type == "Single":
        append_to_item_name = f"{st.session_state['which_image']}"
    elif view_type == "List":
        append_to_item_name = "bulk"

    stages = ["Source Image", "Main Variant","Custom"]
    
    if view_type == "Single":
        if timing_details[item_to_show]['which_stage_to_run_on'] != "":
            if f'index_of_which_stage_to_run_on_{append_to_item_name}' not in st.session_state:
                st.session_state['which_stage_to_run_on'] = timing_details[item_to_show]['which_stage_to_run_on']
                st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(st.session_state['which_stage_to_run_on'])
        else:
            st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = 0
                    
    elif view_type == "List":
        if project_settings[f'last_which_stage_to_run_on'] != "":         
            if f'index_of_which_stage_to_run_on_{append_to_item_name}' not in st.session_state:
                st.session_state['which_stage_to_run_on'] = project_settings['last_which_stage_to_run_on']
                st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(st.session_state['which_stage_to_run_on'])            
        else:            
            st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = 0
    
    stages1, stages2 = st.columns([1,1])
    with stages1:
        st.session_state['which_stage_to_run_on'] = st.radio("What stage of images would you like to run styling on?", options=stages, horizontal=True, index =st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] , help="Extracted frames means the original frames from the video.")                                                                                     
    with stages2:
        if st.session_state['which_stage_to_run_on'] == "Source Image":
            image = timing_details[st.session_state['which_image']]['source_image']            
        elif st.session_state['which_stage_to_run_on'] == "Main Variant":
            image = get_primary_variant_location(timing_details, st.session_state['which_image'])
        if image != "":
            st.image(image, use_column_width=True, caption=f"Image {st.session_state['which_image']}")
        else:
            st.error(f"No {st.session_state['which_stage_to_run_on']} image found for this variant")
            
    if stages.index(st.session_state['which_stage_to_run_on']) != st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}']:
        st.session_state[f'index_of_which_stage_to_run_on_{append_to_item_name}'] = stages.index(st.session_state['which_stage_to_run_on'])
        st.experimental_rerun()
    
    custom_pipelines = ["None","Mystique"]     
                 
    if f'index_of_last_custom_pipeline_{append_to_item_name}' not in st.session_state:
        st.session_state[f'index_of_last_custom_pipeline_{append_to_item_name}'] = 0  

    st.session_state['custom_pipeline'] = st.selectbox(f"Custom Pipeline:", custom_pipelines, index=st.session_state[f'index_of_last_custom_pipeline_{append_to_item_name}'])

    if custom_pipelines.index(st.session_state['custom_pipeline']) != st.session_state[f'index_of_last_custom_pipeline_{append_to_item_name}']:
        st.session_state[f'index_of_last_custom_pipeline_{append_to_item_name}'] = custom_pipelines.index(st.session_state['custom_pipeline'])
        st.experimental_rerun()

    if st.session_state['custom_pipeline'] == "Mystique":
        if st.session_state[f'index_of_last_model_{append_to_item_name}'] > 1:
            st.session_state[f'index_of_last_model_{append_to_item_name}'] = 0       
            st.experimental_rerun()           
        with st.expander("Mystique is a custom pipeline that uses a multiple models to generate a consistent character and style transformation."):
            st.markdown("## How to use the Mystique pipeline")                
            st.markdown("1. Create a fine-tined model in the Custom Model section of the app - we recommend Dreambooth for character transformations.")
            st.markdown("2. It's best to include a detailed prompt. We recommend taking an example input image and running it through the Prompt Finder")
            st.markdown("3. Use [expression], [location], [mouth], and [looking] tags to vary the expression and location of the character dynamically if that changes throughout the clip. Varying this in the prompt will make the character look more natural - especially useful if the character is speaking.")
            st.markdown("4. In our experience, the best strength for coherent character transformations is 0.25-0.3 - any more than this and details like eye position change.")  
        models = ["LoRA","Dreambooth"]                                     
        st.session_state['model'] = st.selectbox(f"Which type of model is trained on your character?", models, index=st.session_state['index_of_last_model'])                    
        if st.session_state[f'index_of_last_model_{append_to_item_name}'] != models.index(st.session_state['model']):
            st.session_state[f'index_of_last_model_{append_to_item_name}'] = models.index(st.session_state['model'])
            st.experimental_rerun()                          
    else:               
        models = ['controlnet','stable_diffusion_xl','stable-diffusion-img2img-v2.1', 'depth2img', 'pix2pix', 'Dreambooth', 'LoRA','StyleGAN-NADA','real-esrgan-upscaling','controlnet_1_1_x_realistic_vision_v2_0']
        
        if view_type == "List":
            if project_settings['last_model'] != "":                                
                st.session_state['model'] = project_settings['last_model']
                st.session_state[f'index_of_last_model_{append_to_item_name}'] = models.index(st.session_state['model'])                    
            else:            
                st.session_state[f'index_of_last_model_{append_to_item_name}'] = 0
        elif view_type == "Single":
            if timing_details[item_to_show]['model_id'] != "":
                st.session_state['model'] = timing_details[item_to_show]['model_id']
                st.session_state[f'index_of_last_model_{append_to_item_name}'] = models.index(st.session_state['model'])
            else:
                st.session_state[f'index_of_last_model_{append_to_item_name}'] = 0
        
        st.session_state['model'] = st.selectbox(f"Which model would you like to use?", models, index=st.session_state[f'index_of_last_model_{append_to_item_name}'])

        if st.session_state[f'index_of_last_model_{append_to_item_name}'] != models.index(st.session_state['model']):
            st.session_state[f'index_of_last_model_{append_to_item_name}'] = models.index(st.session_state['model'])
            st.experimental_rerun() 
            
    
    if st.session_state['model'] == "controlnet":   
        controlnet_adapter_types = ["scribble","normal", "canny", "hed", "seg", "hough", "depth2img", "pose"]
        
        # if f'index_of_controlnet_adapter_type_{append_to_item_name}' not in st.session_state:
        if view_type == "List":
            if project_settings['last_adapter_type'] != "" and project_settings['last_adapter_type'] != "N":                 
                st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'] = controlnet_adapter_types.index(project_settings['last_adapter_type'])
                st.session_state['adapter_type'] = project_settings['last_adapter_type']
            else:
                st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'] = 0
        elif view_type == "Single":
            if timing_details[item_to_show]['adapter_type'] != "" and timing_details[item_to_show]['adapter_type'] != "N":                    
                st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'] = controlnet_adapter_types.index(timing_details[item_to_show]['adapter_type'])
                st.session_state['adapter_type'] = timing_details[item_to_show]['adapter_type']                    
            else:                    
                st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'] = 0
        
        
        st.session_state['adapter_type'] = st.selectbox(f"Adapter Type",controlnet_adapter_types, index=st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'])
        
        if st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'] != controlnet_adapter_types.index(st.session_state['adapter_type']):
            st.session_state[f'index_of_controlnet_adapter_type_{append_to_item_name}'] = controlnet_adapter_types.index(st.session_state['adapter_type'])
            st.experimental_rerun()
        st.session_state['custom_models'] = []   

    elif st.session_state['model'] == "LoRA": 
        if 'index_of_lora_model_1' not in st.session_state:
            st.session_state[f'index_of_lora_model_1_{append_to_item_name}'] = 0
            st.session_state[f'index_of_lora_model_2_{append_to_item_name}'] = 0
            st.session_state[f'index_of_lora_model_3_{append_to_item_name}'] = 0
        df = pd.read_csv('models.csv')
        filtered_df = df[df.iloc[:, 5] == 'LoRA']
        lora_model_list = filtered_df.iloc[:, 0].tolist()
        lora_model_list.insert(0, '')
        st.session_state['lora_model_1'] = st.selectbox(f"LoRA Model 1", lora_model_list, index=st.session_state['index_of_lora_model_1_{append_to_item_name}'])
        if st.session_state[f'index_of_lora_model_1_{append_to_item_name}'] != lora_model_list.index(st.session_state['lora_model_1']):
            st.session_state[f'index_of_lora_model_1_{append_to_item_name}'] = lora_model_list.index(st.session_state['lora_model_1'])
            st.experimental_rerun()
        st.session_state['lora_model_2'] = st.selectbox(f"LoRA Model 2", lora_model_list, index=st.session_state[f'index_of_lora_model_2_{append_to_item_name}'])
        if st.session_state[f'index_of_lora_model_2_{append_to_item_name}'] != lora_model_list.index(st.session_state['lora_model_2']):
            st.session_state[f'index_of_lora_model_2_{append_to_item_name}'] = lora_model_list.index(st.session_state['lora_model_2'])
            st.experimental_rerun()
        st.session_state['lora_model_3'] = st.selectbox(f"LoRA Model 3", lora_model_list, index=st.session_state[f'index_of_lora_model_3_{append_to_item_name}'])
        if st.session_state[f'index_of_lora_model_3_{append_to_item_name}'] != lora_model_list.index(st.session_state['lora_model_3']):
            st.session_state[f'index_of_lora_model_3_{append_to_item_name}'] = lora_model_list.index(st.session_state['lora_model_3'])                     
            st.experimental_rerun()
        st.session_state['custom_models'] = [st.session_state['lora_model_1'], st.session_state['lora_model_2'], st.session_state['lora_model_3']]                    
        st.info("You can reference each model in your prompt using the following keywords: <1>, <2>, <3> - for example '<1> in the style of <2>.")
        lora_adapter_types = ['sketch', 'seg', 'keypose', 'depth', None]
        if f"index_of_lora_adapter_type_{append_to_item_name}" not in st.session_state:
            st.session_state['index_of_lora_adapter_type'] = 0
        st.session_state['adapter_type'] = st.selectbox(f"Adapter Type:", lora_adapter_types, help="This is the method through the model will infer the shape of the object. ", index=st.session_state[f'index_of_lora_adapter_type_{append_to_item_name}'])
        if st.session_state[f'index_of_lora_adapter_type_{append_to_item_name}'] != lora_adapter_types.index(st.session_state['adapter_type']):
            st.session_state[f'index_of_lora_adapter_type_{append_to_item_name}'] = lora_adapter_types.index(st.session_state['adapter_type'])
    elif st.session_state['model'] == "Dreambooth":
        df = pd.read_csv('models.csv')
        filtered_df = df[df.iloc[:, 5] == 'Dreambooth']
        dreambooth_model_list = filtered_df.iloc[:, 0].tolist()
        if f'index_of_dreambooth_model_{append_to_item_name}' not in st.session_state:
            st.session_state[f'index_of_dreambooth_model_{append_to_item_name}'] = 0
        st.session_state['custom_models'] = st.selectbox(f"Dreambooth Model", dreambooth_model_list, index=st.session_state[f'index_of_dreambooth_model_{append_to_item_name}'])
        if st.session_state[f'index_of_dreambooth_model_{append_to_item_name}'] != dreambooth_model_list.index(st.session_state['custom_models']):
            st.session_state[f'index_of_dreambooth_model_{append_to_item_name}'] = dreambooth_model_list.index(st.session_state['custom_models'])                                    
    else:
        st.session_state['custom_models'] = []
        st.session_state['adapter_type'] = "N"
    
    if st.session_state['adapter_type'] == "canny":

        canny1, canny2 = st.columns(2)

        if view_type == "List":

            if project_settings['last_low_threshold'] != "":
                low_threshold_value = project_settings['last_low_threshold']
            else:
                low_threshold_value = 50
            
            if project_settings['last_high_threshold'] != "":
                high_threshold_value = project_settings['last_high_threshold']
            else:
                high_threshold_value = 150
        
        elif view_type == "Single":

            if timing_details[item_to_show]['low_threshold'] != "":
                low_threshold_value = timing_details[item_to_show]['low_threshold']
            else:
                low_threshold_value = 50
            
            if timing_details[item_to_show]['high_threshold'] != "":
                high_threshold_value = timing_details[item_to_show]['high_threshold']
            else:
                high_threshold_value = 150
        
        with canny1:
            st.session_state['low_threshold'] = st.slider('Low Threshold', 0, 255, value=int(low_threshold_value))            
        with canny2:
            st.session_state['high_threshold'] = st.slider('High Threshold', 0, 255, value=int(high_threshold_value))
    else:
        st.session_state['low_threshold'] = 0
        st.session_state['high_threshold'] = 0
    
    if st.session_state['model'] == "StyleGAN-NADA":
        st.warning("StyleGAN-NADA is a custom model that uses StyleGAN to generate a consistent character and style transformation. It only works for square images.")
        st.session_state['prompt'] = st.selectbox("What style would you like to apply to the character?", ['base', 'mona_lisa', 'modigliani', 'cubism', 'elf', 'sketch_hq', 'thomas', 'thanos', 'simpson', 'witcher', 'edvard_munch', 'ukiyoe', 'botero', 'shrek', 'joker', 'pixar', 'zombie', 'werewolf', 'groot', 'ssj', 'rick_morty_cartoon', 'anime', 'white_walker', 'zuckerberg', 'disney_princess', 'all', 'list'])
        st.session_state['strength'] = 0.5
        st.session_state['guidance_scale'] = 7.5
        st.session_state['seed'] = int(0)
        st.session_state['num_inference_steps'] = int(50)      

    else:
        if view_type == "List":
            if project_settings['last_prompt'] != "":
                st.session_state[f'prompt_value_{append_to_item_name}'] = project_settings['last_prompt']
            else:
                st.session_state[f'prompt_value_{append_to_item_name}'] = ""
            
        elif view_type == "Single":
            if timing_details[item_to_show]['prompt'] != "":
                st.session_state[f'prompt_value_{append_to_item_name}'] = timing_details[item_to_show]['prompt']
            else:
                st.session_state[f'prompt_value_{append_to_item_name}'] = ""

        st.session_state['prompt'] = st.text_area(f"Prompt", label_visibility="visible", value=st.session_state[f'prompt_value_{append_to_item_name}'],height=150)
        if st.session_state['prompt'] != st.session_state['prompt_value']:
            st.session_state['prompt_value'] = st.session_state['prompt']
            st.experimental_rerun()
        with st.expander("💡 Learn about dynamic prompting"):
            st.markdown("## Why and how to use dynamic prompting")
            st.markdown("Why:")
            st.markdown("Dynamic prompting allows you to automatically vary the prompt throughout the clip based on changing features in the source image. This makes the output match the input more closely and makes character transformations look more natural.")
            st.markdown("How:")
            st.markdown("You can include the following tags in the prompt to vary the prompt dynamically: [expression], [location], [mouth], and [looking]")
        if st.session_state['model'] == "Dreambooth":
            model_details = get_model_details(st.session_state['custom_models'])
            st.info(f"Must include '{model_details['keyword']}' to run this model")   
            if model_details['controller_type'] != "":                    
                st.session_state['adapter_type']  = st.selectbox(f"Would you like to use the {model_details['controller_type']} controller?", ['Yes', 'No'])
            else:
                st.session_state['adapter_type']  = "No"

        else:
            if st.session_state['model'] == "pix2pix":
                st.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")                    
        
        if view_type == "List":
            if project_settings['last_strength'] != "":
                st.session_state['strength'] = project_settings['last_strength']
            else:
                st.session_state['strength'] = 0.5
        
        elif view_type == "Single":
            if timing_details[item_to_show]['strength'] != "":
                st.session_state['strength'] = timing_details[item_to_show]['strength']
            else:
                st.session_state['strength'] = 0.5


        st.session_state['strength'] = st.slider(f"Strength", value=float(st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)
        
        with st.expander("Advanced settings 😏"):
            if view_type == "List":
                if project_settings['last_guidance_scale'] != "":
                    st.session_state['guidance_scale'] = project_settings['last_guidance_scale']
                else:
                    st.session_state['guidance_scale'] = 7.5
            elif view_type == "Single":
                if timing_details[item_to_show]['guidance_scale'] != "":
                    st.session_state['guidance_scale'] = timing_details[item_to_show]['guidance_scale']
                else:
                    st.session_state['guidance_scale'] = 7.5
            st.session_state['negative_prompt'] = st.text_area(f"Negative prompt", value=st.session_state['negative_prompt_value'], label_visibility="visible")
            if st.session_state['negative_prompt'] != st.session_state['negative_prompt_value']:
                st.session_state['negative_prompt_value'] = st.session_state['negative_prompt']
                st.experimental_rerun()
            st.session_state['guidance_scale'] = st.number_input(f"Guidance scale", value=float(st.session_state['guidance_scale']))
            if view_type == "List":
                if project_settings['last_seed'] != "":
                    st.session_state['seed'] = project_settings['last_seed']
                else:
                    st.session_state['seed'] = 0
            elif view_type == "Single":
                if timing_details[item_to_show]['seed'] != "":
                    st.session_state['seed'] = timing_details[item_to_show]['seed']
                else:
                    st.session_state['seed'] = 0
            st.session_state['seed'] = st.number_input(f"Seed", value=int(st.session_state['seed']))
            if view_type == "List":
                if project_settings['last_num_inference_steps'] != "":
                    st.session_state['num_inference_steps'] = project_settings['last_num_inference_steps']
                else:
                    st.session_state['num_inference_steps'] = 50
            elif view_type == "Single":
                if timing_details[item_to_show]['num_inference_steps'] != "":
                    st.session_state['num_inference_steps'] = timing_details[item_to_show]['num_inference_steps']
                else:
                    st.session_state['num_inference_steps'] = 50
            st.session_state['num_inference_steps'] = st.number_input(f"Inference steps", value=int(st.session_state['num_inference_steps']))
    
    st.session_state["promote_new_generation"] = st.checkbox("Promote new generation to main variant", key="promote_new_generation_to_main_variant")
    st.session_state["use_new_settings"] = st.checkbox("Use new settings for batch query", key="keep_existing_settings", help="If unchecked, the new settings will be applied to the existing variants.")


    if len(timing_details) > 1:
        st.markdown("***")
        st.markdown("## Batch queries")
        batch_run_range = st.slider("Select range:", 1, 0, (0, len(timing_details)-1))  
        first_batch_run_value = batch_run_range[0]
        last_batch_run_value = batch_run_range[1]
        
        st.write(batch_run_range)
        
                        
        
        if 'restyle_button' not in st.session_state:
            st.session_state['restyle_button'] = ''
            st.session_state['item_to_restyle'] = ''                

        btn1, btn2 = st.columns(2)

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

                for i in range(first_batch_run_value, last_batch_run_value+1):
                    for number in range(0, batch_number_of_variants):
                        index_of_current_item = i
                        trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],st.session_state['custom_models'],st.session_state['adapter_type'],st.session_state["use_new_settings"],st.session_state['low_threshold'],st.session_state['high_threshold'])
                st.experimental_rerun()
    
    

def get_primary_variant_location(timing_details, which_image):

    if timing_details[which_image]["alternative_images"] == "":
        return ""
    else:                         
        variants = timing_details[which_image]["alternative_images"]                
        current_variant = int(timing_details[which_image]["primary_image"])       
        primary_variant_location = variants[current_variant]
        return primary_variant_location



def convert_to_minutes_and_seconds(frame_time):
    minutes = int(frame_time/60)
    seconds = frame_time - (minutes*60)
    seconds = round(seconds, 2)
    return f"{minutes} min, {seconds} secs"

def calculate_time_at_frame_number(input_video, frame_number, project_name):
    input_video = "videos/" + \
        str(project_name) + "/assets/resources/input_videos/" + str(input_video)
    video = cv2.VideoCapture(input_video)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_percentage = float(frame_number / frame_count)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    time_at_frame = float(frame_percentage * length_of_video)
    return time_at_frame


def preview_frame(project_name, input_video, frame_num):
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame


def extract_frame(frame_number, project_name, input_video, extract_frame_number, timing_details):
    
    input_video = cv2.VideoCapture(input_video)
    total_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    if extract_frame_number == total_frames:
        extract_frame_number = int(total_frames - 1)
    input_video.set(cv2.CAP_PROP_POS_FRAMES, extract_frame_number)
    ret, frame = input_video.read()

    if timing_details[frame_number]["frame_number"] == "":
        update_specific_timing_value(
            project_name, frame_number, "frame_number", extract_frame_number)

    file_name = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=16)) + ".png"
    file_location = "videos/" + project_name + "/assets/frames/1_selected/" + str(file_name)
    cv2.imwrite(file_location, frame)
    # img = Image.open("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    # img.save("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    return file_location   


def calculate_frame_number_at_time(input_video, time_of_frame, project_name):
    time_of_frame = float(time_of_frame)
    input_video = "videos/" + \
        str(project_name) + "/assets/resources/input_videos/" + str(input_video)
    video = cv2.VideoCapture(input_video)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    percentage_of_video = float(time_of_frame / length_of_video)
    frame_number = int(percentage_of_video * frame_count)
    if frame_number == 0:
        frame_number = 1
    return frame_number


def move_frame(direction, frame_number, project_name):

    timing_details = get_timing_details(project_name)

    current_primary_image = timing_details[frame_number]["primary_image"]
    current_alternative_images = timing_details[frame_number]["alternative_images"]
    current_source_image = timing_details[frame_number]["source_image"]

    if direction == "Up":

        previous_primary_image = timing_details[frame_number - 1]["primary_image"]
        previous_alternative_images = timing_details[frame_number - 1]["alternative_images"]
        previous_source_image = timing_details[frame_number - 1]["source_image"]
        
        update_specific_timing_value(project_name, frame_number - 1, "primary_image", current_primary_image)
        print("current_alternative_images", current_alternative_images)
        update_specific_timing_value(project_name, frame_number - 1, "alternative_images", str(current_alternative_images))
        update_specific_timing_value(project_name, frame_number - 1, "source_image", current_source_image)
        update_specific_timing_value(project_name, frame_number - 1, "interpolated_video", "")
        update_specific_timing_value(project_name, frame_number - 1, "timing_video", "")

        update_specific_timing_value(project_name, frame_number, "primary_image", previous_primary_image)
        update_specific_timing_value(project_name, frame_number, "alternative_images", str(previous_alternative_images))
        update_specific_timing_value(project_name, frame_number, "source_image", previous_source_image)


    elif direction == "Down":

        next_primary_image = timing_details[frame_number + 1]["primary_image"]
        next_alternative_images = timing_details[frame_number + 1]["alternative_images"]
        next_source_image = timing_details[frame_number + 1]["source_image"]

        update_specific_timing_value(project_name, frame_number + 1, "primary_image", current_primary_image)
        update_specific_timing_value(project_name, frame_number + 1, "alternative_images", str(current_alternative_images))
        update_specific_timing_value(project_name, frame_number + 1, "source_image", current_source_image)
        update_specific_timing_value(project_name, frame_number + 1, "interpolated_video", "")
        update_specific_timing_value(project_name, frame_number + 1, "timing_video", "")

        update_specific_timing_value(project_name, frame_number, "primary_image", next_primary_image)
        update_specific_timing_value(project_name, frame_number, "alternative_images", str(next_alternative_images))
        update_specific_timing_value(project_name, frame_number, "source_image", next_source_image)


    update_specific_timing_value(project_name, frame_number, "interpolated_video", "")
    update_specific_timing_value(project_name, frame_number, "timing_video", "")




def get_timing_details(video_name):
    file_path = "videos/" + str(video_name) + "/timings.csv"
    csv_processor = CSVProcessor(file_path)
    column_types = {
        'frame_time': float,
        'frame_number': int,
        'primary_image': int,
        'guidance_scale': float,
        'seed': int,
        'num_inference_steps': int,
        'strength': float
    }
    df = csv_processor.get_df_data().astype(column_types, errors='ignore')
    
    df['primary_image'] = pd.to_numeric(df['primary_image'], errors='coerce').round().astype(pd.Int64Dtype(), errors='ignore')
    df['seed'] = pd.to_numeric(df['seed'], errors='coerce').round().astype(pd.Int64Dtype(), errors='ignore')
    df['num_inference_steps'] = pd.to_numeric(df['num_inference_steps'], errors='coerce').round().astype(pd.Int64Dtype(), errors='ignore')
    # if source_image if empty, set to https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png
    
    
    
    


    # Evaluate the alternative_images column and replace it with the evaluated list
    df['alternative_images'] = df['alternative_images'].fillna(
        '').apply(lambda x: ast.literal_eval(x[1:-1]) if x != '' else '')
    return df.to_dict('records')

# delete keyframe at a particular index from timings.csv
def delete_frame(project_name, index_of_current_item):
    update_specific_timing_value(
        project_name, index_of_current_item - 1, "interpolated_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(
            project_name, index_of_current_item + 1, "interpolated_video", "")

    update_specific_timing_value(
        project_name, index_of_current_item - 1, "timing_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(
            project_name, index_of_current_item + 1, "timing_video", "")

    csv_processor = CSVProcessor(
        "videos/" + str(project_name) + "/timings.csv")
    csv_processor.delete_row(index_of_current_item)


def batch_update_timing_values(project_name, index_of_current_item, prompt, strength, model, custom_pipeline, negative_prompt, guidance_scale, seed, num_inference_steps, source_image, custom_models, adapter_type,low_threshold,high_threshold,which_stage_to_run_on):
    
    csv_processor = CSVProcessor(
        "videos/" + str(project_name) + "/timings.csv")
    df = csv_processor.get_df_data()

    if model != "Dreambooth":
        custom_models = f'"{custom_models}"'
    df.iloc[index_of_current_item, [18, 10, 9, 4, 5, 6, 7, 8, 12, 13, 14,24,25,27]] = [prompt, float(strength), model, custom_pipeline, negative_prompt, float(
        guidance_scale), int(seed), int(num_inference_steps), source_image, custom_models, adapter_type,int(float(low_threshold)),int(float(high_threshold)),which_stage_to_run_on]

    df["primary_image"] = pd.to_numeric(df["primary_image"], downcast='integer', errors='coerce')
    df["primary_image"].fillna(0, inplace=True)
    df["primary_image"] = df["primary_image"].astype(int)

    df["seed"] = pd.to_numeric(df["seed"], downcast='integer', errors='coerce')
    df["seed"].fillna(0, inplace=True)
    df["seed"] = df["seed"].astype(int)
    
    df["num_inference_steps"] = pd.to_numeric(
        df["num_inference_steps"], downcast='integer', errors='coerce')
    df["num_inference_steps"].fillna(0, inplace=True)
    df["num_inference_steps"] = df["num_inference_steps"].astype(int)

    df["low_threshold"] = pd.to_numeric(df["low_threshold"], downcast='integer', errors='coerce')
    df["low_threshold"].fillna(0, inplace=True)
    df["low_threshold"] = df["low_threshold"].astype(int)

    df["high_threshold"] = pd.to_numeric(df["high_threshold"], downcast='integer', errors='coerce')
    df["high_threshold"].fillna(0, inplace=True)
    df["high_threshold"] = df["high_threshold"].astype(int)




    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)


def dynamic_prompting(prompt, source_image, project_name, index_of_current_item):

    if "[expression]" in prompt:
        prompt_expression = facial_expression_recognition(source_image)
        prompt = prompt.replace("[expression]", prompt_expression)

    if "[location]" in prompt:
        prompt_location = prompt_model_blip2(
            source_image, "What's surrounding the character?")
        prompt = prompt.replace("[location]", prompt_location)

    if "[mouth]" in prompt:
        prompt_mouth = prompt_model_blip2(
            source_image, "is their mouth open or closed?")
        prompt = prompt.replace("[mouth]", "mouth is " + str(prompt_mouth))

    if "[looking]" in prompt:
        prompt_looking = prompt_model_blip2(
            source_image, "the person is looking")
        prompt = prompt.replace("[looking]", "looking " + str(prompt_looking))

    update_specific_timing_value(
        project_name, index_of_current_item, "prompt", prompt)


def trigger_restyling_process(timing_details, project_name, index_of_current_item, model, prompt, strength, custom_pipeline, negative_prompt, guidance_scale, seed, num_inference_steps, which_stage_to_run_on, promote_new_generation, project_settings, custom_models, adapter_type, update_inference_settings,low_threshold,high_threshold):

    timing_details = get_timing_details(project_name)
    if update_inference_settings is True:        
        get_model_details(model)
        prompt = prompt.replace(",", ".")
        prompt = prompt.replace("\n", "")
        update_project_setting("last_prompt", prompt, project_name)
        update_project_setting("last_strength", strength, project_name)
        update_project_setting("last_model", model, project_name)
        update_project_setting("last_custom_pipeline",
                            custom_pipeline, project_name)
        update_project_setting("last_negative_prompt",
                            negative_prompt, project_name)
        update_project_setting("last_guidance_scale", guidance_scale, project_name)
        update_project_setting("last_seed", seed, project_name)
        update_project_setting("last_num_inference_steps",
                            num_inference_steps, project_name)
        update_project_setting("last_which_stage_to_run_on",
                            which_stage_to_run_on, project_name)
        update_project_setting("last_custom_models", custom_models, project_name)
        update_project_setting("last_adapter_type", adapter_type, project_name)
        if low_threshold != "":
            update_project_setting("last_low_threshold", low_threshold, project_name)
        if high_threshold != "":
            update_project_setting("last_high_threshold", high_threshold, project_name)


        if timing_details[index_of_current_item]["source_image"] == "":
            source_image = ""
        else:
            source_image = timing_details[index_of_current_item]["source_image"]
        batch_update_timing_values(project_name, index_of_current_item, '"'+prompt+'"', strength, model, custom_pipeline,
                                negative_prompt, guidance_scale, seed, num_inference_steps, source_image, custom_models, adapter_type,low_threshold,high_threshold,which_stage_to_run_on)
        dynamic_prompting(prompt, source_image, project_name,
                      index_of_current_item)
    timing_details = get_timing_details(project_name)
    if which_stage_to_run_on == "Extracted Key Frames":
        source_image = timing_details[index_of_current_item]["source_image"]
    else:
        variants = timing_details[index_of_current_item]["alternative_images"]
        number_of_variants = len(variants)
        primary_image = int(
            timing_details[index_of_current_item]["primary_image"])
        source_image = variants[primary_image]

    
    timing_details = get_timing_details(project_name)

    if st.session_state['custom_pipeline'] == "Mystique":
        output_url = custom_pipeline_mystique(
            index_of_current_item, project_name, project_settings, timing_details, source_image)
    else:
        output_url = restyle_images(
            index_of_current_item, project_name, project_settings, timing_details, source_image)
        
    if output_url != None:

        add_image_variant(output_url, index_of_current_item,
                        project_name, timing_details)

        if promote_new_generation == True:
            timing_details = get_timing_details(project_name)
            variants = timing_details[index_of_current_item]["alternative_images"]
            number_of_variants = len(variants)
            if number_of_variants == 1:
                print("No new generation to promote")
            else:
                promote_image_variant(index_of_current_item,
                                    project_name, number_of_variants - 1)
    else:
        print("No new generation to promote")


def promote_image_variant(index_of_current_item, project_name, variant_to_promote):
    update_specific_timing_value(
        project_name, index_of_current_item, "primary_image", variant_to_promote)
    update_specific_timing_value(
        project_name, index_of_current_item - 1, "interpolated_video", "")
    update_specific_timing_value(
        project_name, index_of_current_item, "interpolated_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(
            project_name, index_of_current_item + 1, "interpolated_video", "")
    update_specific_timing_value(
        project_name, index_of_current_item - 1, "timing_video", "")
    update_specific_timing_value(
        project_name, index_of_current_item, "timing_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(
            project_name, index_of_current_item + 1, "timing_video", "")


def extract_canny_lines(image_path_or_url, project_name, low_threshold=50, high_threshold=150):
    # Check if the input is a URL
    if image_path_or_url.startswith("http"):
        response = r.get(image_path_or_url)
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    else:
        # Read the image from a local file
        image = cv2.imread(image_path_or_url, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply the Canny edge detection
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Reverse the colors (invert the image)
    inverted_canny_edges = 255 - canny_edges

    # Convert the inverted Canny edge result to a PIL Image
    new_canny_image = Image.fromarray(inverted_canny_edges)

    # Save the new image
    unique_file_name = str(uuid.uuid4()) + ".png"
    file_location = f"videos/{project_name}/assets/resources/masks/{unique_file_name}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    new_canny_image.save(file_location)

    return file_location


def create_or_update_mask(project_name, index_of_current_number, image):
    timing_details = get_timing_details(project_name)
    if timing_details[index_of_current_number]["mask"] == "":
        unique_file_name = str(uuid.uuid4()) + ".png"
        update_specific_timing_value(project_name, index_of_current_number, "mask",
                                     f"videos/{project_name}/assets/resources/masks/{unique_file_name}")
        timing_details = get_timing_details(project_name)
    else:
        unique_file_name = timing_details[st.session_state['which_image']]["mask"].split(
            "/")[-1]
    file_location = f"videos/{project_name}/assets/resources/masks/{unique_file_name}"
    image.save(file_location, "PNG")
    return file_location





def create_working_assets(video_name):
    os.mkdir("videos/" + video_name)
    os.mkdir("videos/" + video_name + "/assets")

    os.mkdir("videos/" + video_name + "/assets/frames")

    os.mkdir("videos/" + video_name + "/assets/frames/0_extracted")
    os.mkdir("videos/" + video_name + "/assets/frames/1_selected")
    os.mkdir("videos/" + video_name +
             "/assets/frames/2_character_pipeline_completed")
    os.mkdir("videos/" + video_name +
             "/assets/frames/3_backdrop_pipeline_completed")

    os.mkdir("videos/" + video_name + "/assets/resources")

    os.mkdir("videos/" + video_name + "/assets/resources/backgrounds")
    os.mkdir("videos/" + video_name + "/assets/resources/masks")
    os.mkdir("videos/" + video_name + "/assets/resources/audio")
    os.mkdir("videos/" + video_name + "/assets/resources/input_videos")
    os.mkdir("videos/" + video_name + "/assets/resources/prompt_images")

    os.mkdir("videos/" + video_name + "/assets/videos")

    os.mkdir("videos/" + video_name + "/assets/videos/0_raw")
    os.mkdir("videos/" + video_name + "/assets/videos/1_final")
    os.mkdir("videos/" + video_name + "/assets/videos/2_completed")

    data = {'key': ['last_prompt', 'last_model', 'last_strength', 'last_custom_pipeline', 'audio', 'input_type', 'input_video', 'extraction_type', 'width', 'height', 'last_negative_prompt', 'last_guidance_scale', 'last_seed', 'last_num_inference_steps', 'last_which_stage_to_run_on', 'last_custom_models', 'last_adapter_type','guidance_type','default_animation_style','last_low_threshold','last_high_threshold','last_stage_run_on','zoom_level_input_value','rotation_angle_input_value','x_shift_input_value','y_shift_input_value'],
            'value': ['prompt', 'controlnet', '0.5', 'None', '', 'video', '', 'Extract manually', '', '', '', 7.5, 0, 50, 'Extracted Key Frames', '', '', '', '',100,200,'',100,0,0,0]}

    df = pd.DataFrame(data)

    df.to_csv(f'videos/{video_name}/settings.csv', index=False)

    df = pd.DataFrame(columns=['frame_time', 'frame_number', 'primary_image', 'alternative_images', 'custom_pipeline', 'negative_prompt', 'guidance_scale', 'seed', 'num_inference_steps',
                      'model_id', 'strength', 'notes', 'source_image', 'custom_models', 'adapter_type', 'duration_of_clip', 'interpolated_video', 'timing_video', 'prompt', 'mask','canny_image','preview_video','animation_style','interpolation_steps','low_threshold','high_threshold','zoom_details'])

    df.loc[0] = [0,"", 0, "", "", "", 0, 0, 0, "", 0, "", "", "", "", 0, "", "", "", "", "", "", "", "", "", "",""]

    st.session_state['which_image'] = 0

    df.to_csv(f'videos/{video_name}/timings.csv', index=False)


def inpainting(project_name, input_image, prompt, negative_prompt, index_of_current_item, invert_mask, pass_mask=False):

    app_settings = get_app_settings()
    timing_details = get_timing_details(project_name)

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")

    version = model.versions.get(
        "e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")
    if pass_mask == False:
        mask = timing_details[index_of_current_item]["mask"]
    else:
        mask = "temp/mask.png"

    if not mask.startswith("http"):
        mask = open(mask, "rb")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    output = version.predict(mask=mask, image=input_image, prompt=prompt,
                             invert_mask=invert_mask, negative_prompt=negative_prompt, num_inference_steps=25)

    return output[0]


def add_image_variant(image_url, index_of_current_item, project_name, timing_details):
    if str(timing_details[index_of_current_item]["alternative_images"]) == "":
        alternative_images = f"['{image_url}']"
        additions = []
    else:
        alternative_images = []

        additions = timing_details[index_of_current_item]["alternative_images"]
        for addition in additions:
            alternative_images.append(addition)
        alternative_images.append(image_url)
    update_specific_timing_value(project_name, index_of_current_item,
                                 "alternative_images", '"' + str(alternative_images) + '"')

    if str(timing_details[index_of_current_item]["primary_image"]) == "":
        timing_details[index_of_current_item]["primary_image"] = 0
        update_specific_timing_value(project_name, index_of_current_item,
                                     "primary_image", timing_details[index_of_current_item]["primary_image"])
    return len(additions) + 1
    
def train_model(app_settings, images_list, instance_prompt,class_prompt, max_train_steps, model_name,project_name, type_of_model, type_of_task, resolution, controller_type):
    
    for i in range(len(images_list)):
        images_list[i] = 'training_data/' + images_list[i]

    with zipfile.ZipFile('images.zip', 'w') as zip:
        for image in images_list:
            zip.write(image, arcname=os.path.basename(image))

    os.environ["REPLICATE_API_TOKEN"] = app_settings['replicate_com_api_key']
    url = "https://dreambooth-api-experimental.replicate.com/v1/upload/data.zip"
    headers = {
        "Authorization": "Token " + os.environ.get("REPLICATE_API_TOKEN"),
        "Content-Type": "application/zip"
    }
    response = r.post(url, headers=headers)
    if response.status_code != 200:
        st.error(response.content)
        return
    upload_url = response.json()["upload_url"]
    serving_url = response.json()["serving_url"]
    with open('images.zip', 'rb') as f:
        r.put(upload_url, data=f, headers=headers)
    training_file_url = serving_url
    url = "https://dreambooth-api-experimental.replicate.com/v1/trainings"
    os.remove('images.zip')
    model_name = model_name.replace(" ", "-").lower()
    
    if type_of_model == "Dreambooth":
        # ["normal", "canny", "hed", "scribble", "seg", "openpose", "depth","mlsd"]
        if controller_type == "normal":
            template_version = "b65d36e378a01ef81d81ba49be7deb127e9bb8b74a28af3aa0eaca16b9bcd0eb"
        elif controller_type == "canny":
            template_version = "3c60cbfce253b1d82fea02c7692d13c1e96b36a22da784470fcbedc603a1ed4b"
        elif controller_type == "hed":
            template_version = "bef0803be223ecb38361097771dbea7cd166514996494123db27907da53d75cd"
        elif controller_type == "scribble":
            template_version = "346b487d77a0bdd150c4bbb8f162f7cd4a4491bca5f309105e078556d0789f11"
        elif controller_type == "seg":
            template_version = "a0266713f8c30b35a3f4fc8212fc9450cecea61e4181af63cfb54e5a152ecb24"
        elif controller_type == "openpose":
            template_version = "141b8753e2973933441880e325fd21404923d0877014c9f8903add05ff530e52"
        elif controller_type == "depth":
            template_version = "6cf8fc430894121f2f91867978780011e6859b6956b499b43273afc25ed21121"
        elif controller_type == "mlsd":
            template_version == "04982e9aa6d3998c2a2490f92e7ccfab2dbd93f5be9423cdf0405c7b86339022"        

        headers = {
            "Authorization": "Token " + os.environ.get("REPLICATE_API_TOKEN"),
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "instance_prompt": instance_prompt,
                "class_prompt": class_prompt,
                "instance_data": training_file_url,
                "max_train_steps": max_train_steps
            },
            "model": "peter942/" + str(model_name),
            "trainer_version": "cd3f925f7ab21afaef7d45224790eedbb837eeac40d22e8fefe015489ab644aa",
            "template_version": template_version,
            "webhook_completed": "https://example.com/dreambooth-webhook"
        }    
        response = r.post(url, headers=headers, data=json.dumps(payload))    
        response = (response.json())
        training_status = response["status"]
        model_id = response["id"]
        if training_status == "queued":
            df = pd.read_csv("models.csv")
            df = df.append({}, ignore_index=True)
            new_row_index = df.index[-1]
            df.iloc[new_row_index, 0] = model_name
            df.iloc[new_row_index, 1] = model_id
            df.iloc[new_row_index, 2] = instance_prompt
            df.iloc[new_row_index, 4] = str(images_list)
            df.iloc[new_row_index, 5] = "Dreambooth"
            df.to_csv("models.csv", index=False)
            return "Success - Training Started. Please wait 10-15 minutes for the model to be trained."
        else:
            return "Failed"

    elif type_of_model == "LoRA":
        model = replicate.models.get("cloneofsimo/lora-training")
        version = model.versions.get(
            "b2a308762e36ac48d16bfadc03a65493fe6e799f429f7941639a6acec5b276cc")
        output = version.predict(
            instance_data=training_file_url, task=type_of_task, resolution=int(resolution))
        df = pd.read_csv("models.csv")
        df = df.append({}, ignore_index=True)
        new_row_index = df.index[-1]
        df.iloc[new_row_index, 0] = model_name
        df.iloc[new_row_index, 4] = str(images_list)
        df.iloc[new_row_index, 5] = "LoRA"
        df.iloc[new_row_index, 6] = output
        df.to_csv("models.csv", index=False)
        return f"Successfully trained - the model '{model_name}' is now available for use!"


def get_model_details(model_name):
    with open('models.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == model_name:
                model_details = {
                    'name': row[0],
                    'id': row[1],
                    'keyword': row[2],
                    'version': row[3],
                    'training_images': row[4],
                    'model_type': row[5],
                    'model_url': row[6],
                    'controller_type': row[7]
                }
                return model_details



    

def remove_background(project_name, input_image):
    app_settings = get_app_settings()

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("pollinations/modnet")
    version = model.versions.get("da7d45f3b836795f945f221fc0b01a6d3ab7f5e163f13208948ad436001e2255")

    output = version.predict(image=input_image)
    return output


def replace_background(video_name, foreground_image, background_image):
    if background_image.startswith("http"):
        response = r.get(background_image)
        background_image = Image.open(BytesIO(response.content))
    else:
        background_image = Image.open(f"{background_image}")
    foreground_image = Image.open(f"masked_image.png")
    background_image.paste(foreground_image, (0, 0), foreground_image)
    background_image.save(f"videos/{video_name}/replaced_bg.png")

    return (f"videos/{video_name}/replaced_bg.png")


def prompt_clip_interrogator(input_image, which_model, best_or_fast):

    if which_model == "Stable Diffusion 1.5":
        which_model = "ViT-L-14/openai"
    elif which_model == "Stable Diffusion 2":
        which_model = "ViT-H-14/laion2b_s32b_b79k"

    app_settings = get_app_settings()

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("pharmapsychotic/clip-interrogator")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    version = model.versions.get("a4a8bafd6089e1716b06057c42b19378250d008b80fe87caa5cd36d40c1eda90")

    output = version.predict(
        image=input_image, clip_model_name=which_model, mode=best_or_fast)

    return output


def prompt_model_real_esrgan_upscaling(input_image):
    
    app_settings = get_app_settings()

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("cjwbw/real-esrgan")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    version = model.versions.get("d0ee3d708c9b911f122a4ad90046c5d26a0293b99476d697f6bb7f2e251ce2d4")

    output = version.predict(image=input_image, upscale = 2)

    return output


def touch_up_images(video_name, index_of_current_item, input_image):

    app_settings = get_app_settings()

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("xinntao/gfpgan")
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    version = model.versions.get("6129309904ce4debfde78de5c209bce0022af40e197e132f08be8ccce3050393")

    output = version.predict(img=input_image)

    return output


def resize_image(video_name, new_width, new_height, image):

    response = r.get(image)
    image = Image.open(BytesIO(response.content))
    resized_image = image.resize((new_width, new_height))

    time.sleep(0.1)

    resized_image.save("videos/" + str(video_name) + "/temp_image.png")

    resized_image = upload_image(
        "videos/" + str(video_name) + "/temp_image.png")

    os.remove("videos/" + str(video_name) + "/temp_image.png")

    return resized_image


def face_swap(video_name, index_of_current_item, source_image, timing_details):
    app_settings = get_app_settings()

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("arielreplicate/ghost_face_swap")
    model_id = timing_details[index_of_current_item]["model_id"]

    if model_id == "Dreambooth":
        custom_model = timing_details[index_of_current_item]["custom_models"]
    if model_id == "LoRA":
        custom_model = ast.literal_eval(
            timing_details[index_of_current_item]["custom_models"][1:-1])[0]

    source_face = ast.literal_eval(get_model_details(
        custom_model)["training_images"][1:-1])[0]
    version = model.versions.get(
        "106df0aaf9690354379d8cd291ad337f6b3ea02fe07d90feb1dafd64820066fa")
    target_face = source_image

    if not source_face.startswith("http"):
        source_face = open(source_face, "rb")

    if not target_face.startswith("http"):
        target_face = open(target_face, "rb")

    output = version.predict(source_path=source_face, target_path=target_face)
    return output


def prompt_model_stylegan_nada(index_of_current_item, timing_details, input_image, project_name):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("rinongal/stylegan-nada")
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")
    version = model.versions.get(
        "6b2af4ac56fa2384f8f86fc7620943d5fc7689dcbb6183733743a215296d0e30")
    output = version.predict(
        input=input_image, output_style=timing_details[index_of_current_item]["prompt"])
    output = resize_image(project_name, 512, 512, output)

    return output



def prompt_model_stable_diffusion_xl(project_name, index_of_current_item, timing_details, source_image):

    app_settings = get_app_settings()
    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = app_settings["stability_ai_api_key"]

    # if the image starts with http, it's a URL, otherwise it's a file path
    if source_image.startswith("http"):
        response = r.get(source_image)
        source_image = Image.open(BytesIO(response.content))    
    else:
        source_image = Image.open(source_image)


    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    source_image.save(temp_file, "PNG")
    temp_file.close()
    
    
    source_image.seek(0)

    

    multipart_data = MultipartEncoder(
        fields={
            "text_prompts[0][text]": timing_details[index_of_current_item]["prompt"],
            "init_image": (os.path.basename(temp_file.name), open(temp_file.name, "rb"), "image/png"),
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": str(timing_details[index_of_current_item]["strength"]),
            "cfg_scale": str(timing_details[index_of_current_item]["guidance_scale"]),
            "clip_guidance_preset": "FAST_BLUE",                        
            "samples": "1",
            "steps": str(timing_details[index_of_current_item]["num_inference_steps"]),
        }
    )

    print(multipart_data)

    response = r.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image",
        headers={
            "Content-Type": multipart_data.content_type,
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        data=multipart_data,
    )
    os.unlink(temp_file.name)

    print(response)

    if response.status_code != 200:
        st.error("An error occurred: " + str(response.text))
        time.sleep(5)
        return None
    else:
        data = response.json()
        generated_image = base64.b64decode(data["artifacts"][0]["base64"])
        # generate a random file name with uuid at the location
        file_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(uuid.uuid4()) + ".png"
        
        with open(file_location, "wb") as f:
            f.write(generated_image)
    
    return file_location
        
        
   


    
    
    
    


def prompt_model_stability(project_name, index_of_current_item, timing_details, input_image):

    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    index_of_current_item = int(index_of_current_item)
    prompt = timing_details[index_of_current_item]["prompt"]
    strength = float(timing_details[index_of_current_item]["strength"])
    model = replicate.models.get("cjwbw/stable-diffusion-img2img-v2.1")
    version = model.versions.get("650c347f19a96c8a0379db998c4cd092e0734534591b16a60df9942d11dec15b")    
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb") 
    output = version.predict(image=input_image, prompt_strength=float(strength), prompt=prompt, negative_prompt = timing_details[index_of_current_item]["negative_prompt"], width = int(project_settings["width"]), height = int(project_settings["height"]), guidance_scale = float(timing_details[index_of_current_item]["guidance_scale"]), seed = int(timing_details[index_of_current_item]["seed"]), num_inference_steps = int(timing_details[index_of_current_item]["num_inference_steps"]))
    
    return output[0]


def prompt_model_dreambooth(project_name, image_number, model_name, app_settings,timing_details, project_settings, source_image):

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    replicate_api_key = app_settings["replicate_com_api_key"]
    image_number = int(image_number)
    prompt = timing_details[image_number]["prompt"]
    strength = float(timing_details[image_number]["strength"])
    negative_prompt = timing_details[image_number]["negative_prompt"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    seed = int(timing_details[image_number]["seed"])
    num_inference_steps = int(
        timing_details[image_number]["num_inference_steps"])
    model = replicate.models.get(f"peter942/{model_name}")
    model_details = get_model_details(model_name)
    model_id = model_details["id"]

    if timing_details[image_number]["adapter_type"] == "Yes":
        if source_image.startswith("http"):        
            control_image = source_image
        else:
            control_image = open(source_image, "rb")
    else:
        control_image = None
    
    if model_details["version"] == "":
        headers = {"Authorization": f"Token {replicate_api_key}"}
        url = f"https://dreambooth-api-experimental.replicate.com/v1/trainings/{model_id}"
        response = r.get(url, headers=headers)
        version = (response.json()["version"])
        models_df = pd.read_csv("models.csv")
        row_number = models_df[models_df["id"] == model_id].index[0]
        models_df.iloc[row_number, [3]] = version
        models_df.to_csv("models.csv", index=False)
    else:
        version = model_details["version"]

    version = model.versions.get(version)

    if source_image.startswith("http"):        
        input_image = source_image        
    else:
        input_image = open(source_image, "rb")

    if control_image != None:
        output = version.predict(image=input_image, control_image = control_image, prompt=prompt, prompt_strength=float(strength), height = int(project_settings["height"]), width = int(project_settings["width"]), disable_safety_check=True, negative_prompt = negative_prompt, guidance_scale = float(guidance_scale), seed = int(seed), num_inference_steps = int(num_inference_steps))
    else:
        output = version.predict(image=input_image, prompt=prompt, prompt_strength=float(strength), height = int(project_settings["height"]), width = int(project_settings["width"]), disable_safety_check=True, negative_prompt = negative_prompt, guidance_scale = float(guidance_scale), seed = int(seed), num_inference_steps = int(num_inference_steps))
    
    for i in output:
        return i


def upload_image(image_location):

    app_settings = get_app_settings()

    unique_file_name = str(uuid.uuid4()) + ".png"
    s3_file = f"input_images/{unique_file_name}"
    s3 = boto3.client('s3', aws_access_key_id=app_settings['aws_access_key_id'],
                      aws_secret_access_key=app_settings['aws_secret_access_key'])
    s3.upload_file(image_location, "banodoco", s3_file)
    s3.put_object_acl(ACL='public-read', Bucket='banodoco', Key=s3_file)
    return f"https://s3.amazonaws.com/banodoco/{s3_file}"


def update_slice_of_video_speed(video_name, input_video, desired_speed_change):

    clip = VideoFileClip("videos/" + str(video_name) +
                         "/assets/videos/0_raw/" + str(input_video))

    clip_location = "videos/" + \
        str(video_name) + "/assets/videos/0_raw/" + str(input_video)

    desired_speed_change_text = str(desired_speed_change) + "*PTS"

    video_stream = ffmpeg.input(str(clip_location))

    video_stream = video_stream.filter('setpts', desired_speed_change_text)

    ffmpeg.output(video_stream, "videos/" + str(video_name) +
                  "/assets/videos/0_raw/output_" + str(input_video)).run()

    video_capture = cv2.VideoCapture(
        "videos/" + str(video_name) + "/assets/videos/0_raw/output_" + str(input_video))

    os.remove("videos/" + str(video_name) +
              "/assets/videos/0_raw/" + str(input_video))
    os.rename("videos/" + str(video_name) + "/assets/videos/0_raw/output_" + str(input_video),
              "videos/" + str(video_name) + "/assets/videos/0_raw/" + str(input_video))


def get_duration_from_video(input_video):
    video_capture = cv2.VideoCapture(input_video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_duration = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate
    video_capture.release()
    return total_duration


def get_audio_bytes_for_slice(project_name, index_of_current_item):
    project_settings = get_project_settings(project_name)
    timing_details = get_timing_details(project_name)
    audio = f"videos/{project_name}/assets/resources/audio/{project_settings['audio']}"                    
    audio = AudioSegment.from_file(audio)    
    audio = audio[timing_details[index_of_current_item]['frame_time']*1000:timing_details[index_of_current_item+1]['frame_time']*1000]
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes.seek(0)
    return audio_bytes

def slice_part_of_video(project_name, index_of_current_item, video_start_percentage, video_end_percentage, slice_name, timing_details):
    input_video = timing_details[int(index_of_current_item)]["interpolated_video"]
    total_duration_of_clip = get_duration_from_video(input_video)
    start_time = float(video_start_percentage) * float(total_duration_of_clip)
    end_time = float(video_end_percentage) * float(total_duration_of_clip)
    clip = VideoFileClip(input_video).subclip(t_start=start_time, t_end=end_time)
    output_video = "videos/" + str(project_name) + "/assets/videos/0_raw/" + str(slice_name) + ".mp4"
    clip.write_videofile(output_video, audio=False)
    clip.close()

def update_speed_of_video_clip(project_name, location_of_video, save_to_new_location, index_of_current_item):

    timing_details = get_timing_details(project_name)

    desired_duration = timing_details[int(index_of_current_item)]["duration_of_clip"]

    animation_style = timing_details[int(index_of_current_item)]["animation_style"]

    if animation_style == "Direct Morphing":
        
        # Load the video clip
        clip = VideoFileClip(location_of_video)

        clip = clip.set_fps(120)
        
        # Calculate the number of frames to keep
        input_duration = clip.duration
        total_frames = len(list(clip.iter_frames()))
        target_frames = int(total_frames * (desired_duration / input_duration))

        # Determine which frames to keep
        keep_every_n_frames = total_frames / target_frames
        frames_to_keep = [int(i * keep_every_n_frames) for i in range(target_frames)]

        # Create a new video clip with the selected frames
        updated_clip = concatenate_videoclips([clip.subclip(i/clip.fps, (i+1)/clip.fps) for i in frames_to_keep])

        if save_to_new_location:
            file_name = ''.join(random.choices(
                string.ascii_lowercase + string.digits, k=16)) + ".mp4"
            location_of_video = "videos/" + str(project_name) + "/assets/videos/1_final/" + str(file_name)
        else:
            os.remove(location_of_video)

        updated_clip.write_videofile(location_of_video, codec='libx265')

        clip.close()
        updated_clip.close()


    elif animation_style == "Interpolation":

        clip = VideoFileClip(location_of_video)
        input_video_duration = clip.duration
        desired_duration = timing_details[int(index_of_current_item)]["duration_of_clip"]
        desired_speed_change = float(input_video_duration) / float(desired_duration) 

        print("Desired Speed Change: " + str(desired_speed_change))

        # Apply the speed change using moviepy
        output_clip = clip.fx(vfx.speedx, desired_speed_change)

        # Save the output video
        new_file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16)) + ".mp4"
        new_file_location = "videos/" + str(project_name) + "/assets/videos/1_final/" + str(new_file_name)
        output_clip.write_videofile(new_file_location, codec="libx264", preset="fast")

        if save_to_new_location:        
            location_of_video = new_file_location
        else:
            os.remove(location_of_video)
            location_of_video = new_file_location
    
    return location_of_video



def calculate_desired_duration_of_each_clip(timing_details, project_name):
    for index, timing_detail in enumerate(timing_details):
        total_duration_of_frame = calculate_desired_duration_of_individual_clip(timing_details, index)
        update_specific_timing_value(project_name, index, "duration_of_clip", total_duration_of_frame)

def calculate_desired_duration_of_individual_clip(timing_details, index_of_current_item):
    length_of_list = len(timing_details)

    if index_of_current_item == (length_of_list - 1):
        time_of_frame = timing_details[index_of_current_item]["frame_time"]
        duration_of_static_time = 0.0
        end_duration_of_frame = float(time_of_frame) + float(duration_of_static_time)
        total_duration_of_frame = float(end_duration_of_frame) - float(time_of_frame)
    else:
        time_of_frame = timing_details[index_of_current_item]["frame_time"]
        time_of_next_frame = timing_details[index_of_current_item + 1]["frame_time"]
        total_duration_of_frame = float(time_of_next_frame) - float(time_of_frame)
   
    return total_duration_of_frame


def calculate_desired_duration_of_each_clip(timing_details, project_name):

    length_of_list = len(timing_details)

    for i in timing_details:

        index_of_current_item = timing_details.index(i)
        length_of_list = len(timing_details)

        if index_of_current_item == (length_of_list - 1):

            time_of_frame = timing_details[index_of_current_item]["frame_time"]

            duration_of_static_time = 0.0

            end_duration_of_frame = float(
                time_of_frame) + float(duration_of_static_time)

            total_duration_of_frame = float(
                end_duration_of_frame) - float(time_of_frame)

        else:

            time_of_frame = timing_details[index_of_current_item]["frame_time"]

            time_of_next_frame = timing_details[index_of_current_item +
                                                1]["frame_time"]

            total_duration_of_frame = float(
                time_of_next_frame) - float(time_of_frame)

        duration_of_static_time = 0.0

        duration_of_morph = float(
            total_duration_of_frame) - float(duration_of_static_time)

        update_specific_timing_value(
            project_name, index_of_current_item, "duration_of_clip", total_duration_of_frame)


def hair_swap(source_image, project_name, index_of_current_item):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("cjwbw/style-your-hair")

    version = model.versions.get(
        "c4c7e5a657e2e1abccd57625093522a9928edeccee77e3f55d57c664bcd96fa2")

    source_hair = upload_image("videos/" + str(video_name) + "/face.png")

    target_hair = upload_image("videos/" + str(video_name) +
                               "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png")

    if not source_hair.startswith("http"):
        source_hair = open(source_hair, "rb")

    if not target_hair.startswith("http"):
        target_hair = open(target_hair, "rb")

    output = version.predict(source_image=source_hair,
                             target_image=target_hair)

    return output


def prompt_model_depth2img(strength, image_number, timing_details, source_image):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    prompt = timing_details[image_number]["prompt"]
    num_inference_steps = timing_details[image_number]["num_inference_steps"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    negative_prompt = timing_details[image_number]["negative_prompt"]
    model = replicate.models.get("jagilley/stable-diffusion-depth2img")
    version = model.versions.get(
        "68f699d395bc7c17008283a7cef6d92edc832d8dc59eb41a6cafec7fc70b85bc")

    if not source_image.startswith("http"):
        source_image = open(source_image, "rb")

    output = version.predict(input_image=source_image, prompt_strength=float(strength), prompt=prompt,
                             negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    return output[0]


def prompt_model_blip2(input_image, query):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("salesforce/blip-2")
    version = model.versions.get(
        "4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")
    output = version.predict(image=input_image, question=query)
    print(output)
    return output


def facial_expression_recognition(input_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("phamquiluan/facial-expression-recognition")
    version = model.versions.get(
        "b16694d5bfed43612f1bfad7015cf2b7883b732651c383fe174d4b7783775ff5")
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")
    output = version.predict(input_path=input_image)
    emo_label = output[0]["emo_label"]
    if emo_label == "disgust":
        emo_label = "disgusted"
    elif emo_label == "fear":
        emo_label = "fearful"
    elif emo_label == "surprised":
        emo_label = "surprised"
    emo_proba = output[0]["emo_proba"]
    if emo_proba > 0.95:
        emotion = (f"very {emo_label} expression")
    elif emo_proba > 0.85:
        emotion = (f"{emo_label} expression")
    elif emo_proba > 0.75:
        emotion = (f"somewhat {emo_label} expression")
    elif emo_proba > 0.65:
        emotion = (f"slightly {emo_label} expression")
    elif emo_proba > 0.55:
        emotion = (f"{emo_label} expression")
    else:
        emotion = (f"neutral expression")
    return emotion


def prompt_model_pix2pix(strength, video_name, image_number, timing_details, replicate_api_key, input_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    image_number = int(image_number)
    prompt = timing_details[image_number]["prompt"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    seed = int(timing_details[image_number]["seed"])
    model = replicate.models.get("arielreplicate/instruct-pix2pix")
    version = model.versions.get(
        "10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430")
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")
    output = version.predict(input_image=input_image, instruction_text=prompt,
                             seed=seed, cfg_image=1.2, cfg_text=guidance_scale, resolution=704)

    return output


def restyle_images(index_of_current_item, project_name, project_settings, timing_details, source_image):

    index_of_current_item = int(index_of_current_item)
    model_name = timing_details[index_of_current_item]["model_id"]
    strength = timing_details[index_of_current_item]["strength"]
    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)

    if model_name == "stable-diffusion-img2img-v2.1":
        output_url = prompt_model_stability(
            project_name, index_of_current_item, timing_details, source_image)
    elif model_name == "depth2img":
        output_url = prompt_model_depth2img(
            strength, index_of_current_item, timing_details, source_image)
    elif model_name == "pix2pix":
        output_url = prompt_model_pix2pix(
            strength, project_name, index_of_current_item, timing_details, app_settings, source_image)
    elif model_name == "LoRA":
        output_url = prompt_model_lora(
            project_name, index_of_current_item, timing_details, source_image)
    elif model_name == "controlnet":
        output_url = prompt_model_controlnet(
            timing_details, index_of_current_item, source_image)
    elif model_name == "Dreambooth":
        output_url = prompt_model_dreambooth(project_name, index_of_current_item, timing_details[index_of_current_item]["custom_models"], app_settings,timing_details, project_settings,source_image)
    elif model_name =='StyleGAN-NADA':
        output_url = prompt_model_stylegan_nada(index_of_current_item ,timing_details,source_image,project_name)
    elif model_name == "stable_diffusion_xl":
        output_url = prompt_model_stable_diffusion_xl(project_name, index_of_current_item, timing_details, source_image)
    elif model_name == "real-esrgan-upscaling":
        output_url = prompt_model_real_esrgan_upscaling(source_image)
    elif model_name == 'controlnet_1_1_x_realistic_vision_v2_0':
        output_url = prompt_model_controlnet_1_1_x_realistic_vision_v2_0(timing_details, index_of_current_item,source_image)
    
    return output_url


def custom_pipeline_mystique(index_of_current_item, project_name, project_settings, timing_details, source_image):

    prompt = timing_details[index_of_current_item]["prompt"]

    model_name = timing_details[index_of_current_item]["model_id"]

    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)

    output_url = face_swap(
        project_name, index_of_current_item, source_image, timing_details)
    # output_url = hair_swap(source_image, project_name, index_of_current_item)
    output_url = touch_up_images(
        project_name, index_of_current_item, output_url)
    output_url = resize_image(project_name, int(
        project_settings["width"]), int(project_settings["height"]), output_url)
    if timing_details[index_of_current_item]["model_id"] == "Dreambooth":
        model = timing_details[index_of_current_item]["custom_models"]
        output_url = prompt_model_dreambooth(
            project_name, index_of_current_item, model, app_settings, timing_details, project_settings, output_url)
    elif timing_details[index_of_current_item]["model_id"] == "LoRA":
        output_url = prompt_model_lora(
            project_name, index_of_current_item, timing_details, output_url)

    return output_url


def create_timings_row_at_frame_number(project_name, index_of_new_item):

    csv_processor = CSVProcessor(f'videos/{project_name}/timings.csv')
    project_settings = get_project_settings(project_name)
    
    df = csv_processor.get_df_data()
    new_row = pd.DataFrame({'frame_time': [None], 'frame_number': [None]}, index=[0])
    df = pd.concat([df.iloc[:index_of_new_item], new_row, df.iloc[index_of_new_item:]]).reset_index(drop=True)

    # Set the data types for each column
    column_types = {
        'frame_time': float,
        'frame_number': int,
        'primary_image': int,
        'guidance_scale': float,
        'seed': int,
        'num_inference_steps': float,
        'strength': float
    }
    df = df.astype(column_types, errors='ignore')
    

    df.to_csv(f'videos/{project_name}/timings.csv', index=False)
    'remove the interpolated video from the current row and the row before and after - unless it is the first or last row'
    update_specific_timing_value(project_name, index_of_new_item, "interpolated_video", "")
    default_animation_style = project_settings["default_animation_style"]
    update_specific_timing_value(project_name, index_of_new_item, "animation_style", "")
    

    if index_of_new_item != 0:
        update_specific_timing_value(project_name, index_of_new_item-1, "interpolated_video", "")
    
    if index_of_new_item != len(df)-1:
        update_specific_timing_value(project_name, index_of_new_item+1, "interpolated_video", "")
    
    

    return index_of_new_item


def get_models():
    df = pd.read_csv('models.csv')
    models = df[df.columns[0]].tolist()
    return models


def find_duration_of_clip(index_of_current_item, timing_details, total_number_of_videos):

    total_duration_of_clip = timing_details[index_of_current_item]['duration_of_clip']
    total_duration_of_clip = float(total_duration_of_clip)

    if index_of_current_item == total_number_of_videos:
        total_duration_of_clip = timing_details[index_of_current_item]['duration_of_clip']
        # duration_of_static_time = float(timing_details[index_of_current_item]['static_time'])
        # duration_of_static_time = float(duration_of_static_time) / 2
        duration_of_static_time = 0

    elif index_of_current_item == 0:
        # duration_of_static_time = float(timing_details[index_of_current_item]['static_time'])
        duration_of_static_time = 0
    else:
        # duration_of_static_time = float(timing_details[index_of_current_item]['static_time'])
        duration_of_static_time = 0

    return total_duration_of_clip, duration_of_static_time

def add_audio_to_video_slice(video_location, audio_bytes):
    # Save the audio bytes to a temporary file
    audio_file = "temp_audio.wav"
    with open(audio_file, 'wb') as f:
        f.write(audio_bytes.getvalue())

    # Create an input video stream
    video_stream = ffmpeg.input(video_location)
    
    # Create an input audio stream
    audio_stream = ffmpeg.input(audio_file)

    # Add the audio stream to the video stream
    output_stream = ffmpeg.output(video_stream, audio_stream, "output_with_audio.mp4", vcodec='copy', acodec='aac', strict='experimental')

    # Run the ffmpeg command
    output_stream.run()

    # Remove the original video file and the temporary audio file
    os.remove(video_location)
    os.remove(audio_file)

    # Rename the output file to have the same name as the original video file
    os.rename("output_with_audio.mp4", video_location)

def calculate_desired_speed_change(input_video_location, target_duration):
    # Load the video clip
    input_clip = VideoFileClip(input_video_location)
    
    # Get the duration of the input video clip
    input_duration = input_clip.duration
    
    # Calculate the desired speed change
    desired_speed_change = target_duration / input_duration
    
    return desired_speed_change


def get_actual_clip_duration(clip_location):
    # Load the video clip
    clip = VideoFileClip(clip_location)
    
    # Get the duration of the video clip
    duration = clip.duration

    rounded_duration = round(duration, 5)
    
    return rounded_duration

def calculate_dynamic_interpolations_steps(duration_of_clip):

    if duration_of_clip < 0.17:
        interpolation_steps = 2
    elif duration_of_clip < 0.3:
        interpolation_steps = 3
    elif duration_of_clip < 0.57:
        interpolation_steps = 4
    elif duration_of_clip < 1.1:
        interpolation_steps = 5
    elif duration_of_clip < 2.17:
        interpolation_steps = 6
    elif duration_of_clip < 4.3:
        interpolation_steps = 7
    else:
        interpolation_steps = 8
    return interpolation_steps

def render_video(project_name, final_video_name, timing_details, quality):

    timing_details = get_timing_details(project_name)
    calculate_desired_duration_of_each_clip(timing_details,project_name)
    total_number_of_videos = len(timing_details) - 1

    for i in range(0, total_number_of_videos):
        index_of_current_item = i
        if quality == "High-Quality":     
            update_specific_timing_value(project_name, index_of_current_item, "timing_video", "")       
            timing_details = get_timing_details(project_name)
            interpolation_steps = calculate_dynamic_interpolations_steps(timing_details[index_of_current_item]["duration_of_clip"])
            if timing_details[index_of_current_item]["interpolation_steps"] == "" or timing_details[index_of_current_item]["interpolation_steps"] < interpolation_steps:
                update_specific_timing_value(project_name, index_of_current_item, "interpolation_steps", interpolation_steps)
                update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", "")                                
        else:
            if timing_details[index_of_current_item]["interpolation_steps"] == "" or timing_details[index_of_current_item]["interpolation_steps"] < 3:
                update_specific_timing_value(project_name, index_of_current_item, "interpolation_steps", 3)
        
        timing_details = get_timing_details(project_name)

        if timing_details[index_of_current_item]["interpolated_video"] == "":                                                        

            if total_number_of_videos == index_of_current_item:                                            
                video_location = create_individual_clip(index_of_current_item, project_name)
                update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", video_location)
            else:                
                video_location =  create_individual_clip(index_of_current_item, project_name)
                update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", video_location)
                

    project_settings = get_project_settings(project_name)
    timing_details = get_timing_details(project_name)
    total_number_of_videos = len(timing_details) - 2    
    timing_details = get_timing_details(project_name)

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        if index_of_current_item <= total_number_of_videos:
            
            if timing_details[index_of_current_item]["timing_video"] == "":                                                             
                desired_duration = float(timing_details[index_of_current_item]["duration_of_clip"])                
                location_of_input_video = timing_details[index_of_current_item]["interpolated_video"]                               
                duration_of_input_video = float(get_duration_from_video(location_of_input_video))                                                                                                
                speed_change_required = float(desired_duration/duration_of_input_video)                                
                location_of_output_video = update_speed_of_video_clip(project_name, location_of_input_video, True, index_of_current_item)
                if quality == "Preview":
                    print("")
                    '''
                    clip = VideoFileClip(location_of_output_video)

                    number_text = TextClip(str(index_of_current_item), fontsize=24, color='white')
                    number_background = TextClip(" ", fontsize=24, color='black', bg_color='black', size=(number_text.w + 10, number_text.h + 10))
                    number_background = number_background.set_position(('right', 'bottom')).set_duration(clip.duration)
                    number_text = number_text.set_position((number_background.w - number_text.w - 5, number_background.h - number_text.h - 5)).set_duration(clip.duration)

                    clip_with_number = CompositeVideoClip([clip, number_background, number_text])

                    # remove existing preview video
                    os.remove(location_of_output_video)
                    clip_with_number.write_videofile(location_of_output_video, codec='libx264', bitrate='3000k')
                    '''
                update_specific_timing_value(project_name, index_of_current_item, "timing_video", location_of_output_video)                
                


    video_list = []

    timing_details = get_timing_details(project_name)

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        if index_of_current_item <= total_number_of_videos:
            index_of_current_item = timing_details.index(i)
            video_location = timing_details[index_of_current_item]["timing_video"]
            video_list.append(video_location)

    '''
    input_files = [ffmpeg.input(v) for v in video_list]

    concatenated = ffmpeg.concat(*input_files, v=1).output('concatenated.mp4')
    ffmpeg.run(concatenated)

    # Final output video file location
    output_video_file = f"videos/{project_name}/assets/videos/2_completed/{final_video_name}.mp4"

    # Add audio if it is provided
    if project_settings['audio'] != "":
        audio_location = f"videos/{project_name}/assets/resources/audio/{project_settings['audio']}"
        input_video = ffmpeg.input('concatenated.mp4')
        input_audio = ffmpeg.input(audio_location)
        final_out = ffmpeg.output(input_video, input_audio, output_video_file, vcodec="libx264", acodec="aac", ab="128k", vb="5000k")
        ffmpeg.run(final_out)
    
    '''

    video_clips = [VideoFileClip(v) for v in video_list]
    finalclip = concatenate_videoclips(video_clips)
    output_video_file = f"videos/{project_name}/assets/videos/2_completed/{final_video_name}.mp4"
    if project_settings['audio'] != "":
        audio_location = f"videos/{project_name}/assets/resources/audio/{project_settings['audio']}"
        audio_clip = AudioFileClip(audio_location)
        finalclip = finalclip.set_audio(audio_clip)
    

    finalclip.write_videofile(
        output_video_file, 
        fps=30,  # or 60 if your original video is 60fps
        audio_bitrate="128k", 
        bitrate="5000k", 
        codec="libx264", 
        audio_codec="aac")
     
    

def create_gif_preview(project_name, timing_details):

    list_of_images = []
    for i in timing_details:
        # make index_of_current_item the index of the current item
        index_of_current_item = timing_details.index(i)
        variants = timing_details[index_of_current_item]["alternative_images"]
        primary_image = int(
            timing_details[index_of_current_item]["primary_image"])
        source_image = variants[primary_image]
        list_of_images.append(source_image)

    frames = []
    for url in list_of_images:
        response = r.get(url)
        frame = Image.open(BytesIO(response.content))
        draw = ImageDraw.Draw(frame)
        font_url = 'https://banodoco.s3.amazonaws.com/training_data/arial.ttf'
        font_file = "arial.ttf"
        urllib.request.urlretrieve(font_url, font_file)
        font = ImageFont.truetype(font_file, 40)
        index_of_current_item = list_of_images.index(url)
        draw.text((frame.width - 60, frame.height - 60),
                  str(index_of_current_item), font=font, fill=(255, 255, 255, 255))
        frames.append(np.array(frame))
    imageio.mimsave(f'videos/{project_name}/preview_gif.gif', frames, fps=0.5)


def create_depth_mask_image(input_image, layer, project_name, index_of_current_item):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("cjwbw/midas")
    version = model.versions.get(
        "a6ba5798f04f80d3b314de0f0a62277f21ab3503c60c84d4817de83c5edfdae0")
    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")
    output = version.predict(
        image=input_image, model_type="dpt_beit_large_512")
    try:
        urllib.request.urlretrieve(output, "depth.png")
    except Exception as e:
        print(e)

    depth_map = Image.open("depth.png")
    depth_map = depth_map.convert("L")  # Convert to grayscale image
    pixels = depth_map.load()
    mask = Image.new("L", depth_map.size)
    mask_pixels = mask.load()

    fg_mask = Image.new("L", depth_map.size) if "Foreground" in layer else None
    mg_mask = Image.new("L", depth_map.size) if "Middleground" in layer else None
    bg_mask = Image.new("L", depth_map.size) if "Background" in layer else None

    fg_pixels = fg_mask.load() if fg_mask else None
    mg_pixels = mg_mask.load() if mg_mask else None
    bg_pixels = bg_mask.load() if bg_mask else None

    for i in range(depth_map.size[0]):
        for j in range(depth_map.size[1]):
            depth_value = pixels[i, j]

            if fg_pixels:
                fg_pixels[i, j] = 0 if depth_value > 200 else 255
            if mg_pixels:
                mg_pixels[i, j] = 0 if depth_value <= 200 and depth_value > 50 else 255
            if bg_pixels:
                bg_pixels[i, j] = 0 if depth_value <= 50 else 255

            mask_pixels[i, j] = 255
            if fg_pixels:
                mask_pixels[i, j] &= fg_pixels[i, j]
            if mg_pixels:
                mask_pixels[i, j] &= mg_pixels[i, j]
            if bg_pixels:
                mask_pixels[i, j] &= bg_pixels[i, j]


    return create_or_update_mask(project_name, index_of_current_item, mask)


def prompt_model_controlnet(timing_details, index_of_current_item, input_image):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    if timing_details[index_of_current_item]["adapter_type"] == "normal":
        model = replicate.models.get("jagilley/controlnet-normal")
        version = model.versions.get("cc8066f617b6c99fdb134bc1195c5291cf2610875da4985a39de50ee1f46d81c")
    elif timing_details[index_of_current_item]["adapter_type"] == "canny":
        model = replicate.models.get("jagilley/controlnet-canny")
        version = model.versions.get("aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613")
    elif timing_details[index_of_current_item]["adapter_type"] == "hed":
        model = replicate.models.get("jagilley/controlnet-hed")
        version = model.versions.get("cde353130c86f37d0af4060cd757ab3009cac68eb58df216768f907f0d0a0653")
    elif timing_details[index_of_current_item]["adapter_type"] == "scribble":
        model = replicate.models.get("jagilley/controlnet-scribble")
        version = model.versions.get("435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117")
        if timing_details[index_of_current_item]["canny_image"] != "":
            input_image = timing_details[index_of_current_item]["canny_image"]
    elif timing_details[index_of_current_item]["adapter_type"] == "seg":
        model = replicate.models.get("jagilley/controlnet-seg")
        version = model.versions.get("f967b165f4cd2e151d11e7450a8214e5d22ad2007f042f2f891ca3981dbfba0d")
    elif timing_details[index_of_current_item]["adapter_type"] == "hough":
        model = replicate.models.get("jagilley/controlnet-hough")
        version = model.versions.get("854e8727697a057c525cdb45ab037f64ecca770a1769cc52287c2e56472a247b")
    elif timing_details[index_of_current_item]["adapter_type"] == "depth2img":
        model = replicate.models.get("jagilley/controlnet-depth2img")
        version = model.versions.get("922c7bb67b87ec32cbc2fd11b1d5f94f0ba4f5519c4dbd02856376444127cc60")
    elif timing_details[index_of_current_item]["adapter_type"] == "pose":
        model = replicate.models.get("jagilley/controlnet-pose")
        version = model.versions.get("0304f7f774ba7341ef754231f794b1ba3d129e3c46af3022241325ae0c50fb99")

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    inputs = {
        'image': input_image,
        'prompt': timing_details[index_of_current_item]["prompt"],
        'num_samples': "1",
        'image_resolution': "512",
        'ddim_steps': int(timing_details[index_of_current_item]["num_inference_steps"]),
        'scale': float(timing_details[index_of_current_item]["guidance_scale"]),
        'eta': 0,
        'seed': int(timing_details[index_of_current_item]["seed"]),
        'a_prompt': "best quality, extremely detailed",
        'n_prompt': timing_details[index_of_current_item]["negative_prompt"] + ", longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        'detect_resolution': 512,
        'bg_threshold': 0,
        'low_threshold': int(timing_details[index_of_current_item]["low_threshold"]),
        'high_threshold': int(timing_details[index_of_current_item]["high_threshold"]),
    }
    
    output = version.predict(**inputs)

    return output[1]

def prompt_model_controlnet_1_1_x_realistic_vision_v2_0(timing_details, index_of_current_item, input_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    if not input_image.startswith("http"):
        input_image = open(input_image, "rb")

    model = replicate.models.get("usamaehsan/controlnet-1.1-x-realistic-vision-v2.0")
    version = model.versions.get("7fbf4c86671738f97896c9cb4922705adfcdcf54a6edab193bb8c176c6b34a69")
    
    inputs = {

        'image': input_image,
        'prompt': timing_details[index_of_current_item]["prompt"],
        'ddim_steps': int(timing_details[index_of_current_item]["num_inference_steps"]),
        'strength': float(timing_details[index_of_current_item]["strength"]),
        'scale': float(timing_details[index_of_current_item]["guidance_scale"]),
        'seed': int(timing_details[index_of_current_item]["seed"]),

    }

    output = version.predict(**inputs)

    return output[1]


def prompt_model_lora(project_name, index_of_current_item, timing_details, source_image):

    app_settings = get_app_settings()

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    timing_details = get_timing_details(project_name)
    lora_models = ast.literal_eval(timing_details[index_of_current_item]["custom_models"][1:-1])
    project_settings = get_project_settings(project_name)
    default_model_url = "https://replicate.delivery/pbxt/nWm6eP9ojwVvBCaWoWZVawOKRfgxPJmkVk13ES7PX36Y66kQA/tmpxuz6k_k2datazip.safetensors"

    lora_model_urls = []
    
    for lora_model in lora_models:
        if lora_model != "":
            lora_model_details = get_model_details(lora_model)
            print(lora_model_details)
            if lora_model_details['model_url'] != "":
                lora_model_url = lora_model_details['model_url']
            else:
                lora_model_url = default_model_url
        else:
            lora_model_url = default_model_url
        lora_model_urls.append(lora_model_url)

    lora_model_1_model_url = lora_model_urls[0]
    lora_model_2_model_url = lora_model_urls[1]
    lora_model_3_model_url = lora_model_urls[2]    

    if source_image[:4] == "http":
        input_image = source_image
    else:
        input_image = open(source_image, "rb")
    
    if timing_details[index_of_current_item]["adapter_type"] != "None":
        if source_image[:4] == "http":
            adapter_condition_image = source_image
        else:
            adapter_condition_image = open(source_image, "rb")
    else:
        adapter_condition_image = ""

    model = replicate.models.get("cloneofsimo/lora")
    version = model.versions.get(
        "fce477182f407ffd66b94b08e761424cabd13b82b518754b83080bc75ad32466")
    inputs = {
    'prompt': timing_details[index_of_current_item]["prompt"],
    'negative_prompt': timing_details[index_of_current_item]["negative_prompt"],
    'width': int(project_settings["width"]),
    'height': int(project_settings["height"]),
    'num_outputs': 1,
    'image': input_image,
    'num_inference_steps': int(timing_details[index_of_current_item]["num_inference_steps"]),
    'guidance_scale': float(timing_details[index_of_current_item]["guidance_scale"]),
    'prompt_strength': float(timing_details[index_of_current_item]["strength"]),
    'scheduler': "DPMSolverMultistep",
    'lora_urls': lora_model_1_model_url + "|" + lora_model_2_model_url + "|" + lora_model_3_model_url,
    'lora_scales': "0.5 | 0.5 | 0.5",
    'adapter_type': timing_details[index_of_current_item]["adapter_type"],
    'adapter_condition_image': adapter_condition_image,
     }
    
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        try:
            output = version.predict(**inputs)
            print(output)
            return output[0]
        except replicate.exceptions.ModelError as e:
            if "NSFW content detected" in str(e):
                print("NSFW content detected. Attempting to rerun code...")
                attempts += 1
                continue
            else:
                raise e
    return "https://i.ibb.co/ZG0hxzj/Failed-3x-In-A-Row.png"


def attach_audio_element(project_name, project_settings, expanded):
    with st.expander("Audio"):
        uploaded_file = st.file_uploader("Attach audio", type=[
                                         "mp3"], help="This will attach this audio when you render a video")
        if st.button("Upload and attach new audio"):
            with open(os.path.join(f"videos/{project_name}/assets/resources/audio", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
                update_project_setting(
                    "audio", uploaded_file.name, project_name)
                st.experimental_rerun()
        if project_settings["audio"] == "extracted_audio.mp3":
            st.info("You have attached the audio from the video you uploaded.")
        if project_settings["audio"] != "":
            st.audio(
                f"videos/{project_name}/assets/resources/audio/{project_settings['audio']}")


def execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, editing_image, prompt, negative_prompt, width, height, layer, index_of_current_item):

    if type_of_mask_selection == "Automated Background Selection":
        removed_background = remove_background(project_name, editing_image)
        response = r.get(removed_background)
        with open("masked_image.png", "wb") as f:
            f.write(response.content)
        if type_of_mask_replacement == "Replace With Image":
            replace_background(
                project_name, "masked_image.png", background_image)
            edited_image = upload_image(
                f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":
            image = Image.open("masked_image.png")
            converted_image = Image.new("RGB", image.size, (255, 255, 255))
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    if pixel[3] == 0:
                        converted_image.putpixel((x, y), (0, 0, 0))
                    else:
                        converted_image.putpixel((x, y), (255, 255, 255))
            create_or_update_mask(
                project_name, index_of_current_item, converted_image)
            edited_image = inpainting(
                project_name, editing_image, prompt, negative_prompt, index_of_current_item, True)

    elif type_of_mask_selection == "Manual Background Selection":
        if type_of_mask_replacement == "Replace With Image":
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content))
            else:
                bg_img = Image.open(editing_image)
            timing_detials = get_timing_details(project_name)
            mask_location = timing_detials[index_of_current_item]["mask"]
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                mask_img = Image.open(BytesIO(response.content))
            else:
                mask_img = Image.open(mask_location)

            result_img = Image.new("RGBA", bg_img.size, (255, 255, 255, 0))
            for x in range(bg_img.size[0]):
                for y in range(bg_img.size[1]):
                    if mask_img.getpixel((x, y)) == (0, 0, 0, 255):
                        result_img.putpixel((x, y), (255, 255, 255, 0))
                    else:
                        result_img.putpixel((x, y), bg_img.getpixel((x, y)))
            result_img.save("masked_image.png")
            replace_background(
                project_name, "masked_image.png", background_image)
            edited_image = upload_image(
                f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":
            timing_detials = get_timing_details(project_name)
            mask_location = timing_detials[index_of_current_item]["mask"]
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                im = Image.open(BytesIO(response.content))
            else:
                im = Image.open(mask_location)
            if "A" in im.getbands():
                mask = Image.new('RGB', (width, height), color=(255, 255, 255))
                mask.paste(im, (0, 0), im)
                create_or_update_mask(
                    project_name, index_of_current_item, mask)
            edited_image = inpainting(
                project_name, editing_image, prompt, negative_prompt, index_of_current_item,True)
    elif type_of_mask_selection == "Automated Layer Selection":
        mask_location = create_depth_mask_image(
            editing_image, layer, project_name, index_of_current_item)
        if type_of_mask_replacement == "Replace With Image":
            if mask_location.startswith("http"):
                mask = Image.open(
                    BytesIO(r.get(mask_location).content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new(
                'RGBA', bg_img.size, (0, 0, 0, 0)), mask)
            masked_img.save("masked_image.png")
            replace_background(
                project_name, "masked_image.png", background_image)
            edited_image = upload_image(
                f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(
                project_name, editing_image, prompt, negative_prompt, index_of_current_item, True)

    elif type_of_mask_selection == "Re-Use Previous Mask":
        timing_detials = get_timing_details(project_name)
        mask_location = timing_detials[index_of_current_item]["mask"]
        if type_of_mask_replacement == "Replace With Image":
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                mask = Image.open(BytesIO(response.content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new(
                'RGBA', bg_img.size, (0, 0, 0, 0)), mask)
            masked_img.save("masked_image.png")
            replace_background(
                project_name, "masked_image.png", background_image)
            edited_image = upload_image(
                f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(
                project_name, editing_image, prompt, negative_prompt, index_of_current_item, True)
    
    elif type_of_mask_selection == "Invert Previous Mask":
        timing_detials = get_timing_details(project_name)
        mask_location = timing_detials[index_of_current_item]["mask"]
        if type_of_mask_replacement == "Replace With Image":
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                mask = Image.open(BytesIO(response.content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            inverted_mask = ImageOps.invert(mask)
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new(
                'RGBA', bg_img.size, (0, 0, 0, 0)), inverted_mask)
            masked_img.save("masked_image.png")
            replace_background(
                project_name, "masked_image.png", background_image)
            edited_image = upload_image(
                f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":            
            edited_image = inpainting(
                project_name, editing_image, prompt, negative_prompt, index_of_current_item, False)




    return edited_image


def page_switcher(pages, page):

    section = [section["section_name"]
               for section in pages if page in section["pages"]][0]
    index_of_section = [section["section_name"]
                        for section in pages].index(section)
    index_of_page_in_section = pages[index_of_section]["pages"].index(page)

    return index_of_page_in_section, index_of_section
