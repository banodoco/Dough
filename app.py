import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_comparison import image_comparison
from moviepy.editor import *
import cv2
import re
from streamlit_javascript import st_javascript
from moviepy.video.io.VideoFileClip import VideoFileClip
import csv
import pandas as pd
import replicate
import urllib
import requests as r
import shutil
import ffmpeg
import datetime
import string
import json
import boto3
import time
import zipfile
import random
import uuid
from pathlib import Path
from io import BytesIO
import ast
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.set_page_config(page_title="Banodoco")

def create_or_update_mask(project_name, index_of_current_number, image):
    timing_details = get_timing_details(project_name)
    if timing_details[index_of_current_number]["mask"] == "":
        unique_file_name = str(uuid.uuid4()) + ".png"
        update_specific_timing_value(project_name, index_of_current_number, "mask", f"videos/{project_name}/assets/resources/masks/{unique_file_name}")
        timing_details = get_timing_details(project_name)
    else:                                        
        unique_file_name = timing_details[st.session_state['which_image']]["mask"].split("/")[-1]                                                                                                                                                                                     
    file_location = f"videos/{project_name}/assets/resources/masks/{unique_file_name}"
    image.save(file_location, "PNG")
    return file_location

def inpainting(video_name, input_image, prompt, negative_prompt,index_of_current_item):

    app_settings = get_app_settings()
    timing_details = get_timing_details(video_name)        
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")
    version = model.versions.get("e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")    
    mask = timing_details[index_of_current_item]["mask"]

    if not mask.startswith("http"):
        mask = open(mask, "rb")        
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")

    output = version.predict(mask=mask, image=input_image,prompt=prompt, invert_mask=True, negative_prompt=negative_prompt,num_inference_steps=25)    

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

    update_specific_timing_value(project_name, index_of_current_item, "alternative_images", '"' + str(alternative_images) + '"')

    if str(timing_details[index_of_current_item]["primary_image"]) == "":
        timing_details[index_of_current_item]["primary_image"] = 0
        update_specific_timing_value(project_name, index_of_current_item, "primary_image", timing_details[index_of_current_item]["primary_image"])        
    return len(additions) + 1
                    
def promote_image_variant(index_of_current_item, project_name, variant_to_promote):
    
    update_specific_timing_value(project_name, index_of_current_item, "primary_image", variant_to_promote)    
    update_specific_timing_value(project_name, index_of_current_item -1, "interpolated_video", "")
    update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(project_name, index_of_current_item +1, "interpolated_video", "")
    update_specific_timing_value(project_name, index_of_current_item -1, "timing_video", "")
    update_specific_timing_value(project_name, index_of_current_item, "timing_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(project_name, index_of_current_item +1, "timing_video", "")
    

def train_model(app_settings, images_list, instance_prompt,class_prompt, max_train_steps, model_name,project_name, type_of_model, type_of_task, resolution):
    
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
    upload_url = response.json()["upload_url"]
    serving_url = response.json()["serving_url"]
    with open('images.zip', 'rb') as f:
        r.put(upload_url, data=f, headers=headers)
    training_file_url = serving_url
    url = "https://dreambooth-api-experimental.replicate.com/v1/trainings"
    os.remove('images.zip')

    if type_of_model == "Dreambooth":
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
        version = model.versions.get("b2a308762e36ac48d16bfadc03a65493fe6e799f429f7941639a6acec5b276cc")
        output = version.predict(instance_data = training_file_url,task=type_of_task, resolution=int(resolution))        
        df = pd.read_csv("models.csv")
        df = df.append({}, ignore_index=True)
        new_row_index = df.index[-1]
        df.iloc[new_row_index, 0] = model_name
        df.iloc[new_row_index, 4] = str(images_list)
        df.iloc[new_row_index, 5] = "LoRA"
        df.iloc[new_row_index, 6] = output
        df.to_csv("models.csv", index=False)
        return f"Successfully trained - the model '{model_name}' is now available for use!"       

def remove_existing_timing(project_name):

    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")
    df = df.drop(df.index[0:])
    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)

def move_frame(project_name, index_of_current_item, distance_to_move, timing_details,input_video):
    
    current_frame_number = int(calculate_frame_number_at_time(input_video, timing_details[index_of_current_item]["frame_time"], project_name))

    if distance_to_move == 0:                        
        extract_frame(index_of_current_item, project_name, input_video, current_frame_number,timing_details)
        new_frame_number = current_frame_number
        
    elif distance_to_move > 0:    
        next_frame_number = int(calculate_frame_number_at_time(input_video, timing_details[index_of_current_item + 1]["frame_time"],project_name))
        abs_distance_to_move = abs(distance_to_move) / 100
        difference_between_frames = abs(next_frame_number - current_frame_number)
        new_frame_number = current_frame_number + (difference_between_frames * abs_distance_to_move)
        extract_frame(index_of_current_item, project_name, input_video, new_frame_number,timing_details)

    elif distance_to_move < 0:
        last_frame_number = int(calculate_frame_number_at_time(input_video, timing_details[index_of_current_item - 1]["frame_time"],project_name))
        abs_distance_to_move = abs(distance_to_move) / 100
        difference_between_frames = abs(current_frame_number - last_frame_number)
        new_frame_number = current_frame_number - (difference_between_frames * abs_distance_to_move)
        extract_frame(index_of_current_item, project_name, input_video, new_frame_number,timing_details)

    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")
    new_time = calculate_time_at_frame_number(input_video, new_frame_number, project_name)
    df.iloc[index_of_current_item, [16,1]] = [int(distance_to_move),new_time]
    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)
      
def get_app_settings():

    csv_file_path = "app_settings.csv"    
    df = pd.read_csv(csv_file_path, header=None,na_filter=False)
    app_settings = {row[0]: row[1] for _, row in df.iterrows()}
    return app_settings


def get_project_settings(project_name):

    project_settings = {}
    data = pd.read_csv("videos/" + str(project_name)  + "/settings.csv", na_filter=False)
    for i, row in data.iterrows():
        project_settings[row['key']] = row['value']
    return project_settings

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
                    'model_url': row[6]
                }
                return model_details


def update_app_setting(key, pair_value):
    
    csv_file_path = 'app_settings.csv'    
    df = pd.read_csv(csv_file_path,na_filter=False)        
    row_number = df[df.iloc[:, 0] == key].index[0]
    df.at[row_number, df.columns[1]] = pair_value
    
    df.to_csv(csv_file_path, index=False)

def create_working_assets(video_name):

    os.mkdir("videos/" + video_name)
    os.mkdir("videos/" + video_name + "/assets")

    os.mkdir("videos/" + video_name + "/assets/frames")

    os.mkdir("videos/" + video_name + "/assets/frames/0_extracted")
    os.mkdir("videos/" + video_name + "/assets/frames/1_selected")
    os.mkdir("videos/" + video_name + "/assets/frames/2_character_pipeline_completed")
    os.mkdir("videos/" + video_name + "/assets/frames/3_backdrop_pipeline_completed")

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

    data = {'key': ['last_prompt', 'last_model','last_strength','last_custom_pipeline','audio', 'input_type', 'input_video','extraction_type','width','height','last_negative_prompt','last_guidance_scale','last_seed','last_num_inference_steps','last_which_stage_to_run_on','last_custom_models','last_adapter_type'],
        'value': ['prompt', 'controlnet', '0.5','None','', 'video', '','Extract manually','','','',7.5,0,50,'Extracted Frames',"None",""]}

    df = pd.DataFrame(data)
    df.to_csv(f'videos/{video_name}/settings.csv', index=False)
    df = pd.DataFrame(columns=['frame_time','frame_number','primary_image','alternative_images','custom_pipeline','negative_prompt','guidance_scale','seed','num_inference_steps','model_id','strength','notes','source_image','custom_models','adapter_type','duration_of_clip','interpolated_video','timing_video','prompt','mask'])
    df.to_csv(f'videos/{video_name}/timings.csv', index=False)

def update_project_setting(key, pair_value, project_name):
    
    csv_file_path = f'videos/{project_name}/settings.csv'
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = []
        for row in csv_reader:
            if row[0] == key:            
                row_number = csv_reader.line_num - 2            
                new_value = pair_value        
    
    df = pd.read_csv(csv_file_path)
    df.iat[row_number, 1] = new_value
    df.to_csv(csv_file_path, index=False)

def prompt_interpolation_model(img1, img2, project_name, video_number, interpolation_steps, replicate_api_key):

    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    model = replicate.models.get("google-research/frame-interpolation")

    if not img1.startswith("http"):
        img1 = open(img1, "rb")

    if not img2.startswith("http"):
        img2 = open(img2, "rb")

    output = model.predict(frame1=img1, frame2=img2, times_to_interpolate=interpolation_steps)
    file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 16)) + ".mp4"

    video_location = "videos/" + project_name + \
        "/assets/videos/0_raw/" + str(file_name)
    try:
        urllib.request.urlretrieve(output, video_location)

    except Exception as e:
        print(e)

    clip = VideoFileClip(video_location)
    update_specific_timing_value(project_name, video_number, "interpolated_video", video_location)
    update_specific_timing_value(project_name, video_number, "timing_video", "")

    

def get_timing_details(video_name):

    file_path = "videos/" + str(video_name) + "/timings.csv"
    df = pd.read_csv(file_path, na_filter=False)

    # Evaluate the alternative_images column and replace it with the evaluated list
    df['alternative_images'] = df['alternative_images'].fillna('').apply(lambda x: ast.literal_eval(x[1:-1]) if x != '' else '')
    
    return df.to_dict('records')

def calculate_time_at_frame_number(input_video, frame_number, project_name):

    input_video = "videos/" + str(project_name) + "/assets/resources/input_videos/" + str(input_video)
    video = cv2.VideoCapture(input_video)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_percentage = float(frame_number / frame_count)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    time_at_frame = float(frame_percentage * length_of_video)
    return time_at_frame

def calculate_frame_number_at_time(input_video, time_of_frame, project_name):

    time_of_frame = float(time_of_frame)
    input_video = "videos/" + str(project_name) + "/assets/resources/input_videos/" + str(input_video)
    video = cv2.VideoCapture(input_video)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    percentage_of_video = float(time_of_frame / length_of_video)
    frame_number = int(percentage_of_video * frame_count)
    if frame_number == 0:
        frame_number = 1
    return frame_number

def remove_background(project_name, input_image):
    app_settings = get_app_settings()

    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("pollinations/modnet")    
    output = model.predict(image=input_image)
    return output

def replace_background(video_name, foreground_image, background_image):
    
    if background_image.startswith("http"):
        response = r.get(background_image)
        background_image = Image.open(BytesIO(response.content))
    else:
        background_image = Image.open(f"videos/{video_name}/assets/resources/backgrounds/{background_image}")        
    foreground_image = Image.open(f"masked_image.png")
    background_image.paste(foreground_image, (0, 0), foreground_image)
    background_image.save(f"videos/{video_name}/replaced_bg.png")

    return (f"videos/{video_name}/replaced_bg.png")
    
def extract_frame(frame_number, project_name, input_video, extract_frame_number,timing_details):

    input_video = "videos/" + str(project_name) + "/assets/resources/input_videos/" + str(input_video)
    input_video = cv2.VideoCapture(input_video)
    total_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    if extract_frame_number == total_frames:
        extract_frame_number = int(total_frames - 1)
    input_video.set(cv2.CAP_PROP_POS_FRAMES, extract_frame_number)
    ret, frame = input_video.read()
    
    if timing_details[frame_number]["frame_number"] == "":
        update_specific_timing_value(project_name, frame_number, "frame_number", extract_frame_number)
    
    file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 16)) + ".png"
    cv2.imwrite("videos/" + project_name + "/assets/frames/1_selected/" + str(file_name), frame)
    update_specific_timing_value(project_name, frame_number, "source_image", "videos/" + project_name + "/assets/frames/1_selected/" + str(file_name))
        

def prompt_clip_interrogator(input_image,which_model,best_or_fast):

    if which_model == "Stable Diffusion 1.5":
        which_model = "ViT-L-14/openai"
    elif which_model == "Stable Diffusion 2":
        which_model = "ViT-H-14/laion2b_s32b_b79k"

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("pharmapsychotic/clip-interrogator")    
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")        
    output = model.predict(image=input_image, clip_model_name=which_model, mode=best_or_fast)
    
    return output


def touch_up_images(video_name, index_of_current_item, input_image):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("xinntao/gfpgan")    

    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")
    
    output = model.predict(img=input_image)
    
    return output

def resize_image(video_name, new_width,new_height, image):

    response = r.get(image)
    image = Image.open(BytesIO(response.content))
    resized_image = image.resize((new_width, new_height))
    time.sleep(0.1)
    resized_image.save("videos/" + str(video_name) + "/temp_image.png")
    resized_image = upload_image("videos/" + str(video_name) + "/temp_image.png")
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
        custom_model = ast.literal_eval(timing_details[index_of_current_item]["custom_models"][1:-1])[0]
                    
    source_face = ast.literal_eval(get_model_details(custom_model)["training_images"][1:-1])[0]
    version = model.versions.get("106df0aaf9690354379d8cd291ad337f6b3ea02fe07d90feb1dafd64820066fa")
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
    output = model.predict(input=input_image,output_style = timing_details[index_of_current_item]["prompt"])        
    output = resize_image(project_name, 512, 512, output)
    
    return output

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
    new_image = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png" 

    return output[0]


def delete_frame(project_name, index_of_current_item):

    update_specific_timing_value(project_name, index_of_current_item -1, "interpolated_video", "")    
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(project_name, index_of_current_item +1, "interpolated_video", "")
    update_specific_timing_value(project_name, index_of_current_item -1, "timing_video", "")    
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(project_name, index_of_current_item +1, "timing_video", "")    

    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")    
    df = df.drop([int(index_of_current_item)])
    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)

def prompt_model_dreambooth(project_name, image_number, model_name, app_settings,timing_details, project_settings, image_url):

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    replicate_api_key = app_settings["replicate_com_api_key"]
    image_number = int(image_number)
    prompt = timing_details[image_number]["prompt"]
    strength = float(timing_details[image_number]["strength"])
    negative_prompt = timing_details[image_number]["negative_prompt"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    seed = int(timing_details[image_number]["seed"])    
    num_inference_steps = int(timing_details[image_number]["num_inference_steps"])    
    model = replicate.models.get(f"peter942/{model_name}")
    model_details = get_model_details(model_name)    
    model_id = model_details["id"]

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

    input_image = image_url
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")
    
    output = version.predict(image=input_image, prompt=prompt, prompt_strength=float(strength), height = int(project_settings["height"]), width = int(project_settings["width"]), disable_safety_check=True, negative_prompt = negative_prompt, guidance_scale = float(guidance_scale), seed = int(seed), num_inference_steps = int(num_inference_steps))
    new_image = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(image_number) + ".png"
    
    try:
        urllib.request.urlretrieve(output[0], new_image)
    except Exception as e:
        print(e)
    return output[0]


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

def slice_part_of_video(project_name, index_of_current_item, video_start_percentage, video_end_percentage, slice_name,timing_details):

    input_video = timing_details[int(index_of_current_item)]["interpolated_video"]
    video_capture = cv2.VideoCapture(input_video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_duration_of_clip = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate
    start_time = float(video_start_percentage) * float(total_duration_of_clip)
    end_time = float(video_end_percentage) * float(total_duration_of_clip)
    clip = VideoFileClip(input_video).subclip(
        t_start=start_time, t_end=end_time)
    output_video = "videos/" + \
        str(project_name) + "/assets/videos/0_raw/" + str(slice_name) + ".mp4"
    clip.write_videofile(output_video, audio=False)

def update_video_speed(project_name, index_of_current_item, duration_of_static_time, total_duration_of_clip,timing_details):

    slice_part_of_video(project_name, index_of_current_item, 0, 0.00000000001, "static",timing_details)

    slice_part_of_video(project_name, index_of_current_item, 0, 1, "moving",timing_details)

    video_capture = cv2.VideoCapture(
        "videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    total_duration_of_static = video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT) / frame_rate

    desired_speed_change_of_static = float(
        duration_of_static_time) / float(total_duration_of_static)

    update_slice_of_video_speed(
        project_name, "static.mp4", desired_speed_change_of_static)

    video_capture = cv2.VideoCapture(
        "videos/" + str(project_name) + "/assets/videos/0_raw/moving.mp4")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    total_duration_of_moving = video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT) / frame_rate

    total_duration_of_moving = float(total_duration_of_moving)

    total_duration_of_clip = float(total_duration_of_clip)

    duration_of_static_time = float(duration_of_static_time)

    desired_speed_change_of_moving = (total_duration_of_clip - duration_of_static_time) / total_duration_of_moving

    update_slice_of_video_speed(project_name, "moving.mp4", desired_speed_change_of_moving)

    file_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 16)) + ".mp4"

    if duration_of_static_time == 0:
        
        # shutil.move("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4", "videos/" + str(video_name) + "/assets/videos/1_final/" + str(video_number) + ".mp4")
        os.rename("videos/" + str(project_name) + "/assets/videos/0_raw/moving.mp4",
                  "videos/" + str(project_name) + "/assets/videos/1_final/" + str(file_name))
        os.remove("videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4")
    else:
        final_clip = concatenate_videoclips([VideoFileClip("videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4"),                                        VideoFileClip("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4")])

        final_clip.write_videofile(
            "videos/" + str(project_name) + "/assets/videos/0_raw/full_output.mp4", fps=30)

        os.remove("videos/" + str(project_name) + "/assets/videos/0_raw/moving.mp4")
        os.remove("videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4")
        os.rename("videos/" + str(project_name) + "/assets/videos/0_raw/full_output.mp4",
                "videos/" + str(file_name))
        
    update_specific_timing_value(project_name, index_of_current_item, "timing_video", "videos/" + str(project_name) + "/assets/videos/1_final/" + str(file_name))

def calculate_desired_duration_of_each_clip(timing_details,project_name):

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
        
        update_specific_timing_value(project_name, index_of_current_item, "duration_of_clip", total_duration_of_frame)
        

def hair_swap(source_image, project_name, index_of_current_item):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]   
    model = replicate.models.get("cjwbw/style-your-hair")
    version = model.versions.get("c4c7e5a657e2e1abccd57625093522a9928edeccee77e3f55d57c664bcd96fa2")
    source_hair = upload_image("videos/" + str(project_name) + "/face.png")
    target_hair = upload_image("videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png")

    if not source_hair.startswith("http"):        
        source_hair = open(source_hair, "rb")
    if not target_hair.startswith("http"):        
        target_hair = open(target_hair, "rb")

    output = version.predict(source_image=source_hair, target_image=target_hair)

    return output

def prompt_model_depth2img(strength, image_number, timing_details, source_image):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]   
    prompt = timing_details[image_number]["prompt"]
    num_inference_steps = timing_details[image_number]["num_inference_steps"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    negative_prompt = timing_details[image_number]["negative_prompt"]
    model = replicate.models.get("jagilley/stable-diffusion-depth2img")
    version = model.versions.get("68f699d395bc7c17008283a7cef6d92edc832d8dc59eb41a6cafec7fc70b85bc")    

    if not source_image.startswith("http"):        
        source_image = open(source_image, "rb")

    output = version.predict(input_image=source_image, prompt_strength=float(strength), prompt=prompt, negative_prompt = negative_prompt, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale)
    
    return output[0]

def prompt_model_blip2(input_image, query):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]        
    model = replicate.models.get("salesforce/blip-2")
    version = model.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")
    output = version.predict(image=input_image, question=query)
    print (output)
    return output

def facial_expression_recognition(input_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]        
    model = replicate.models.get("phamquiluan/facial-expression-recognition")
    version = model.versions.get("b16694d5bfed43612f1bfad7015cf2b7883b732651c383fe174d4b7783775ff5")
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
    
def trigger_restyling_process(timing_details, project_name, index_of_current_item,model,prompt,strength,custom_pipeline,negative_prompt,guidance_scale,seed,num_inference_steps,which_stage_to_run_on,promote_new_generation, project_settings, custom_models,adapter_type):
    
                                     
    get_model_details(model)                    
    prompt = prompt.replace(",", ".")                              
    prompt = prompt.replace("\n", "")
    update_project_setting("last_prompt", prompt, project_name)
    update_project_setting("last_strength", strength,project_name)
    update_project_setting("last_model", model, project_name)
    update_project_setting("last_custom_pipeline", custom_pipeline, project_name)
    update_project_setting("last_negative_prompt", negative_prompt, project_name)
    update_project_setting("last_guidance_scale", guidance_scale, project_name)
    update_project_setting("last_seed", seed, project_name)
    update_project_setting("last_num_inference_steps",  num_inference_steps, project_name)   
    update_project_setting("last_which_stage_to_run_on", which_stage_to_run_on, project_name)
    update_project_setting("last_custom_models", custom_models, project_name)
    update_project_setting("last_adapter_type", adapter_type, project_name)
                     
    if timing_details[index_of_current_item]["source_image"] == "":
        source_image = upload_image("videos/" + str(project_name) + "/assets/frames/1_selected/" + str(index_of_current_item) + ".png")
    else:
        source_image = timing_details[index_of_current_item]["source_image"]
    batch_update_timing_values(project_name, index_of_current_item, '"'+prompt+'"', strength, model,custom_pipeline, negative_prompt,guidance_scale,seed,num_inference_steps, source_image,custom_models, adapter_type)   
    timing_details = get_timing_details(project_name)
    if which_stage_to_run_on == "Extracted Key Frames":
        source_image = timing_details[index_of_current_item]["source_image"]
    else:
        variants = timing_details[index_of_current_item]["alternative_images"]
        number_of_variants = len(variants)                       
        primary_image = int(timing_details[index_of_current_item]["primary_image"])
        source_image = variants[primary_image]
    
    dynamic_prompting(prompt, source_image, project_name, index_of_current_item)
    timing_details = get_timing_details(project_name)        

    if st.session_state['custom_pipeline'] == "Mystique":                        
        output_url = custom_pipeline_mystique(index_of_current_item, project_name, project_settings, timing_details, source_image)
    else:                            
        output_url = restyle_images(index_of_current_item, project_name, project_settings, timing_details, source_image)

    add_image_variant(output_url, index_of_current_item, project_name, timing_details)
    
    if promote_new_generation == True:                              
        timing_details = get_timing_details(project_name)                           
        variants = timing_details[index_of_current_item]["alternative_images"]
        number_of_variants = len(variants)                   
        if number_of_variants == 1:
            print("No new generation to promote")
        else:                            
            promote_image_variant(index_of_current_item, project_name, number_of_variants - 1)

def prompt_model_pix2pix(strength,video_name, image_number, timing_details, replicate_api_key, input_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    image_number = int(image_number)
    prompt = timing_details[image_number]["prompt"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    seed = int(timing_details[image_number]["seed"])
    model = replicate.models.get("arielreplicate/instruct-pix2pix")
    version = model.versions.get("10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430") 
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")
    output = version.predict(input_image=input_image, instruction_text=prompt, seed=seed, cfg_image=1.2, cfg_text = guidance_scale, resolution=704)    

    return output

def restyle_images(index_of_current_item,project_name, project_settings, timing_details, source_image):

    index_of_current_item = int(index_of_current_item)
    model_name = timing_details[index_of_current_item]["model_id"]
    strength = timing_details[index_of_current_item]["strength"]
    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)

    if model_name == "stable-diffusion-img2img-v2.1":
        output_url = prompt_model_stability(project_name, index_of_current_item,timing_details, source_image)
    elif model_name == "depth2img":    
        output_url = prompt_model_depth2img(strength, index_of_current_item, timing_details, source_image)
    elif model_name == "pix2pix":
        output_url = prompt_model_pix2pix(strength,project_name, index_of_current_item, timing_details, app_settings, source_image)
    elif model_name == "LoRA":
        output_url = prompt_model_lora(project_name, index_of_current_item, timing_details, source_image)
    elif model_name == "controlnet":
        output_url = prompt_model_controlnet(timing_details, index_of_current_item, source_image)
    elif model_name == "Dreambooth":
        output_url = prompt_model_dreambooth(project_name, index_of_current_item, timing_details[index_of_current_item]["custom_models"], app_settings,timing_details, project_settings,source_image)
    elif model_name =='StyleGAN-NADA':
        output_url = prompt_model_stylegan_nada(index_of_current_item ,timing_details,source_image,project_name)
    elif model_name == "dreambooth_controlnet":
        output_url = prompt_model_dreambooth_controlnet(source_image, timing_details, project_name, index_of_current_item)
    
    return output_url
    
def dynamic_prompting(prompt, source_image, project_name, index_of_current_item):
     
    if "[expression]" in prompt:
        prompt_expression = facial_expression_recognition(source_image)
        prompt = prompt.replace("[expression]", prompt_expression)                

    if "[location]" in prompt:
        prompt_location = prompt_model_blip2(source_image, "What's surrounding the character?")
        prompt = prompt.replace("[location]", prompt_location)        

    if "[mouth]" in prompt:
        prompt_mouth = prompt_model_blip2(source_image, "is their mouth open or closed?")
        prompt = prompt.replace("[mouth]", "mouth is " + str(prompt_mouth))
                
    if "[looking]" in prompt:
        prompt_looking = prompt_model_blip2(source_image, "the person is looking")
        prompt = prompt.replace("[looking]", "looking " + str(prompt_looking))        

    update_specific_timing_value(project_name, index_of_current_item, "prompt", prompt)

def custom_pipeline_mystique(index_of_current_item, project_name, project_settings, timing_details, source_image):
    
    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)
        
    output_url = face_swap(project_name, index_of_current_item, source_image, timing_details)
    # output_url = hair_swap(source_image, project_name, index_of_current_item)
    output_url = touch_up_images(project_name, index_of_current_item, output_url)
    output_url = resize_image(project_name, int(project_settings["width"]),int(project_settings["height"]), output_url)
    if timing_details[index_of_current_item]["model_id"] == "Dreambooth":
        model = timing_details[index_of_current_item]["custom_models"]
        output_url = prompt_model_dreambooth(project_name, index_of_current_item, model, app_settings,timing_details, project_settings,output_url)
    elif timing_details[index_of_current_item]["model_id"] == "LoRA":
        output_url = prompt_model_lora(project_name, index_of_current_item, timing_details, output_url)
    
    return output_url

def create_timings_row_at_frame_number(project_name, input_video, extract_frame_number, timing_details, index_of_new_item):
    
    df = pd.read_csv(f'videos/{project_name}/timings.csv')
    length_of_df = len(df)
    frame_time = calculate_time_at_frame_number(input_video, float(extract_frame_number),project_name)
    new_row = {'frame_time': frame_time, 'frame_number': extract_frame_number}
            
    df.loc[len(df)] = new_row
    df = df.sort_values(by=['frame_number'])
    
    df.to_csv(f'videos/{project_name}/timings.csv', index=False)
   
    return index_of_new_item
        
def get_models():
    df = pd.read_csv('models.csv')
    models = df[df.columns[0]].tolist()
    return models

def update_source_image(project_name, index_of_current_item,new_image):

    update_specific_timing_value(project_name, index_of_current_item, "source_image", new_image)
    
    df["primary_image"] = pd.to_numeric(df["primary_image"], downcast='integer', errors='coerce')
    df["seed"] = pd.to_numeric(df["seed"], downcast='integer', errors='coerce')
    df["num_inference_steps"] = pd.to_numeric(df["num_inference_steps"], downcast='integer', errors='coerce')

    df["primary_image"].fillna(0, inplace=True)
    df["seed"].fillna(0, inplace=True)
    df["num_inference_steps"].fillna(0, inplace=True)

    df["primary_image"] = df["primary_image"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["num_inference_steps"] = df["num_inference_steps"].astype(int)

    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)

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

def render_video(project_name, final_video_name):

    project_settings = get_project_settings(project_name)

    timing_details = get_timing_details(project_name)     
    total_number_of_videos = len(timing_details) - 2
    calculate_desired_duration_of_each_clip(timing_details, project_name)
    timing_details = get_timing_details(project_name)

    for i in timing_details:    
        index_of_current_item = timing_details.index(i)
        if index_of_current_item <= total_number_of_videos:
            if timing_details[index_of_current_item]["timing_video"] == "":
                total_duration_of_clip, duration_of_static_time = find_duration_of_clip(index_of_current_item, timing_details, total_number_of_videos)                    
                update_video_speed(project_name, index_of_current_item, duration_of_static_time, total_duration_of_clip,timing_details)

    video_list = []

    timing_details = get_timing_details(project_name)

    for i in timing_details:
        index_of_current_item = timing_details.index(i)
        if index_of_current_item <= total_number_of_videos:
            index_of_current_item = timing_details.index(i)
            video_location = timing_details[index_of_current_item]["timing_video"]
            video_list.append(video_location)
           
    video_clips = [VideoFileClip(v) for v in video_list]
    finalclip = concatenate_videoclips(video_clips) 
    output_video_file = f"videos/{project_name}/assets/videos/2_completed/{final_video_name}.mp4"
    if project_settings['audio'] != "":        
        audio_location = f"videos/{project_name}/assets/resources/audio/{project_settings['audio']}"
        audio_clip = AudioFileClip(audio_location)
        finalclip = finalclip.set_audio(audio_clip)
    finalclip.write_videofile(output_video_file, fps=60, audio_bitrate="1000k", bitrate="4000k", codec="libx264")

def update_specific_timing_value(project_name, index_of_current_item, parameter, value):
    
    df = pd.read_csv(f"videos/{project_name}/timings.csv")
    
    try:
        col_index = df.columns.get_loc(parameter)
    except KeyError:
        raise ValueError(f"Invalid parameter: {parameter}")
    
    df.iloc[index_of_current_item, col_index] = value
    numeric_cols = ["primary_image", "seed", "num_inference_steps"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
        df[col].fillna(0, inplace=True)
        df[col] = df[col].astype(int)
    
    df.to_csv(f"videos/{project_name}/timings.csv", index=False)

def batch_update_timing_values(project_name, index_of_current_item,prompt, strength, model, custom_pipeline,negative_prompt,guidance_scale,seed,num_inference_steps, source_image, custom_models,adapter_type):

    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")    
    if model != "Dreambooth":
        custom_models = f'"{custom_models}"' 
    df.iloc[index_of_current_item, [18, 10, 9, 4, 5, 6, 7, 8, 12, 13, 14]] = [prompt, float(strength), model, custom_pipeline, negative_prompt, float(guidance_scale), int(seed), int(num_inference_steps), source_image, custom_models, adapter_type]


    df["primary_image"] = pd.to_numeric(df["primary_image"], downcast='integer', errors='coerce')
    df["seed"] = pd.to_numeric(df["seed"], downcast='integer', errors='coerce')
    df["num_inference_steps"] = pd.to_numeric(df["num_inference_steps"], downcast='integer', errors='coerce')

    df["primary_image"].fillna(0, inplace=True)
    df["seed"].fillna(0, inplace=True)
    df["num_inference_steps"].fillna(0, inplace=True)

    df["primary_image"] = df["primary_image"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["num_inference_steps"] = df["num_inference_steps"].astype(int)

    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)

def create_gif_preview(project_name, timing_details):

    list_of_images = []
    for i in timing_details:
        # make index_of_current_item the index of the current item
        index_of_current_item = timing_details.index(i) 
        variants = timing_details[index_of_current_item]["alternative_images"]                                          
        primary_image = int(timing_details[index_of_current_item]["primary_image"])
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
        draw.text((frame.width - 60, frame.height - 60), str(index_of_current_item), font=font, fill=(255, 255, 255, 255))
        frames.append(np.array(frame))
    imageio.mimsave(f'videos/{project_name}/preview_gif.gif', frames, fps=0.5)


def create_depth_mask_image(input_image,layer,project_name, index_of_current_item):
    
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]            
    model = replicate.models.get("cjwbw/midas")
    version = model.versions.get("a6ba5798f04f80d3b314de0f0a62277f21ab3503c60c84d4817de83c5edfdae0")
    if not input_image.startswith("http"):        
        input_image = open(input_image, "rb")
    output = version.predict(image=input_image, model_type="dpt_beit_large_512")
    try:
        urllib.request.urlretrieve(output, "depth.png")    
    except Exception as e:
        print(e)

    depth_map = Image.open("depth.png")
    depth_map = depth_map.convert("L")  # Convert to grayscale image
    pixels = depth_map.load()
    mask = Image.new("L", depth_map.size)
    mask_pixels = mask.load()

    for i in range(depth_map.size[0]):
        for j in range(depth_map.size[1]):
            depth_value = pixels[i, j]
            if layer == "Foreground":
                mask_pixels[i, j] = 0 if depth_value > 200 else 255  # Set foreground pixels to black
            elif layer == "Middleground":
                mask_pixels[i, j] = 0 if depth_value <= 200 and depth_value > 50 else 255  # Set middleground pixels to black
            elif layer == "Background":
                mask_pixels[i, j] = 0 if depth_value <= 50 else 255  # Set background pixels to black

    return create_or_update_mask(project_name, index_of_current_item, mask)

def prompt_model_dreambooth_controlnet(input_image, timing_details, project_name, index_of_current_item):

    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)
    sd_api_key = app_settings['sd_api_key']
    sd_url = "https://stablediffusionapi.com/api/v5/controlnet"
    input_image = upload_image(input_image)

    payload = {
        "key": sd_api_key,
        "prompt": timing_details[index_of_current_item]["prompt"],
        "width": project_settings["width"],
        "height": project_settings["height"],
        "samples": "1",
        "num_inference_steps": timing_details[index_of_current_item]["num_inference_steps"],
        "seed": timing_details[index_of_current_item]["seed"],
        "guidance_scale": timing_details[index_of_current_item]["guidance_scale"],
        "webhook": "0",
        "track_id": "null",
        "init_image": input_image,
        "controlnet_model": timing_details[index_of_current_item]["adapter_type"],
        "model_id": "CfYgXuhAeyaqQTFDdnIUG6BjH",
        "auto_hint": "yes",
        "negative_prompt": None,
        "scheduler": "UniPCMultistepScheduler",
        "safety_checker": "no",
        "enhance_prompt": "yes",
        "strength": timing_details[index_of_current_item]["strength"],
    }
    
    
    completed = "false"

    while completed == "false":
        response = r.post(sd_url, json=payload)

        if response.json()["status"] == "processing":
            wait = int(response.json()["eta"])
            print("Processing, ETA: " + str(wait) + " seconds")
            time.sleep(wait)
            response = "https://stablediffusionapi.com/api/v3/dreambooth/fetch/" + str(response.json()["id"])
        
        elif response.json()["status"] == "success":        
            output_url = response.json()["output"][0]        
            image = r.get(output_url)            
            unique_file_name = str(uuid.uuid4()) + ".png"    
            with open(unique_file_name, "wb") as f:
                f.write(image.content)    
            url = upload_image(unique_file_name)
            os.remove(unique_file_name)
            completed = "true"

        else:
            print(response)
            return "failed"

    return url



def prompt_model_controlnet(timing_details, index_of_current_item, input_image):

    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    if timing_details[index_of_current_item]["adapter_type"] == "normal":
        model = replicate.models.get("jagilley/controlnet-normal")
    elif timing_details[index_of_current_item]["adapter_type"] == "canny":
        model = replicate.models.get("jagilley/controlnet-canny")
    elif timing_details[index_of_current_item]["adapter_type"] == "hed":
        model = replicate.models.get("jagilley/controlnet-hed")
    elif timing_details[index_of_current_item]["adapter_type"] == "scribble":
        model = replicate.models.get("jagilley/controlnet-scribble")
    elif timing_details[index_of_current_item]["adapter_type"] == "seg":
        model = replicate.models.get("jagilley/controlnet-seg")
    elif timing_details[index_of_current_item]["adapter_type"] == "hough":
        model = replicate.models.get("jagilley/controlnet-hough")
    elif timing_details[index_of_current_item]["adapter_type"] == "depth2img":
        model = replicate.models.get("jagilley/controlnet-depth2img")
    elif timing_details[index_of_current_item]["adapter_type"] == "pose":
        model = replicate.models.get("jagilley/controlnet-pose")

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
    }

    output = model.predict(**inputs)

    return output[1]

def preview_frame(project_name,video_name, frame_num):                    
    cap = cv2.VideoCapture(f'videos/{project_name}/assets/resources/input_videos/{video_name}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)                                                
    ret, frame = cap.read()                                            
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                        
    cap.release()                        
    return frame

def prompt_model_lora(project_name, index_of_current_item, timing_details, source_image):

    app_settings = get_app_settings()
    
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    # read as list and remove first and last characters
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

    if source_image.startswith("http"):
        source_image = source_image
    else:
        source_image = open(source_image, "rb")
    
    model = replicate.models.get("cloneofsimo/lora")
    version = model.versions.get("fce477182f407ffd66b94b08e761424cabd13b82b518754b83080bc75ad32466")
    inputs = {
    'prompt': timing_details[index_of_current_item]["prompt"],
    'negative_prompt': timing_details[index_of_current_item]["negative_prompt"],
    'width': int(project_settings["width"]),
    'height': int(project_settings["height"]),
    'num_outputs': 1,
    'image': source_image,
    'num_inference_steps': int(timing_details[index_of_current_item]["num_inference_steps"]),
    'guidance_scale': float(timing_details[index_of_current_item]["guidance_scale"]),
    'prompt_strength': float(timing_details[index_of_current_item]["strength"]),
    'scheduler': "DPMSolverMultistep",
    'lora_urls': lora_model_1_model_url + "|" + lora_model_2_model_url + "|" + lora_model_3_model_url,
    'lora_scales': "0.5 | 0.5 | 0.5",
    'adapter_type': timing_details[index_of_current_item]["adapter_type"],
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

def attach_audio_element(project_name, project_settings,expanded):
    with st.expander("Audio"):                        
        uploaded_file = st.file_uploader("Attach audio", type=["mp3"], help="This will attach this audio when you render a video")
        if st.button("Upload and attach new audio"):                               
            with open(os.path.join(f"videos/{project_name}/assets/resources/audio",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())
                update_project_setting("audio", uploaded_file.name, project_name)
                st.experimental_rerun()
        if project_settings["audio"] == "extracted_audio.mp3":
            st.info("You have attached the audio from the video you uploaded.")
        if project_settings["audio"] != "":
            st.audio(f"videos/{project_name}/assets/resources/audio/{project_settings['audio']}")
    
def execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, editing_image, prompt, negative_prompt, width, height, layer, index_of_current_item):

    if type_of_mask_selection == "Automated Background Selection":  
        removed_background = remove_background(project_name, editing_image)
        response = r.get(removed_background)                                
        with open("masked_image.png", "wb") as f:
            f.write(response.content)    
        if type_of_mask_replacement == "Replace With Image":                                               
            replace_background(project_name, "masked_image.png", background_image) 
            edited_image = upload_image(f"videos/{project_name}/replaced_bg.png") 
        elif type_of_mask_replacement == "Inpainting":            
            image = Image.open("masked_image.png")
            converted_image = Image.new("RGB", image.size, (255, 255, 255))
            for x in range(image.width):
                for y in range(image.height):
                    pixel = image.getpixel((x, y))
                    if pixel[3] == 0:                                    
                        converted_image.putpixel((x, y), (0,0,0))
                    else:                                    
                        converted_image.putpixel((x, y), (255, 255, 255))    
            create_or_update_mask(project_name, index_of_current_item, converted_image)            
            edited_image = inpainting(project_name, editing_image, prompt, negative_prompt,index_of_current_item)                                                                                                
                
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
            replace_background(project_name, "masked_image.png", background_image)
            edited_image = upload_image(f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":
            timing_detials = get_timing_details(project_name)
            mask_location = timing_detials[index_of_current_item]["mask"]
            if mask_location.startswith("http"):
                response = r.get(mask_location)
                im = Image.open(BytesIO(response.content))
            else:
                im = Image.open(mask_location)
            if "A" in im.getbands():
                mask = Image.new('RGB', (width, height), color = (255, 255, 255))
                mask.paste(im, (0, 0), im)                                    
                create_or_update_mask(project_name, index_of_current_item, mask)
            edited_image = inpainting(project_name, editing_image, prompt, negative_prompt,index_of_current_item)
    
    elif type_of_mask_selection == "Automated Layer Selection":
        mask_location = create_depth_mask_image(editing_image,layer, project_name, index_of_current_item)
        if type_of_mask_replacement == "Replace With Image":
            if mask_location.startswith("http"):
                mask = Image.open(BytesIO(r.get(mask_location).content)).convert('1')
            else:
                mask = Image.open(mask_location).convert('1')
            if editing_image.startswith("http"):
                response = r.get(editing_image)
                bg_img = Image.open(BytesIO(response.content)).convert('RGBA')                            
            else:
                bg_img = Image.open(editing_image).convert('RGBA')
            masked_img = Image.composite(bg_img, Image.new('RGBA', bg_img.size, (0,0,0,0)), mask)            
            masked_img.save("masked_image.png")                                                                                                                  
            replace_background(project_name, "masked_image.png", background_image)
            edited_image = upload_image(f"videos/{project_name}/replaced_bg.png")            
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(project_name, editing_image, prompt, negative_prompt,index_of_current_item)
                        
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
            masked_img = Image.composite(bg_img, Image.new('RGBA', bg_img.size, (0,0,0,0)), mask)
            masked_img.save("masked_image.png")
            replace_background(project_name, "masked_image.png", background_image)
            edited_image = upload_image(f"videos/{project_name}/replaced_bg.png")
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(project_name, editing_image, prompt, negative_prompt,index_of_current_item)

    return edited_image

            
def page_switcher(pages, page):
    
    section = [section["section_name"] for section in pages if page in section["pages"]][0]    
    index_of_section = [section["section_name"] for section in pages].index(section)
    index_of_page_in_section = pages[index_of_section]["pages"].index(page)

    return index_of_page_in_section, index_of_section
                
def resize_video(input_path, output_path, width, height, crop_type=None, output_format='mp4'):
    # Read input video
    input_video = cv2.VideoCapture(input_path)
    if not input_video.isOpened():
        raise ValueError(f"Could not open the video file: {input_path}")

    # Get source video properties
    src_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate aspect ratios
    src_aspect_ratio = src_width / src_height
    target_aspect_ratio = width / height

    if target_aspect_ratio > src_aspect_ratio:
        # Scale to target width, maintaining aspect ratio
        new_width = width
        new_height = int(src_height * (width / src_width))
    else:
        # Scale to target height, maintaining aspect ratio
        new_width = int(src_width * (height / src_height))
        new_height = height

    # Determine the crop type based on the input dimensions, if not provided
    if crop_type is None:
        width_diff = abs(src_width - width) / src_width
        height_diff = abs(src_height - height) / src_height
        crop_type = 'top_bottom' if height_diff > width_diff else 'left_right'

    # Calculate crop dimensions
    if crop_type == 'top_bottom':
        crop_top = (new_height - height) // 2
        crop_bottom = new_height - crop_top
        crop_left = 0
        crop_right = new_width
    elif crop_type == 'left_right':
        crop_top = 0
        crop_bottom = new_height
        crop_left = (new_width - width) // 2
        crop_right = new_width - crop_left
    else:
        raise ValueError("Invalid crop_type. Must be 'top_bottom' or 'left_right'.")

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # delete the video at the input_path
    os.remove(input_path)

    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        ret, frame = input_video.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Crop frame
        frame = frame[crop_top:crop_bottom, crop_left:crop_right]

        # Write frame to output video
        output_video.write(frame)

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

def main():
    
    app_settings = get_app_settings()
    
    if "online" not in st.session_state:
        current_url = st_javascript("await fetch('').then(r => window.parent.location.href)")
        time.sleep(1.5)
        # if current_url contains streamlit.app
        if "streamlit.app" in current_url:
            st.session_state["online"] = True    
        else:
            st.session_state["online"] = False
                           
        st.session_state["welcome_state"] = app_settings["welcome_state"]                       
    if st.session_state["online"] == True:
        st.error("**PLEASE READ:** This is a demo app. While you can click around, *buttons & queries won't work* and some things won't display properly. To use the proper version, follow the instructions [here](https://github.com/peter942/banodoco) to run it locally.")
    else:
        if app_settings["replicate_com_api_key"] == "":
            st.error("**To run restyling and other functions, you need to set your Replicate.com API key by going to Settings -> App Settings.**")
    
    def project_changed():
        st.session_state["project_changed"] = True

    if "project_changed" not in st.session_state:
        st.session_state["project_changed"] = False

    if st.session_state["project_changed"] == True:
        update_app_setting("previous_project", st.session_state["project_name"])

        st.session_state["project_changed"] = False        

    st.sidebar.title("Banodoco")    

    def buttons(step):
        button1, button2, button3 = st.columns([2,6,2])
        with button1:
            if step == 0:
                if st.button("Previous Step", disabled=True):
                    st.write("")
            else:
                if st.button("Previous Step"):
                    st.session_state["welcome_state"] = str(int(step) - 1)
                    update_app_setting("welcome_state", st.session_state["welcome_state"])                                    
                    st.experimental_rerun()
            
            if st.button("Skip Intro"):
                st.session_state["welcome_state"] = 7
                update_app_setting("welcome_state", st.session_state["welcome_state"]) 
                st.experimental_rerun()
        with button2:
            st.write("")
            
        with button3:
            if st.button("Next Step", type="primary"):
                st.session_state["welcome_state"] = step + 1  
                update_app_setting("welcome_state", st.session_state["welcome_state"])                  
                st.experimental_rerun()
            

                        
    if int(st.session_state["welcome_state"]) == 0 and st.session_state["online"] == False:
        
        st.header("Welcome to Banodoco!")                
        st.subheader("First, a quick demo!")            
        st.write("I've put together a quick demo video to show you how to use the app. While I recommend you watch it, you can also click the button to skip it and go straight to the app.")
        st.video("https://youtu.be/YQkwcsPGLnA")
        buttons(int(st.session_state["welcome_state"]))
            
        

    elif int(st.session_state["welcome_state"]) == 1 and st.session_state["online"] == False:
       
        st.subheader("Next, a example of a video made with it!")
        st.write("I've put together a quick video to show you how to use the app. While I recommend you watch it, you can also click the button to skip it and go straight to the app.")
        st.video("https://www.youtube.com/watch?v=vWWBiDjwKkg&t")
        
        buttons(int(st.session_state["welcome_state"]))

    elif int(st.session_state["welcome_state"]) == 2 and st.session_state["online"] == False:
        
        st.subheader("And here's a more abstract video made with it...")
        st.write("I've put together a quick video to show you how to use the app. While I recommend you watch it, you can also click the button to skip it and go straight to the app.")
        st.video("https://youtu.be/ynJyxnEzepM")
            
        buttons(int(st.session_state["welcome_state"]))


    elif int(st.session_state["welcome_state"]) == 3 and st.session_state["online"] == False:
      
        st.subheader("Add your Replicate credentials")
        st.write("Currently, we use Replicate.com for our model hosting. If you don't have an account, you can sign up for free [here](https://replicate.com/signin) and grab your API key [here](https://replicate.com/account) - this data is stored locally on your computer.")
        with st.expander("Why Replicate.com? Can I run the models locally?"):
            st.info("Replicate.com allows us to rapidly implement a wide array of models that work on any computer. Currently, these are delivered via API, which means that you pay for GPU time - but it tends to be very cheap. Getting it locally [via COG](https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md) shouldn't be too difficult but I don't have the hardware to do that")
        st.session_state["replicate_user_name"] = st.text_input("replicate_user_name", value = app_settings["replicate_user_name"])
        st.session_state["replicate_com_api_key"]  = st.text_input("replicate_com_api_key", value = app_settings["replicate_com_api_key"])
        st.warning("You can add this in App Settings later if you wish.")
        buttons(int(st.session_state["welcome_state"]))

    elif int(st.session_state["welcome_state"]) == 4 and st.session_state["online"] == False:
        
        st.subheader("That's it! Just click below when you feel sufficiently welcomed, and you'll be taken to the app!")                        
        if st.button("I feel welcomed!", type="primary"):
            st.balloons()
            if st.session_state["replicate_com_api_key"] != "":
                update_app_setting("replicate_user_name", st.session_state["replicate_user_name"])
            if st.session_state["replicate_com_api_key"] != "":
                update_app_setting("replicate_com_api_key", st.session_state["replicate_com_api_key"])
            update_app_setting("welcome_state", 7)
            st.session_state["welcome_state"] = 7
            st.experimental_rerun()        
    else:            
        if "project_set" not in st.session_state:
            st.session_state["project_set"] = "No"
            st.session_state["page_updated"] = "Yes"
                
        if st.session_state["project_set"] == "Yes":
            st.session_state["index_of_project_name"] = os.listdir("videos").index(st.session_state["project_name"])
            st.session_state["project_set"] = "No"
            st.experimental_rerun()
            
        
        if app_settings["previous_project"] != "":
            st.session_state["project_name"] = app_settings["previous_project"]
            video_list = os.listdir("videos")
            st.session_state["index_of_project_name"] = video_list.index(st.session_state["project_name"])
            st.session_state['project_set'] = 'No'
        else:
            st.session_state["project_name"] = project_name
            st.session_state["index_of_project_name"] = ""
            
        st.session_state["project_name"] = st.sidebar.selectbox("Select which project you'd like to work on:", os.listdir("videos"),index=st.session_state["index_of_project_name"], on_change=project_changed())    
        project_name = st.session_state["project_name"]
        
        if project_name == "":
            st.info("No projects found - create one in the 'New Project' section")
        else:  

            if not os.path.exists("videos/" + project_name + "/assets"):
                create_working_assets(project_name)
            
            if "index_of_section" not in st.session_state:
                st.session_state["index_of_section"] = 0
                st.session_state["index_of_page"] = 0
            
            pages = [
            {
                "section_name": "Main Process",        
                "pages": ["Key Frame Selection","Frame Styling", "Frame Editing","Frame Interpolation","Video Rendering"]
            },
            {
                "section_name": "Tools",
                "pages": ["Custom Models","Prompt Finder", "Batch Actions","Timing Adjustment"]
            },
            {
                "section_name": "Settings",
                "pages": ["Project Settings","App Settings"]
            },
            {
                "section_name": "New Project",
                "pages": ["New Project"]
            }
            ]

            
            timing_details = get_timing_details(project_name)                        

            st.session_state["section"] = st.sidebar.radio("Select a section:", [page["section_name"] for page in pages],horizontal=True)
                                                                      
            st.session_state["page"] = st.sidebar.radio("Select a page:", [page for page in pages if page["section_name"] == st.session_state["section"]][0]["pages"],horizontal=False)

            mainheader1, mainheader2 = st.columns([3,2])
            with mainheader1:
                st.header(st.session_state["page"])   

        
            if st.session_state["page"] == "Key Frame Selection":                
                with mainheader2:
                    with st.expander("💡 How key frame selection works"):
                        st.info("Key Frame Selection is a process that allows you to select the frames that you want to style. These Key Frames act as the anchor points for your animations. On the left, you can bulk select these, while on the right, you can refine your choices, or manually select them.")
                timing_details = get_timing_details(project_name)                              
                project_settings = get_project_settings(project_name)                        
                
                
                st.sidebar.subheader("Upload new videos")
                st.sidebar.write("Open the toggle below to upload and select new inputs video to use for this project.")
                          
                if project_settings["input_video"] == "":
                    st.sidebar.warning("No input video selected - please select one below.")
                if project_settings["input_video"] != "":
                    st.sidebar.success("Input video selected - you can change this below.")
                with st.sidebar.expander("Select input video", expanded=False):   
                    input_video_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/input_videos') if f.endswith(('.mp4', '.mov','.MOV', '.avi'))]       
                    if project_settings["input_video"] != "": 
                        input_video_index = input_video_list.index(project_settings["input_video"])                
                        input_video = st.selectbox("Input video:", input_video_list, index = input_video_index)
                        input_video_cv2 = cv2.VideoCapture(f'videos/{project_name}/assets/resources/input_videos/{input_video}')
                        total_frames = input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT)
                        fps = input_video_cv2.get(cv2.CAP_PROP_FPS)
                        # duration to 2 decimal places
                        duration = round(total_frames / fps, 2)
                        
                        preview1, preview2, preview3 = st.columns([1,1,1])
                        with preview1:                        
                            st.image(preview_frame(project_name, input_video, total_frames * 0.25))
                        with preview2:
                            st.image(preview_frame(project_name, input_video, total_frames * 0.5))
                        with preview3:
                            st.image(preview_frame(project_name, input_video, total_frames * 0.75))
                        st.caption(f"This video is {duration} seconds long, and has {total_frames} frames.")
                        # st.video(f'videos/{project_name}/assets/resources/input_videos/{input_video}')
                    else:
                        input_video = st.selectbox("Input video:", input_video_list)

                    if st.button("Update Video"):
                        update_project_setting("input_video", input_video, project_name)                   
                        st.experimental_rerun()
                    st.markdown("***")
                    st.subheader("Upload new video")
                    width = int(project_settings["width"])
                    height = int(project_settings["height"])
                    
                    uploaded_file = st.file_uploader("Choose a file")
                    keep_audio = st.checkbox("Keep audio from original video.")
                    resize_this_video = st.checkbox("Resize video to match project settings: " + str(width) + "px x " + str(height)+ "px", value=True)
                    
                    if st.button("Upload new video"):   
                        video_path = f'videos/{project_name}/assets/resources/input_videos/{uploaded_file.name}'                
                        with open(video_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                         
                        width = int(project_settings["width"])
                        height = int(project_settings["height"])
                        if resize_this_video == True:
                            resize_video(input_path=video_path,output_path=video_path,width=width,height=height)                    
                        st.success("Video uploaded successfully")
                        if keep_audio == True:
                            clip = VideoFileClip(f'videos/{project_name}/assets/resources/input_videos/{uploaded_file.name}')                    
                            clip.audio.write_audiofile(f'videos/{project_name}/assets/resources/audio/extracted_audio.mp3')
                            update_project_setting("audio", "extracted_audio.mp3", project_name)
                        update_project_setting("input_video", input_video, project_name)
                        project_settings = get_project_settings(project_name)                
                        time.sleep(1)
                        st.experimental_rerun()
                        
                st.sidebar.subheader("Bulk extract key frames from video")
                with st.sidebar.expander("💡 Learn about bulk extraction vs manual selection", expanded=False):
                    st.info("You can use either of the options below to extract key frames from your video in bulk, or you can use the manual key on the bottom right. If need be, you can also refine your key frame selection after bulk extraction using by clicking into the single frame view.")
                types_of_extraction = ["Regular intervals", "Extract from csv"]            
                    
                type_of_extraction = st.sidebar.radio("Choose type of key frame extraction", types_of_extraction)
                input_video_cv2 = cv2.VideoCapture(f'videos/{project_name}/assets/resources/input_videos/{input_video}')
                total_frames = input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = input_video_cv2.get(cv2.CAP_PROP_FPS)
                st.sidebar.caption(f"This video is {total_frames} frames long and has a framerate of {fps} fps.")

                if type_of_extraction == "Regular intervals":
                    frequency_of_extraction = st.sidebar.slider("How frequently would you like to extract frames?", min_value=1, max_value=120, step=1, value = 10, help=f"This will extract frames at regular intervals. For example, if you choose 15 it'll extract every 15th frame.")
                    if st.sidebar.checkbox("I understand that running this will remove all existing frames and styling."):                    
                        if st.sidebar.button("Extract frames"):
                            update_project_setting("extraction_type", "Regular intervals",project_name)
                            update_project_setting("input_video", input_video,project_name)
                            number_of_extractions = int(total_frames/frequency_of_extraction)
                            remove_existing_timing(project_name)

                            for i in range (0, number_of_extractions):
                                timing_details = get_timing_details(project_name)
                                extract_frame_number = i * frequency_of_extraction
                                last_index = len(timing_details)
                                create_timings_row_at_frame_number(project_name, input_video, extract_frame_number, timing_details,last_index)
                                timing_details = get_timing_details(project_name)
                                extract_frame(i, project_name, input_video, extract_frame_number,timing_details)                                                                                                
                            st.experimental_rerun()
                    else:                    
                        st.sidebar.button("Extract frames", disabled=True)
                        
                elif type_of_extraction == "Extract manually":
                    st.sidebar.info("On the right, you'll see a toggle to choose which frames to extract. You can also use the slider to choose the granularity of the frames you want to extract.")
                           
                elif type_of_extraction == "Extract from csv":
                    st.sidebar.subheader("Re-extract key frames using existing timings file")
                    st.sidebar.write("This will re-extract all frames based on the timings file. This is useful if you've changed the granularity of your key frames manually.")
                    if st.sidebar.checkbox("I understand that running this will remove every existing frame"):
                        if st.sidebar.button("Re-extract frames"):
                            update_project_setting("extraction_type", "Extract from csv",project_name)
                            update_project_setting("input_video", input_video,project_name)
                            get_timing_details(project_name)                                                                        
                            for i in timing_details:
                                index_of_current_item = timing_details.index(i)   
                                extract_frame_number = calculate_frame_number_at_time(input_video, timing_details[index_of_current_item]["frame_time"], project_name)
                                extract_frame(index_of_current_item, project_name, input_video, extract_frame_number,timing_details)
                                
                    else:
                        st.sidebar.button("Re-extract frames",disabled=True)    
                
                
                if len(timing_details) == 0:
                    st.info("Once you've added key frames, they'll appear here.")                
                else:

                    if "which_image_value" not in st.session_state:
                        st.session_state['which_image_value'] = 0

                    
                    timing_details = get_timing_details(project_name)
                                        
                    if 'key_frame_view_type_index' not in st.session_state:
                        st.session_state['key_frame_view_type_index'] = 0
                    

                    view_types = ["List View","Single Frame"]

                    st.session_state['key_frame_view_type'] = st.radio("View type:", view_types, key="which_view_type", horizontal=True, index=st.session_state['key_frame_view_type_index'])                        
                    
                    if view_types.index(st.session_state['key_frame_view_type']) != st.session_state['key_frame_view_type_index']:
                        st.session_state['key_frame_view_type_index'] = view_types.index(st.session_state['key_frame_view_type'])
                        st.experimental_rerun()     

                    if st.session_state['key_frame_view_type'] == "Single Frame":
                        header1,header2,header3 = st.columns([1,1,1])
                        with header1:                            
                            st.session_state['which_image'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", min_value=0, max_value=len(timing_details)-1, step=1, value=st.session_state['which_image_value'], key="which_image_checker")
                            if st.session_state['which_image_value'] != st.session_state['which_image']:
                                st.session_state['which_image_value'] = st.session_state['which_image']
                                st.experimental_rerun()
                            index_of_current_item = st.session_state['which_image']
                            
                        with header3:
                            st.write("")
                                                                                                                            
                                                        
                        slider1, slider2 = st.columns([6,12]) 
                        # make a slider for choosing the frame to extract, starting from the previous frame number, and ending at the next frame number       
                        if index_of_current_item == 0:
                            min_frames = 0
                        else:
                            min_frames = int(float(timing_details[index_of_current_item-1]['frame_number'])) + 1
                        if index_of_current_item == len(timing_details)-1:
                            max_frames = int(total_frames) - 2
                        else:
                            max_frames = int(float(timing_details[index_of_current_item+1]['frame_number'])) - 1
                        with slider1:
                            
                            st.markdown(f"Frame # for Key Frame {index_of_current_item}: {timing_details[index_of_current_item]['frame_number']}")                    
                            # show frame time to the nearest 2 decimal places
                            st.markdown(f"Frame time: {round(float(timing_details[index_of_current_item]['frame_time']),2)}")
                            
                            
                            if st.button("Delete current key frame"):
                                delete_frame(project_name, index_of_current_item)                            
                                timing_details = get_timing_details(project_name)
                                st.experimental_rerun()  
                        with slider2:
                            if timing_details[index_of_current_item]["frame_number"]-1 ==  timing_details[index_of_current_item-1]["frame_number"] and timing_details[index_of_current_item]["frame_number"] + 1 == timing_details[index_of_current_item+1]["frame_number"]:
                                st.warning("There's nowhere to move this frame due to it being 1 frame away from both the next and previous frame.")
                                new_frame_number = int(float(timing_details[index_of_current_item]['frame_number']))
                            else:
                                new_frame_number = st.slider(f"Choose which frame to preview for Key Frame #{index_of_current_item}:", min_value=min_frames, max_value=max_frames, step=1, value = int(float(timing_details[index_of_current_item]['frame_number'])))
                                                    
                                
                        preview1,preview2 = st.columns([1,2])
                        with preview1:
                            
                            if index_of_current_item == 0:
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                            else:
                                st.write("Previous frame:")
                                st.image(timing_details[index_of_current_item-1]["source_image"], use_column_width=True)
                            
                            if index_of_current_item == len(timing_details)-1: 
                                st.write("")
                            else:
                                st.write("Next frame:")
                                st.image(timing_details[index_of_current_item+1]["source_image"], use_column_width=True)
                        with preview2:
                            flag1,flag2 = st.columns([1,1])
                            with flag1:
                                st.write("Preview frame:")
                                st.write("")
                            with flag2:
                                if new_frame_number == int(float(timing_details[index_of_current_item]['frame_number'])):                            
                                    st.info(f"This is the current frame")
                                else:
                                    st.info(f"{timing_details[index_of_current_item]['frame_number']} is the current frame")
                            
                            st.image(preview_frame(project_name, input_video, new_frame_number))

                            bottom1, bottom2 = st.columns([1,1])
                            if new_frame_number != int(float(timing_details[index_of_current_item]['frame_number'])): 
                                
                                with bottom1:                            
                                        if st.button("Update this frame to here"):
                                            update_specific_timing_value(project_name, index_of_current_item, "frame_number", new_frame_number)
                                            timing_details = get_timing_details(project_name)
                                            extract_frame(index_of_current_item, project_name, input_video, new_frame_number,timing_details)
                                            st.experimental_rerun()
                                with bottom2:
                                    if st.button("Add new key frame at this time"):
                                        if new_frame_number > int(float(timing_details[index_of_current_item]['frame_number'])): 
                                            created_row = create_timings_row_at_frame_number(project_name, input_video, new_frame_number, timing_details, index_of_current_item+1)                       
                                            timing_details = get_timing_details(project_name)
                                            extract_frame(created_row, project_name, input_video, new_frame_number,timing_details) 
                                        elif new_frame_number < int(float(timing_details[index_of_current_item]['frame_number'])):
                                            created_row = create_timings_row_at_frame_number(project_name, input_video, new_frame_number, timing_details, index_of_current_item)                       
                                            timing_details = get_timing_details(project_name)
                                            extract_frame(created_row, project_name, input_video, new_frame_number,timing_details) 
                                        timing_details = get_timing_details(project_name)
                                        st.session_state['which_image_value'] = created_row
                                        st.experimental_rerun()
                            else:
                                with bottom1:
                                    st.button("Update key frame to this frame #", disabled=True, help="This is the current frame.")                            
                                with bottom2:
                                    st.button("Add new frame at this time", disabled=True, help="This is the current frame.")
                                                                                                                                
                        
                    elif st.session_state['key_frame_view_type'] == "List View":     
                        for image_name in timing_details:

                            index_of_current_item = timing_details.index(image_name)
                            
                            # if image starts with http
                            if image_name["source_image"].startswith("http"):
                                image = timing_details[index_of_current_item]["source_image"]
                            else:
                                image = Image.open(timing_details[index_of_current_item]["source_image"])            
                            st.subheader(f'Image Name: {index_of_current_item}')                
                            st.image(image, use_column_width=True)
                            
                            col1, col2,col3,col4 = st.columns([1,1,1,2])
                            
                            with col1:
                                frame_time = round(float(timing_details[index_of_current_item]['frame_time']),2)                                    
                                st.markdown(f"Frame Time: {frame_time}")
                                
                            with col2:
                                # return frame number to 2 decimal places
                                frame_number = round(float(timing_details[index_of_current_item]['frame_number']),2)                                    
                                st.markdown(f"Frame Number: {frame_number}")
                    
                            with col4:                                                      
                                if st.button(f"Jump to single frame view for #{index_of_current_item}", help="This will switch to a Single Frame view type and open this individual image."):
                                    st.session_state['which_image_value'] = index_of_current_item
                                    st.session_state['key_frame_view_type'] = "Single View"
                                    st.session_state['key_frame_view_type_index'] = 1
                                    st.session_state['open_manual_extractor'] = False
                                    st.experimental_rerun()                   
                    
                    st.markdown("***")
                
                if 'open_manual_extractor' not in st.session_state:
                    st.session_state['open_manual_extractor'] = True
                
                open_manual_extractor = st.checkbox("Open manual Key Frame extractor", value=st.session_state['open_manual_extractor'])
                                                                            
                if open_manual_extractor is True:
                    if project_settings["input_video"] == "":
                        st.info("You need to add an input video on the left before you can add key frames.")
                    else:
                        manual1,manual2 = st.columns([3,1])
                        with manual1:
                            st.subheader('Add key frames to the end of your video:')
                            st.write("Select a frame from the slider below and click 'Add Frame' it to the end of your project.")
                            # if there are >10 frames, and show_current_key_frames == "Yes", show an info 
                            if len(timing_details) > 10 and st.session_state['key_frame_view_type'] == "List View":
                                st.info("You have over 10 frames visible. To keep the frame selector running fast, we recommend hiding the currently selected key frames by selecting 'No' in the 'Show currently selected key frames' section at the top of the page.")
                        with manual2:
                            st.write("")
                            granularity = st.number_input("Choose selector granularity", min_value=1, max_value=50, step=1, value = 1, help=f"This will extract frames for you to manually choose from. For example, if you choose 15 it'll extract every 15th frame.")
                                    
                        if timing_details == []:
                            min_frames = 0
                        else:
                            length_of_timing_details = len(timing_details) - 1
                            min_frames= int(float(timing_details[length_of_timing_details]["frame_number"]))

                        max_frames = min_frames + 100

                        if max_frames > int(float(total_frames)):
                            max_frames = int(float(total_frames)) -2
                
                        slider = st.slider("Choose frame:", max_value=max_frames, min_value=min_frames,step=granularity, value = min_frames)

                        st.image(preview_frame(project_name, input_video, slider), use_column_width=True)

                        if st.button(f"Add Frame {slider} to Project"):  
                            last_index = len(timing_details)
                            created_row = create_timings_row_at_frame_number(project_name, input_video, slider, timing_details, last_index)                       
                            timing_details = get_timing_details(project_name)
                            extract_frame(created_row, project_name, input_video, slider,timing_details)                                                                                                
                            st.experimental_rerun()   
                    st.markdown("***")
                    st.subheader("Make preview video at current timings")
                    if st.button("Make Preview Video"):
                        create_video_without_interpolation(timing_details, "preview")
                        st.video(f'videos/{project_name}/preview.mp4')
                                

            elif st.session_state["page"] == "App Settings":

                app_settings = get_app_settings()
                        
                with st.expander("Replicate API Keys:"):
                    replicate_user_name = st.text_input("replicate_user_name", value = app_settings["replicate_user_name"])
                    replicate_com_api_key = st.text_input("replicate_com_api_key", value = app_settings["replicate_com_api_key"])
                    if st.button("Save Settings"):
                        update_app_setting("replicate_user_name", replicate_user_name)
                        update_app_setting("replicate_com_api_key", replicate_com_api_key)
                        # update_app_setting("aws_access_key_id", aws_access_key_id)
                        # update_app_setting("aws_secret_access_key", aws_secret_access_key)
                        st.experimental_rerun()

                with st.expander("Reset Welcome Sequence"):
                    st.write("This will reset the welcome sequence so you can see it again.")
                    if st.button("Reset Welcome Sequence"):
                        st.session_state["welcome_state"] = 0
                        update_app_setting("welcome_state", 0)
                        st.experimental_rerun()

                locally_or_hosted = st.radio("Do you want to store your files locally or on AWS?", ("Locally", "AWS"),disabled=True, help="Only local storage is available at the moment, let me know if you need AWS storage - it should be pretty easy.")
                
                if locally_or_hosted == "AWS":
                    with st.expander("AWS API Keys:"):
                        aws_access_key_id = st.text_input("aws_access_key_id", value = app_settings["aws_access_key_id"])
                        aws_secret_access_key = st.text_input("aws_secret_access_key", value = app_settings["aws_secret_access_key"])
                                                                                                      

            elif st.session_state["page"] == "New Project":
                a1, a2 = st.columns(2)
                with a1:
                    new_project_name = st.text_input("Project name:", value="")
                with a2:
                    st.write("")            
                b1, b2, b3 = st.columns(3)
                with b1:
                    width = int(st.selectbox("Select video width:", options=["512","704","768","1024"], key="video_width"))
                    
                with b2:
                    height = int(st.selectbox("Select video height:", options=["512","704","768","1024"], key="video_height"))
                with b3:
                    st.info("We recommend a small size + then scaling up afterwards.")
                
                input_type = st.radio("Select input type:", options=["Video","Image"], key="input_type", disabled=True,help="Only video is available at the moment, let me know if you need image support - it should be pretty easy.", horizontal=True)
                
                c1, c2 = st.columns(2)
                with c1:               
                    uploaded_video = st.file_uploader("Choose a video file:")
                with c2:
                    st.write("")
                    st.write("")
                    audio_options = ["No audio","Attach new audio"]
                    if uploaded_video is not None:
                        audio_options.append("Keep audio from original video")               
                        
                    st.info("Make sure that this video is the same size as you've specified above.")
                if uploaded_video is not None:
                    resize_this_video = st.checkbox("Resize video to match video dimensions above", value=True)
                
                audio = st.radio("Audio:", audio_options, key="audio",horizontal=True)
                if uploaded_video is None:
                    st.info("You can also keep the audio from your original video - just upload the video above and the option will appear.")                
                    
                if audio == "Attach new audio":
                    d1, d2 = st.columns([4,5])
                    with d1:                
                        uploaded_audio = st.file_uploader("Choose a audio file:")
                    with d2:
                        st.write("")
                        st.write("")
                        st.info("Make sure that this audio is around the same length as your video.")
                
                st.write("")
                if st.button("Create New Project"):                
                    new_project_name = new_project_name.replace(" ", "_")                      
                    create_working_assets(new_project_name)                    
                    update_project_setting("width", width, new_project_name)
                    update_project_setting("height", height, new_project_name)  
                    update_project_setting("input_type", input_type, new_project_name)
                    
                    
                    if uploaded_video is not None:
                        video_path = f'videos/{new_project_name}/assets/resources/input_videos/{uploaded_video.name}'
                        with open(video_path, 'wb') as f:
                            f.write(uploaded_video.getbuffer())
                        update_project_setting("input_video", uploaded_video.name, new_project_name)
                        if resize_this_video == True:
                            resize_video(input_path=video_path,output_path=video_path,width=width,height=height) 
                        if audio == "Keep audio from original video":
                            clip = VideoFileClip(video_path)                    
                            clip.audio.write_audiofile(f'videos/{new_project_name}/assets/resources/audio/extracted_audio.mp3')
                            update_project_setting("audio", "extracted_audio.mp3", new_project_name) 
                    if audio == "Attach new audio":
                        if uploaded_audio is not None:
                            with open(os.path.join(f"videos/{new_project_name}/assets/resources/audio",uploaded_audio.name),"wb") as f: 
                                f.write(uploaded_audio.getbuffer())
                            update_project_setting("audio", uploaded_audio.name, new_project_name)                
                                                                
                    st.session_state["project_name"] = new_project_name
                    st.session_state["project_set"] = "Yes"            
                    st.success("Project created! It should be open now. Click into 'Main Process' to get started")
                    time.sleep(1)
                    st.experimental_rerun()
                                                                        
        
            elif st.session_state["page"] == "Frame Styling":  
                timing_details = get_timing_details(project_name)
                with mainheader2:
                    with st.expander("💡 How frame styling works"):
                        st.info("On the left, there are a bunch of differnet models and processes you can use to style frames. You can even use combinatinos of models through custom pipelines or by running them one after another. We recommend experimenting on 1-2 frames before doing bulk runs for the sake of efficiency.")

                if "project_settings" not in st.session_state:
                    st.session_state['project_settings'] = get_project_settings(project_name)
                    print("HERE LAD")
                print(st.session_state['project_settings'])
                                        
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
                
                if timing_details == []:
                    st.info("You need to select and load key frames first in the Key Frame Selection section.")                            
                else:
                    top1, top2, top3 = st.columns([3,1,2])
                    with top1:
                        view_types = ["List View","Single Frame"]
                        st.session_state['frame_styling_view_type'] = st.radio("View type:", view_types, key="which_view_type", horizontal=True, index=st.session_state['frame_styling_view_type_index'])                        
                        if view_types.index(st.session_state['frame_styling_view_type']) != st.session_state['frame_styling_view_type_index']:
                            st.session_state['frame_styling_view_type_index'] = view_types.index(st.session_state['frame_styling_view_type'])
                            st.experimental_rerun()
                                                                        
                    with top2:
                        st.write("")


                    if st.session_state['frame_styling_view_type'] == "Single Frame":
                        with top3:
                            st.session_state['show_comparison'] = st.radio("Show comparison to original", options=["Don't show", "Show"], horizontal=True)                            
                                                
                        f1, f2, f3  = st.columns([1,4,1])
                        
                        with f1:
                            st.session_state['which_image'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(timing_details)-1, value=st.session_state['which_image_value'], step=1, key="which_image_selector")
                            if st.session_state['which_image_value'] != st.session_state['which_image']:
                                st.session_state['which_image_value'] = st.session_state['which_image']
                                st.experimental_rerun()
                        if timing_details[st.session_state['which_image']]["alternative_images"] != "":                                                                                       
                            variants = timing_details[st.session_state['which_image']]["alternative_images"]
                            number_of_variants = len(variants)
                            current_variant = int(timing_details[st.session_state['which_image']]["primary_image"])                                                                                                
                            which_variant = current_variant     
                            with f2:
                                which_variant = st.radio(f'Main variant = {current_variant}', range(number_of_variants), index=current_variant, horizontal = True, key = f"Main variant for {st.session_state['which_image']}")
                            with f3:
                                if which_variant == current_variant:   
                                    st.write("")                                   
                                    st.success("Main variant")
                                else:
                                    st.write("")
                                    if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['which_image']}", help="Promote this variant to the primary image"):
                                        promote_image_variant(st.session_state['which_image'], project_name, which_variant)
                                        time.sleep(0.5)
                                        st.experimental_rerun()                        
                        
                        if st.session_state['show_comparison'] == "Don't show":
                            if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                                st.image(variants[which_variant], use_column_width=True)   
                            else:
                                st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True)   
                        else:
                            if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                                img2=variants[which_variant]
                            else:
                                img2='https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'          
                            image_comparison(starting_position=50,
                                img1=timing_details[st.session_state['which_image']]["source_image"],
                                img2=img2,make_responsive=False)
                            
                        

                    if 'index_of_last_model' not in st.session_state:
                        st.session_state['index_of_last_model'] = 0

                    if len(timing_details) == 0:
                        st.info("You first need to select key frames at the Key Frame Selection stage.")

                    st.sidebar.header("Restyle Frames")   
                    if 'index_of_which_stage_to_run_on' not in st.session_state:                        
                        st.session_state['index_of_which_stage_to_run_on'] = 0
                    stages = ["Extracted Key Frames", "Current Main Variants"]
                    st.session_state['which_stage_to_run_on'] = st.sidebar.radio("What stage of images would you like to run styling on?", options=stages, horizontal=True, index =st.session_state['index_of_which_stage_to_run_on'] , help="Extracted frames means the original frames from the video.")                                                                                     
                    if stages.index(st.session_state['which_stage_to_run_on']) != st.session_state['index_of_which_stage_to_run_on']:
                        st.session_state['index_of_which_stage_to_run_on'] = stages.index(st.session_state['which_stage_to_run_on'])
                        st.experimental_rerun()

                    custom_pipelines = ["None","Mystique"]                   
                    if 'index_of_last_custom_pipeline' not in st.session_state:
                        st.session_state['index_of_last_custom_pipeline'] = 0
                    
                    st.session_state['custom_pipeline'] = st.sidebar.selectbox(f"Custom Pipeline:", custom_pipelines, index=st.session_state['index_of_last_custom_pipeline'])
                    if st.session_state['custom_pipeline'] == "Mystique" and st.session_state['index_of_last_custom_pipeline'] == 0:
                        st.session_state['index_of_last_custom_pipeline'] = 1
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
                        models = ['LoRA','Dreambooth']
                        st.session_state['model'] = st.sidebar.selectbox(f"Which type of model is trained on your character?", models, index=st.session_state['index_of_last_model'])                    
                        if st.session_state['index_of_last_model'] != models.index(st.session_state['model']):
                            st.session_state['index_of_last_model'] = models.index(st.session_state['model'])
                            st.experimental_rerun()                      
                    else:
                        models = ['stable-diffusion-img2img-v2.1', 'depth2img', 'pix2pix', 'controlnet', 'Dreambooth', 'LoRA','StyleGAN-NADA','dreambooth_controlnet']
                        st.session_state['model'] = st.sidebar.selectbox(f"Model", models, index=st.session_state['index_of_last_model'])
                        if st.session_state['index_of_last_model'] != models.index(st.session_state['model']):
                            st.session_state['index_of_last_model'] = models.index(st.session_state['model'])
                            st.experimental_rerun()
                            
                    
                    if st.session_state['model'] == "controlnet" or st.session_state['model'] == 'dreambooth_controlnet':   
                        controlnet_adapter_types = ["normal", "canny", "hed", "scribble", "seg", "hough", "depth2img", "pose"]
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
                        st.session_state['adapter_type'] = ""
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
                                    trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type'])
                            st.experimental_rerun()

                    if st.session_state['frame_styling_view_type'] == "Single Frame":
                    
                        detail1, detail2, detail3, detail4 = st.columns([2,2,1,2])

                        with detail1:
                            individual_number_of_variants = st.number_input(f"How many variants?", min_value=1, max_value=10, value=1, key=f"number_of_variants_{st.session_state['which_image']}")
                                                        
                        with detail2:
                            st.write("")
                            st.write("")
                            if st.button(f"Generate Variants", key=f"new_variations_{st.session_state['which_image']}",help="This will generate new variants based on the settings to the left."):
                                for i in range(0, individual_number_of_variants):
                                    index_of_current_item = st.session_state['which_image']
                                    trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type']) 
                                st.experimental_rerun()
                                
                        with detail4:
                            st.write("")
                            with st.expander("💡 Editing key frames"):
                                st.info("You can edit the key frames in Tools > Frame Editing.")

                    
                    elif st.session_state['frame_styling_view_type'] == "List View":
                        for i in range(0, len(timing_details)):
                            index_of_current_item = i
                        
                            st.subheader(f"Frame {i}")                
                                                
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
                                if st.button(f"Generate Variants", key=f"new_variations_{index_of_current_item}",help="This will generate new variants based on the settings to the left."):
                                    for i in range(0, individual_number_of_variants):
                                        index_of_current_item = st.session_state['which_image']
                                        trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type']) 
                                    st.experimental_rerun()
                                
                                
                            with detail3:
                                st.write("")
                            with detail4:
                                if st.button(f"Jump to single frame view for #{index_of_current_item}", help="This will switch to a Single Frame view type and open this individual image."):
                                    st.session_state['which_image_value'] = index_of_current_item
                                    st.session_state['frame_styling_view_type'] = "Single View"
                                    st.session_state['frame_styling_view_type_index'] = 1                                    
                                    st.experimental_rerun() 
                                                                            
                    st.markdown("***")        
                    st.subheader("Preview video")
                    st.write("You can get a gif of the video by clicking the button below.")
                    if st.button("Create gif of current main variants"):
                        
                        create_gif_preview(project_name, timing_details)
                        st.image(f"videos/{project_name}/preview_gif.gif", use_column_width=True)                
                        st.balloons()
                                                        

            elif st.session_state["page"] == "Frame Interpolation":
                with mainheader2:
                    with st.expander("💡 How frame interpolation works"):
                        st.info("Frame Interpolation fills the gap between 2 different frames - if the distance between the images is far, this will be a vivid switch. If it's close, for example, an eye-blinking, it can look subtle and natural.")

                timing_details = get_timing_details(project_name)

                if len(timing_details) == 0:
                    styled_frames_missing = True
                else:
                    for i in range(0, len(timing_details)):
                        if timing_details[i]["alternative_images"] == [] or timing_details[i]["alternative_images"] == "":                        
                            styled_frames_missing = True
                            break
                        else:
                            styled_frames_missing = False
                if styled_frames_missing == True:
                    st.info("You first need to select key frames and restyle them first.")
                else:
                    st.write("This is the frame interpolation view")
                    timing_details = get_timing_details(project_name)
                    key_settings = get_app_settings()
                    total_number_of_videos = len(timing_details) - 1

                    dynamic_interolation_steps = st.radio("Interpolation step selection:", options=["Static","Dynamic"], index=0, help="If static, you will be able to select a number of interpolation steps - this is good for seeing a quick render when testing. If dynamic, the number of interpolation steps will be calculated based on the length of the gap between each frame.", horizontal=True)
                    
                    if dynamic_interolation_steps == "Static":
                        interpolation_steps = st.slider("Number of interpolation steps", min_value=1, max_value=8, value=3)
                        with st.expander("Unsure what to pick? Click to see what this means."):
                            st.write("Interpolation steps are the number of frames to generate between each frame. We recommend varying the number of interpolation steps roughly based on how long the gap between each frame is is.")
                            st.write("0.17 seconds = 2 steps")
                            st.write("0.3 seconds = 3 steps")
                            st.write("0.57 seconds = 4 steps")
                            st.write("1.1 seconds = 5 steps")
                            st.write("2.17 seconds = 6 steps")
                            st.write("4.3 seconds = 7 steps")
                            st.write("8.57 seconds = 8 steps")
                    elif dynamic_interolation_steps == "Dynamic":
                        st.info("The number of interpolation steps will be calculated based on the length of the gap between each frame.")
                    
                    

                    which_video = st.radio("Which video to interpolate", options=["All","Single"], horizontal=True)
                    delete_existing_videos = st.checkbox("Delete existing videos:", help="This will delete any existing interpolated videos before generating new ones. If you don't want to delete existing videos, leave this unchecked.")

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

                    if which_video == "All":

                        if st.button("Interpolate All Videos"):

                            
                            if delete_existing_videos == True:                            
                                for i in timing_details:   
                                    index_of_current_item = timing_details.index(i)                                                             
                                    update_specific_timing_value(project_name, timing_details.index(i), "interpolated_video", "")
                                timing_details = get_timing_details(project_name)
                                time.sleep(1) 

                            for i in range(0, total_number_of_videos):

                                index_of_current_item = i

                                if dynamic_interolation_steps == "Dynamic":
                                    calculate_desired_duration_of_each_clip(timing_details,project_name)
                                    timing_details = get_timing_details(project_name)
                                    interpolation_steps = calculate_dynamic_interpolations_steps(timing_details[index_of_current_item]["duration_of_clip"])

                                if timing_details[index_of_current_item]["interpolated_video"] == "":                                                        

                                    if total_number_of_videos == index_of_current_item:

                                        current_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png"

                                        final_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(key_settings["ending_image"])

                                        prompt_interpolation_model(current_image_location, final_image_location, project_name, index_of_current_item,
                                                                interpolation_steps, key_settings["replicate_com_api_key"])

                                    else:
                                        current_image_variants = timing_details[index_of_current_item]["alternative_images"]                
                                        current_image_number = timing_details[index_of_current_item]["primary_image"]
                                        current_image_location = current_image_variants[current_image_number]
                                        next_image_variants = timing_details[index_of_current_item+1]["alternative_images"]                  
                                        next_image_number = timing_details[index_of_current_item+1]["primary_image"]
                                        next_image_location = next_image_variants[next_image_number]

                                        prompt_interpolation_model(current_image_location, next_image_location, project_name, index_of_current_item,
                                                                interpolation_steps, key_settings["replicate_com_api_key"])
                            st.success("All videos interpolated!")

                    else:
                        specific_video = st.number_input("Which video to interpolate", min_value=0, max_value=total_number_of_videos, value=0)

                        if st.button("Interpolate this video"):

                            if dynamic_interolation_steps == "Dynamic":
                                calculate_desired_duration_of_each_clip(timing_details,project_name)
                                timing_details = get_timing_details(project_name)
                                interpolation_steps = calculate_dynamic_interpolations_steps(timing_details[specific_video]["duration_of_clip"])                            
                            
                            current_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(specific_video) + ".png"

                            next_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(specific_video+1) + ".png"

                            prompt_interpolation_model(current_image_location, next_image_location, project_name, specific_video,
                                                                interpolation_steps, key_settings["replicate_com_api_key"])                                            

            elif st.session_state["page"] == "Video Rendering":
                with mainheader2:
                    with st.expander("💡 How video rendering works"):
                        st.info("This is simply pulling together the interpolated frames to deliver the final video. You can edit the timing if need be in in Tools > Timing Adjustment")

                timing_details = get_timing_details(project_name)
                project_settings = get_project_settings(project_name)
            
                disable_rendering = False
                for i in timing_details:
                    if i["interpolated_video"] == "" and timing_details.index(i) != len(timing_details)-1 and disable_rendering == False:                    
                        st.error("You need to interpolate all the videos before you can render the final video. If you delete frames or change the primary image, you will need to interpolate the video again.")
                        disable_rendering = True
                parody_movie_names = ["The_Lord_of_the_Onion_Rings", "Jurassic_Pork", "Harry_Potter_and_the_Sorcerer_s_Kidney_Stone", "Star_Wars_The_Phantom_of_the_Oprah", "The_Silence_of_the_Yams", "The_Hunger_Pains", "Free_Willy_Wonka_and_the_Chocolate_Factory", "The_Da_Vinci_Chode", "Forrest_Dump", "The_Shawshank_Inebriation", "A_Clockwork_Orange_Juice", "The_Big_Lebowski_2_Dude_Where_s_My_Car", "The_Princess_Diaries_The_Dark_Knight_Rises", "Eternal_Sunshine_of_the_Spotless_Behind", "Rebel_Without_a_Clue", "The_Terminal_Dentist", "Dr_Strangelove_or_How_I_Learned_to_Stop_Worrying_and_Love_the_Bombastic", "The_Wolf_of_Sesame_Street", "The_Good_the_Bad_and_the_Fluffy", "The_Sound_of_Mucus", "Back_to_the_Fuchsia", "The_Curious_Case_of_Benjamin_s_Button", "The_Fellowship_of_the_Bing", "The_Texas_Chainsaw_Manicure",  "The_Iron_Manatee", "Night_of_the_Living_Bread", "Indiana_Jones_and_the_Temple_of_Groom", "Kill_Billiards", "The_Bourne_Redundancy", "The_SpongeBob_SquarePants_Movie_Sponge_Out_of_Water_and_Ideas","Planet_of_the_Snapes", "No_Country_for_Old_Yentas", "The_Expendable_Accountant", "The_Terminal_Illness", "A_Streetcar_Named_Retire", "The_Secret_Life_of_Walter_s_Mitty", "The_Hunger_Games_Catching_Foam", "The_Godfather_Part_Time_Job", "How_To_Kill_a_Mockingbird", "Star_Trek_III_The_Search_for_Spock_s_Missing_Sock", "Gone_with_the_Wind_Chimes", "Dr_No_Clue", "Ferris_Bueller_s_Day_Off_Sick", "Monty_Python_and_the_Holy_Fail", "A_Fistful_of_Quarters", "Willy_Wonka_and_the_Chocolate_Heartburn", "The_Good_the_Bad_and_the_Dandruff", "The_Princess_Bride_of_Frankenstein", "The_Wizard_of_Bras", "Pulp_Friction", "Die_Hard_with_a_Clipboard", "Indiana_Jones_and_the_Last_Audit", "Finding_Nemoy", "The_Silence_of_the_Lambs_The_Musical", "Titanic_2_The_Iceberg_Strikes_Back", "Fast_Times_at_Ridgemont_Mortuary", "The_Graduate_But_Only_Because_He_Has_an_Advanced_Degree", "Beauty_and_the_Yeast","The_Blair_Witch_Takes_Manhattan","Reservoir_Bitches","Die_Hard_with_a_Pension"]
                
                random_name = random.choice(parody_movie_names)

                final_video_name = st.text_input("What would you like to name this video?",value=random_name)

                attach_audio_element(project_name, project_settings,False)

                delete_existing_videos = st.checkbox("Delete all the existing timings", value=False)

                if st.button("Render New Video",disabled=disable_rendering):
                    if delete_existing_videos == True:
                        for i in timing_details:   
                            index_of_current_item = timing_details.index(i)                                                             
                            update_specific_timing_value(project_name, timing_details.index(i), "timing_video", "")
                        timing_details = get_timing_details(project_name)
                    
                    render_video(project_name, final_video_name)
                    st.success("Video rendered!")
                    time.sleep(1.5)
                    st.experimental_rerun()            
                

                video_list = [list_of_files for list_of_files in os.listdir(
                    "videos/" + project_name + "/assets/videos/2_completed") if list_of_files.endswith('.mp4')]            

                video_dir = "videos/" + project_name + "/assets/videos/2_completed"

                video_list.sort(key=lambda f: int(re.sub('\D', '', f)))

                video_list = sorted(video_list, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)                        
                import datetime
                for video in video_list:

                    st.subheader(video)       

                    st.write(datetime.datetime.fromtimestamp(
                        os.path.getmtime("videos/" + project_name + "/assets/videos/2_completed/" + video)))

                    st.video(f"videos/{project_name}/assets/videos/2_completed/{video}")
                    
                    col1, col2 = st.columns(2)

                    with col1:

                        if st.checkbox(f"Confirm {video} Deletion"):

                            if st.button(f"Delete {video}"):
                                os.remove("videos/" + project_name +
                                        "/assets/videos/2_completed/" + video)
                                st.experimental_rerun()
                        else:
                            st.button(f"Delete {video}",disabled=True)

                            
            elif st.session_state["page"] == "Batch Actions":

                timing_details = get_timing_details(project_name)

                st.markdown("***")

                st.markdown("#### Make extracted key frames into completed key frames")
                st.write("This will move all the extracted key frames to completed key frames - good for if you don't want to make any changes to the key frames")
                if st.button("Move initial key frames to completed key frames"):
                    for i in timing_details:
                        index_of_current_item = timing_details.index(i)
                        add_image_variant(timing_details[index_of_current_item]["source_image"], index_of_current_item, project_name, timing_details)
                        promote_image_variant(index_of_current_item, project_name, 0)
                    st.success("All initial key frames moved to completed key frames")

                st.markdown("***")
                
                st.markdown("#### Remove all existing timings")
                st.write("This will remove all the timings and key frames from the project")
                if st.button("Remove Existing Timings"):
                    remove_existing_timing(project_name)

                st.markdown("***")
                
                st.markdown("#### Bulk adjust the timings")
                st.write("This will adjust the timings of all the key frames by the number of seconds you enter below")
                bulk_adjustment = st.number_input("What multiple would you like to adjust the timings by?", value=1.0)
                if st.button("Adjust Timings"):
                    for i in timing_details:
                        index_of_current_item = timing_details.index(i)
                        new_frame_time = float(timing_details[index_of_current_item]["frame_time"]) * bulk_adjustment
                        update_specific_timing_value(project_name, index_of_current_item, "frame_time", new_frame_time)
                        
                    st.success("Timings adjusted successfully!")
                    time.sleep(1)
                    st.experimental_rerun()

            elif st.session_state["page"] == "Project Settings":

                
                project_settings = get_project_settings(project_name)
                # make a list of all the files in videos/{project_name}/assets/resources/music
                
                attach_audio_element(project_name, project_settings,False)
                
                with st.expander("Version History"):
                    

                    version_name = st.text_input("What would you liket to call this version?", key="version_name")
                    version_name = version_name.replace(" ", "_")

                    if st.button("Make a copy of this project", key="copy_project"):                    
                        shutil.copyfile(f"videos/{project_name}/timings.csv", f"videos/{project_name}/timings_{version_name}.csv")
                        st.success("Project copied successfully!")             
                
                    version_list = [list_of_files for list_of_files in os.listdir(
                        "videos/" + project_name) if list_of_files.startswith('timings_')]
                    
                    header1, header2, header3 = st.columns([1,1,1])

                    with header1:
                        st.markdown("### Version Name")
                    with header2:
                        st.markdown("### Created On")
                    with header3:
                        st.markdown("### Restore Version")

                    for i in version_list:
                        col1, col2, col3 = st.columns([1,1,1])

                        with col1:
                            st.write(i)
                        with col2:
                            st.write(f"{time.ctime(os.path.getmtime(f'videos/{project_name}/{i}'))}")
                        with col3:
                            if st.button("Restore this version", key=f"restore_version_{i}"):
                                # change timings.csv to last_timings.csv
                                os.rename(f"videos/{project_name}/timings.csv", f"videos/{project_name}/timings_previous.csv")
                                # rename i to timings.csv
                                os.rename(f"videos/{project_name}/{i}", f"videos/{project_name}/timings.csv")
                                st.success("Version restored successfully! Just in case, the previous version has been saved as last_timings.csv")
                                time.sleep(2)
                                st.experimental_rerun()
                
                with st.expander("Frame Size"):
                    st.write("Current Size = ", project_settings["width"], "x", project_settings["height"])
                    width = st.selectbox("Select video width", options=["512","704","1024"], key="video_width")
                    height = st.selectbox("Select video height", options=["512","704","1024"], key="video_height")
                    if st.button("Save"):
                        update_project_setting("width", width, project_name)
                        update_project_setting("height", height, project_name)
                        st.experimental_rerun()


            elif st.session_state["page"] == "Custom Models":
                
                app_settings = get_app_settings()

                

                with st.expander("Existing models"):
                    
                    st.subheader("Existing Models:")

                    models = get_models() 
                    if models == []:
                        st.info("You don't have any models yet. Train a new model below.")
                    else:
                        header1, header2, header3, header4, header5, header6 = st.columns(6)
                        with header1:
                            st.markdown("###### Model Name")
                        with header2:
                            st.markdown("###### Trigger Word")
                        with header3:
                            st.markdown("###### Model ID")
                        with header4:
                            st.markdown("###### Example Image #1")
                        with header5:
                            st.markdown("###### Example Image #2")
                        with header6:                
                            st.markdown("###### Example Image #3")
                        
                        for i in models:   
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            with col1:
                                model_details = get_model_details(i)
                                model_details["name"]
                            with col2:
                                if model_details["keyword"] != "":
                                    model_details["keyword"]
                            with col3:
                                if model_details["keyword"] != "":
                                    model_details["id"]
                            with col4:
                                st.image(ast.literal_eval(model_details["training_images"])[0])
                            with col5:
                                st.image(ast.literal_eval(model_details["training_images"])[1])
                            with col6:
                                st.image(ast.literal_eval(model_details["training_images"])[2])
                            st.markdown("***")            

                with st.expander("Train a new model"):
                    st.subheader("Train a new model:")
                    
                    type_of_model = st.selectbox("Type of model:",["LoRA","Dreambooth"],help="If you'd like to use other methods for model training, let us know - or implement it yourself :)")
                    model_name = st.text_input("Model name:",value="", help="No spaces or special characters please")
                    if type_of_model == "Dreambooth":
                        instance_prompt = st.text_input("Trigger word:",value="", help = "This is the word that will trigger the model")        
                        class_prompt = st.text_input("Describe what your prompts depict generally:",value="", help="This will help guide the model to learn what you want it to do")
                        max_train_steps = st.number_input("Max training steps:",value=2000, help=" The number of training steps to run. Fewer steps make it run faster but typically make it worse quality, and vice versa.")
                        type_of_task = ""
                        resolution = ""
                        
                    elif type_of_model == "LoRA":
                        type_of_task = st.selectbox("Type of task:",["Face","Object","Style"]).lower()
                        resolution = st.selectbox("Resolution:",["512","768","1024"],help="The resolution for input images. All the images in the train/validation dataset will be resized to this resolution.")
                        instance_prompt = ""
                        class_prompt = ""
                        max_train_steps = ""
                    uploaded_files = st.file_uploader("Images you'd like to train the model based on:", type=['png','jpg','jpeg'], key="prompt_file",accept_multiple_files=True)
                    if uploaded_files is not None:   
                        column = 0                             
                        for image in uploaded_files:
                            # if it's an even number 
                            if uploaded_files.index(image) % 2 == 0:
                                column = column + 1                                   
                                row_1_key = str(column) + 'a'
                                row_2_key = str(column) + 'b'                        
                                row_1_key, row_2_key = st.columns([1,1])
                                with row_1_key:
                                    st.image(uploaded_files[uploaded_files.index(image)], width=300)
                            else:
                                with row_2_key:
                                    st.image(uploaded_files[uploaded_files.index(image)], width=300)
                                                                                            
                        st.write(f"You've selected {len(uploaded_files)} images.")
                        
                    if len(uploaded_files) <= 5 and model_name == "":
                        st.write("Select at least 5 images and fill in all the fields to train a new model.")
                        st.button("Train Model",disabled=True)
                    else:
                        if st.button("Train Model",disabled=False):
                            st.info("Loading...")
                            images_for_model = []
                            for image in uploaded_files:
                                with open(os.path.join(f"training_data",image.name),"wb") as f: 
                                    f.write(image.getbuffer())                                                        
                                    images_for_model.append(image.name)                                                  
                            model_status = train_model(app_settings,images_for_model, instance_prompt,class_prompt,max_train_steps,model_name, project_name, type_of_model, type_of_task, resolution)
                            st.success(model_status)

                with st.expander("Add model from internet"):
                    st.subheader("Add a model the internet:")    
                    uploaded_type_of_model = st.selectbox("Type of model:",["LoRA","Dreambooth"], key="uploaded_type_of_model", disabled=True, help="You can currently only upload LoRA models - this will change soon.")
                    uploaded_model_name = st.text_input("Model name:",value="", help="No spaces or special characters please", key="uploaded_model_name")                                   
                    uploaded_model_images = st.file_uploader("Please add at least 2 sample images from this model:", type=['png','jpg','jpeg'], key="uploaded_prompt_file",accept_multiple_files=True)
                    uploaded_link_to_model = st.text_input("Link to model:",value="", key="uploaded_link_to_model")
                    st.info("The model should be a direct link to a .safetensors files. You can find models on websites like: https://civitai.com/" )
                    if uploaded_model_name == "" or uploaded_link_to_model == "" or uploaded_model_images is None:
                        st.write("Fill in all the fields to add a model from the internet.")
                        st.button("Upload Model",disabled=True)
                    else:
                        if st.button("Upload Model",disabled=False):                    
                            images_for_model = []
                            for image in uploaded_model_images:
                                with open(os.path.join(f"training_data",image.name),"wb") as f: 
                                    f.write(image.getbuffer())                                                        
                                    images_for_model.append(image.name)
                            for i in range(len(images_for_model)):
                                images_for_model[i] = 'training_data/' + images_for_model[i]
                            df = pd.read_csv("models.csv")
                            df = df.append({}, ignore_index=True)
                            new_row_index = df.index[-1]
                            df.iloc[new_row_index, 0] = uploaded_model_name
                            df.iloc[new_row_index, 4] = str(images_for_model)
                            df.iloc[new_row_index, 5] = uploaded_type_of_model
                            df.iloc[new_row_index, 6] = uploaded_link_to_model
                            df.to_csv("models.csv", index=False)
                            st.success(f"Successfully uploaded - the model '{model_name}' is now available for use!")
                            time.sleep(1.5)
                            st.experimental_rerun()

                    
            elif st.session_state["page"] == "Frame Editing":
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
                        # st.session_state['which_image'] = st.slider('Select image to edit:', 0, len(timing_details)-1, st.session_state["which_image"])

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
                                    # save the image with a random filename
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
                                update_source_image(project_name, st.session_state['which_image'], edited_image)
                            elif st.session_state['which_stage'] == "Styled Key Frame":
                                variants = timing_details[i]["alternative_images"]
                                primary_image = timing_details[i]["primary_image"]             
                                editing_image = variants[primary_image]   
                                edited_image = execute_image_edit(type_of_mask_selection, st.session_state["type_of_mask_replacement"], project_name, background_image, editing_image, prompt, negative_prompt, width, height,st.session_state['which_layer'], st.session_state['which_image'])
                                number_of_image_variants = add_image_variant(edited_image, i, project_name, timing_details)                        
                                promote_image_variant(i, project_name, number_of_image_variants-1)

                        
            elif st.session_state["page"] == "Timing Adjustment":

                col1,col2 = st.columns(2)
                with col1:
                    automatically_rerender_clips = st.radio("Automatically rerender clips when timing changes", ["Yes","No"], help="If you want to automatically rerender clips when you change the timing, tick this box. If you want to rerender clips manually, untick this box.", index=1, horizontal=True)
                with col2:
                    st.write("")
                    
                
                video_list = [list_of_files for list_of_files in os.listdir(
                    "videos/" + project_name + "/assets/videos/2_completed") if list_of_files.endswith('.mp4')]
                video_dir = "videos/" + project_name + "/assets/videos/2_completed"
                video_list.sort(key=lambda f: int(re.sub('\D', '', f)))
                video_list = sorted(video_list, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)
                
                if len(video_list) > 0:
                    most_recent_video = video_list[0]
                    st.sidebar.markdown("### Last Video:")
                    st.sidebar.video("videos/" + project_name + "/assets/videos/2_completed/" + most_recent_video)
                parody_movie_names = ["The_Lord_of_the_Onion_Rings", "Jurassic_Pork", "Harry_Potter_and_the_Sorcerer_s_Kidney_Stone", "Star_Wars_The_Phantom_of_the_Oprah", "The_Silence_of_the_Yams", "The_Hunger_Pains", "Honey_I_Shrunk_the_Audience", "Free_Willy_Wonka_and_the_Chocolate_Factory", "The_Da_Vinci_Chode", "Forrest_Dump", "The_Shawshank_Inebriation", "A_Clockwork_Orange_Juice", "The_Big_Lebowski_2_Dude_Where_s_My_Car", "The_Princess_Diaries_The_Dark_Knight_Rises", "Eternal_Sunshine_of_the_Spotless_Behind", "Rebel_Without_a_Clue", "The_Terminal_Dentist", "Dr_Strangelove_or_How_I_Learned_to_Stop_Worrying_and_Love_the_Bombastic", "The_Wolf_of_Sesame_Street", "The_Good_the_Bad_and_the_Fluffy", "The_Sound_of_Mucus", "Back_to_the_Fuchsia", "The_Curious_Case_of_Benjamin_s_Button", "The_Fellowship_of_the_Bing", "The_Green_Mild", "My_Big_Fat_Greek_Tragedy", "Ghostbusted", "The_Texas_Chainsaw_Manicure", "The_Fast_and_the_Furniture", "The_Dark_Knight_s_Gotta_Go_Potty", "The_Iron_Manatee", "Night_of_the_Living_Bread", "Twilight_Breaking_a_Nail", "Indiana_Jones_and_the_Temple_of_Groom", "Kill_Billiards", "The_Bourne_Redundancy", "The_SpongeBob_SquarePants_Movie_Sponge_Out_of_Water_and_Ideas", "The_Social_Nutwork", "Planet_of_the_Snapes", "No_Country_for_Old_Yentas", "The_Expendable_Accountant", "The_Terminal_Illness", "A_Streetcar_Named_Retire", "The_Secret_Life_of_Walter_s_Mitty", "The_Hunger_Games_Catching_Foam", "The_Godfather_Part_Time_Job", "To_Kill_a_Rockingbird", "Star_Trek_III_The_Search_for_Spock_s_Missing_Sock", "Gone_with_the_Wind_Chimes", "Dr_No_Clue", "Ferris_Bueller_s_Day_Off_Sick", "Monty_Python_and_the_Holy_Fail", "A_Fistful_of_Quarters", "Willy_Wonka_and_the_Chocolate_Heartburn", "The_Good_the_Bad_and_the_Dandruff", "The_Princess_Bride_of_Frankenstein", "The_Wizard_of_Bras", "Pulp_Friction", "Die_Hard_with_a_Clipboard", "Indiana_Jones_and_the_Last_Audit", "Finding_Nemoy", "The_Silence_of_the_Lambs_The_Musical", "Titanic_2_The_Iceberg_Strikes_Back", "Fast_Times_at_Ridgemont_Mortuary", "The_Graduate_But_Only_Because_He_Has_an_Advanced_Degree", "Beauty_and_the_Yeast"]            
                random_name = random.choice(parody_movie_names)
                final_video_name = st.sidebar.text_input("What would you like to name this video?", value=random_name)

                if st.sidebar.button("Render New Video"):                
                    render_video(project_name, final_video_name)
                    st.success("Video rendered! Updating above...")
                    time.sleep(1.5)
                    st.experimental_rerun()
                
                
                timing_details = get_timing_details(project_name)

                for i in timing_details:
                        
                    index_of_current_item = timing_details.index(i)                                
                    variants = timing_details[index_of_current_item]["alternative_images"]
                    current_variant = int(timing_details[index_of_current_item]["primary_image"])                                                                                                                    
                    image_url = variants[current_variant]    
                                                
                    st.markdown(f"**Frame #{index_of_current_item}: {timing_details[index_of_current_item]['frame_time']:.2f} seconds**")
                    col1,col2  = st.columns([1,1])
                                        
                    with col1:
                        if timing_details[index_of_current_item]["timing_video"] != "":
                            st.video(timing_details[index_of_current_item]["timing_video"])
                        else:                            
                            st.image(image_url)
                            st.info("Re-render the video to see the new videos")
                                                
                    with col2:
                        if index_of_current_item == 0:                            
                            frame_time = st.slider(f"Starts at: {timing_details[index_of_current_item]['frame_time']:.2f} seconds", min_value=float(0), max_value=timing_details[index_of_current_item+1]['frame_time'], value=timing_details[index_of_current_item]['frame_time'], step=0.01, help="This is the time in seconds that the frame will be displayed for.")                                                                             
                            
                        elif index_of_current_item == len(timing_details)-1:
                            frame_time = st.slider(f"Starts at: {timing_details[index_of_current_item]['frame_time']:.2f} seconds", min_value=timing_details[index_of_current_item-1]['frame_time'], max_value=timing_details[index_of_current_item]['frame_time'], value=timing_details[index_of_current_item]['frame_time'], step=0.01, help="This is the time in seconds that the frame will be displayed for.")                                                 
                            
                        else:
                            frame_time = st.slider(f"Starts at: {timing_details[index_of_current_item]['frame_time']:.2f} seconds", min_value=timing_details[index_of_current_item-1]['frame_time'], max_value=timing_details[index_of_current_item+1]['frame_time'], value=timing_details[index_of_current_item]['frame_time'], step=0.01, help="This is the time in seconds that the frame will be displayed for.")                                                                         
                        if st.button(f"Save new frame time", help="This will save the new frame time.", key=f"save_frame_time_{index_of_current_item}"):
                            update_specific_timing_value(project_name, index_of_current_item, "frame_time", frame_time)
                            update_specific_timing_value(project_name, index_of_current_item-1, "timing_video", "")
                            update_specific_timing_value(project_name, index_of_current_item, "timing_video", "")                                                 
                            update_specific_timing_value(project_name, index_of_current_item+1, "timing_video", "")                                                        
                            if automatically_rerender_clips == "Yes":
                                total_duration_of_clip, duration_of_static_time = find_duration_of_clip(index_of_current_item-1, timing_details, total_number_of_videos)                    
                                update_video_speed(project_name, index_of_current_item-1, duration_of_static_time, total_duration_of_clip,timing_details)
                                total_duration_of_clip, duration_of_static_time = find_duration_of_clip(index_of_current_item, timing_details, total_number_of_videos)                    
                                update_video_speed(project_name, index_of_current_item, duration_of_static_time, total_duration_of_clip,timing_details)
                                total_duration_of_clip, duration_of_static_time = find_duration_of_clip(index_of_current_item+1, timing_details, total_number_of_videos)                    
                                update_video_speed(project_name, index_of_current_item+1, duration_of_static_time, total_duration_of_clip,timing_details)
                            st.experimental_rerun()
                        st.write("")                        
                        confirm_deletion = st.checkbox(f"Confirm  you want to delete Frame #{index_of_current_item}", help="This will delete the key frame from your project. This will not affect the video.")
                        if confirm_deletion == True:          
                            if st.button(f"Delete Frame", disabled=False, help="This will delete the key frame from your project. This will not affect the video.", key=f"delete_frame_{index_of_current_item}"):
                                delete_frame(project_name, index_of_current_item)
                                st.experimental_rerun()
                        else:
                            st.button(f"Delete Frame", disabled=True, help="This will delete the key frame from your project. This will not affect the video.", key=f"delete_frame_{index_of_current_item}")                                                                                                                                                                                               
            
            elif st.session_state["page"] == "Prompt Finder":
                
                st.write("This tool helps you find the prompt that would result in an image you provide. It's useful if you want to find a prompt for a specific image that you'd like to align your style with.")
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
                                
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.markdown("### Prompt")
                    with col2:
                        st.markdown("### Example Image")
                    with open(f"videos/{project_name}/prompts.csv", "r") as f:
                        for i in prompts:
                            index_of_current_item = prompts.index(i)                  
                            col1, col2 = st.columns([1,2])
                            with col1:
                                st.write(prompts[index_of_current_item]["prompt"])                        
                            with col2:                            
                                st.image(prompts[index_of_current_item]["example_image"], use_column_width=True)
                            st.markdown("***")
                else:
                    st.info("You haven't added any prompts yet. Add an image to get started.")
                                                
if __name__ == '__main__':
    main()
