import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import csv
import pandas as pd
import replicate
import urllib
import requests as r
import ffmpeg
import string
import json
import boto3
import time
import zipfile
import random
import uuid
from io import BytesIO
import ast
import numpy as np
from repository.local_repo.csv_repo import CSVProcessor, get_app_settings, get_project_settings, update_project_setting, update_specific_timing_value


def calculate_time_at_frame_number(input_video, frame_number, project_name):
    input_video = "videos/" + str(project_name) + "/assets/resources/input_videos/" + str(input_video)
    video = cv2.VideoCapture(input_video)
    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_percentage = float(frame_number / frame_count)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length_of_video = float(frame_count / fps)
    time_at_frame = float(frame_percentage * length_of_video)
    return time_at_frame

def preview_frame(project_name,video_name, frame_num):                    
    cap = cv2.VideoCapture(f'videos/{project_name}/assets/resources/input_videos/{video_name}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)                                                
    ret, frame = cap.read()                                            
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                        
    cap.release()                        
    return frame

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
    # img = Image.open("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    # img.save("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")
    update_specific_timing_value(project_name, frame_number, "source_image", "videos/" + project_name + "/assets/frames/1_selected/" + str(file_name))

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

# timing_details stores frame_number and frame_time map along with other meta details 
def get_timing_details(video_name):
    file_path = "videos/" + str(video_name) + "/timings.csv"
    csv_processor = CSVProcessor(file_path)
    df = csv_processor.get_df_data()

    # Evaluate the alternative_images column and replace it with the evaluated list
    df['alternative_images'] = df['alternative_images'].fillna('').apply(lambda x: ast.literal_eval(x[1:-1]) if x != '' else '')
    return df.to_dict('records')

# delete keyframe at a particular index from timings.csv
def delete_frame(project_name, index_of_current_item):
    update_specific_timing_value(project_name, index_of_current_item -1, "interpolated_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(project_name, index_of_current_item +1, "interpolated_video", "")
        
    update_specific_timing_value(project_name, index_of_current_item -1, "timing_video", "")
    if index_of_current_item < len(get_timing_details(project_name)) - 1:
        update_specific_timing_value(project_name, index_of_current_item +1, "timing_video", "")

    csv_processor = CSVProcessor("videos/" + str(project_name) + "/timings.csv")
    csv_processor.delete_row(index_of_current_item)

def batch_update_timing_values(project_name, index_of_current_item,prompt, strength, model, custom_pipeline,negative_prompt,guidance_scale,seed,num_inference_steps, source_image, custom_models,adapter_type):
    csv_processor = CSVProcessor("videos/" + str(project_name) + "/timings.csv")
    df = csv_processor.get_df_data()

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

def create_video_without_interpolation(timing_details, output_file):
    # Create a list of ffmpeg inputs, each input being a frame with its duration
    inputs = []
    for i in range(len(timing_details)):
        # Get the current frame details
        frame = timing_details[i]
        frame_time = frame['frame_time']
        source_image = frame['source_image']

        # Get the duration of this frame
        if i == len(timing_details) - 1:
            # This is the last frame, just make its duration the same as the previous
            duration = frame_time - timing_details[i-1]['frame_time']
        else:
            # This is not the last frame, get the duration until the next frame
            duration = timing_details[i+1]['frame_time'] - frame_time

        # Create an ffmpeg input for this frame
        inputs.append(
            ffmpeg.input(source_image, t=str(duration), ss=str(frame_time))
        )

    # Concatenate the inputs and export the video
    ffmpeg.concat(*inputs).output(output_file).run()


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

    df = pd.DataFrame(columns=['frame_time','frame_number','primary_image','alternative_images','custom_pipeline','negative_prompt','guidance_scale','seed','num_inference_steps','model_id','strength','notes','source_image','custom_models','adapter_type','duration_of_clip','interpolated_video','timing_video','prompt', 'mask'])

    # df.loc[0] = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    df.to_csv(f'videos/{video_name}/timings.csv', index=False)

def inpainting(video_name, input_image, prompt, negative_prompt, index_of_current_item):

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
    model_name = model_name.replace(" ", "-").lower()
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
            "template_version": "b65d36e378a01ef81d81ba49be7deb127e9bb8b74a28af3aa0eaca16b9bcd0eb",
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
        background_image = Image.open(f"{background_image}")
    foreground_image = Image.open(f"masked_image.png")
    background_image.paste(foreground_image, (0, 0), foreground_image)
    background_image.save(f"videos/{video_name}/replaced_bg.png")

    return (f"videos/{video_name}/replaced_bg.png")
        

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
        final_clip = concatenate_videoclips([VideoFileClip("videos/" + str(project_name) + "/assets/videos/0_raw/static.mp4"), VideoFileClip("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4")])

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

    source_hair = upload_image("videos/" + str(video_name) + "/face.png")

    target_hair = upload_image("videos/" + str(video_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png")

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


def custom_pipeline_mystique(index_of_current_item, project_name, project_settings, timing_details, source_image):

    prompt = timing_details[index_of_current_item]["prompt"]

    model_name = timing_details[index_of_current_item]["model_id"]

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
    csv_processor = CSVProcessor(f'videos/{project_name}/timings.csv')
    df = csv_processor.get_df_data()
    length_of_df = len(df)
    frame_time = calculate_time_at_frame_number(input_video, float(extract_frame_number),project_name)
    new_row = {'frame_time': frame_time, 'frame_number': extract_frame_number}
    
    
    # ADD IT TO THE END OF THE DATAFRAM 
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

def prompt_model_lora(project_name, index_of_current_item, timing_details, source_image):

    app_settings = get_app_settings()
    
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    # read as list and remove first and last characters
    lora_models = ast.literal_eval(timing_details[index_of_current_item]["custom_models"][1:-1])
    project_settings = get_project_settings(project_name)
    default_model_url = "https://replicate.delivery/pbxt/nWm6eP9ojwVvBCaWoWZVawOKRfgxPJmkVk13ES7PX36Y66kQA/tmpxuz6k_k2datazip.safetensors"

    lora_model_urls = []


    print(lora_models)

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
   