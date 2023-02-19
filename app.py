import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_comparison import image_comparison
from moviepy.editor import *
import cv2
import re
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
import base64
from io import BytesIO
import ast
from streamlit_drawable_canvas import st_canvas
import numpy as np

st.set_page_config(page_title="Banodoco")


def inpainting(video_name, image_url, prompt, negative_prompt):

    app_settings = get_app_settings()
    timing_details = get_timing_details(video_name)
        
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")

    version = model.versions.get("e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")
    mask = "mask.png"
    mask = upload_image("mask.png")
        
    #image = upload_image(f'videos/nirvana_test/assets/frames/1_selected/1.png')

    output = version.predict(mask=mask, image=image_url,prompt=prompt, invert_mask=True, negative_prompt=negative_prompt)    

    return output[0]

def add_image_variant(image_url, index_of_current_item, project_name, timing_details):
    

    if str(timing_details[index_of_current_item]["alternative_images"]) == "":
        alternative_images = f"['{image_url}']"
        additions = []
    else:
        alternative_images = []

        additions = timing_details[index_of_current_item]["alternative_images"]        
        additions = additions[1:-1]
        additions = ast.literal_eval(additions)        
        for addition in additions:                
            alternative_images.append(addition)
        alternative_images.append(image_url)
    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")
    df.iloc[index_of_current_item, [4]] = '"' + str(alternative_images) + '"'
    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)

    if str(timing_details[index_of_current_item]["primary_image"]) == "":
        timing_details[index_of_current_item]["primary_image"] = 0
        df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")
        df.iloc[index_of_current_item, [3]] = timing_details[index_of_current_item]["primary_image"]
        df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)
    return len(additions) + 1
                
    
def promote_image_variant(index_of_current_item, project_name, variant_to_promote):
    
    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")
    df.iloc[index_of_current_item, [3]] = variant_to_promote
    df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)
        
        
                






def train_model(app_settings, images_list, instance_prompt,class_prompt, max_train_steps, model_name,project_name):

    bucket_name = 'banodoco'
    folder_name = 'training_data/'

    # replace special characters in model_model_name with underscores

    model_name = re.sub('[^A-Za-z0-9]+', '_', model_name)

    s3 = boto3.client('s3', aws_access_key_id=app_settings['aws_access_key_id'], aws_secret_access_key=app_settings['aws_secret_access_key'])

    # for every image in the images list, set the image name to the location of the image /videos/{project_name}/resources/training_data/{image_name}

    for i in range(len(images_list)):
        images_list[i] = 'videos/' + project_name + '/assets/resources/training_data/' + images_list[i]

    with zipfile.ZipFile('images.zip', 'w') as zip:
        for image in images_list:
            zip.write(image, arcname=os.path.basename(image))
                
    aws_file_name = 'images_' + ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8)) + '.zip'

    s3.upload_file('images.zip', bucket_name, folder_name + aws_file_name)

    s3.put_object_acl(Bucket=bucket_name, Key=folder_name + aws_file_name, ACL='public-read')

    url = "https://dreambooth-api-experimental.replicate.com/v1/trainings"
    training_file_url = 'https://banodoco.s3.amazonaws.com/training_data/' + str(aws_file_name)

    os.remove('images.zip')

    os.environ["REPLICATE_API_TOKEN"] = app_settings['replicate_com_api_key']

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
        
        df.to_csv("models.csv", index=False)

        return "Success - Training Started. Please wait 10-15 minutes for the model to be trained."        
    else:
        return "Failed"






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

    app_settings = {}

    with open("app_settings.csv") as f:

        lines = [line.split(',') for line in f.read().splitlines()]
    # find the number of rows in the csv file
    number_of_rows = len(lines)

    for i in range(1, number_of_rows):

        app_settings[lines[i][0]] = lines[i][1]

    return app_settings



def get_project_settings(project_name):

    project_settings = {}

    data = pd.read_csv("videos/" + str(project_name)  + "/settings.csv", na_filter=False)

    for i, row in data.iterrows():
        project_settings[row['key']] = row['value']

    return project_settings

    '''

    project_settings = {}

    with open("videos/" + str(project_name)  + "/settings.csv") as f:

        lines = [line.split(',') for line in f.read().splitlines()]

    number_of_rows = len(lines)

    for i in range(1, number_of_rows):

        project_settings[lines[i][0]] = lines[i][1]

    return project_settings

    '''


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
                    'training_images': row[4]

                }
                return model_details


def update_app_setting(key, pair_value):
    
    csv_file_path = f'app_settings.csv'
    
    with open(csv_file_path, 'r') as csv_file:

        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            if row[0] == key:            
                row_number = csv_reader.line_num - 2            
                new_value = pair_value        
    
    df = pd.read_csv(csv_file_path)

    df.iat[row_number, 1] = new_value

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
    os.mkdir("videos/" + video_name + "/assets/resources/music")
    os.mkdir("videos/" + video_name + "/assets/resources/training_data")
    os.mkdir("videos/" + video_name + "/assets/resources/input_videos")

    os.mkdir("videos/" + video_name + "/assets/videos")

    os.mkdir("videos/" + video_name + "/assets/videos/0_raw")
    os.mkdir("videos/" + video_name + "/assets/videos/1_final")
    os.mkdir("videos/" + video_name + "/assets/videos/2_completed")

    data = {'key': ['last_prompt', 'last_model','last_strength','last_character_pipeline','song', 'input_type', 'input_video','extraction_type','width','height','last_negative_prompt','last_guidance_scale','last_seed','last_num_inference_steps','last_which_stage_to_run_on'],
        'value': ['prompt', 'sd', '0.5','no','', 'video', '','Regular intervals','','','',7.5,0,50,'Extracted Frames']}

    df = pd.DataFrame(data)

    df.to_csv(f'videos/{video_name}/settings.csv', index=False)

    df = pd.DataFrame(columns=['index_number','frame_time','frame_number','primary_image','alternative_images','character_pipeline','negative_prompt','guidance_scale','seed','num_inference_steps','model_id','strength','notes','source_image','interpolation_style','static_time', 'duration_of_clip','prompt'])

    df.loc[0] = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

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

def prompt_interpolation_model(img1, img2, video_name, video_number, interpolation_steps, replicate_api_key):

    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    model = replicate.models.get("google-research/frame-interpolation")

    output = model.predict(frame1=img1, frame2=img2, times_to_interpolate=interpolation_steps)

    video_name = "videos/" + video_name + \
        "/assets/videos/0_raw/" + str(video_number) + ".mp4"

    try:

        urllib.request.urlretrieve(output, video_name)

    except Exception as e:

        print(e)

    clip = VideoFileClip(video_name)

def get_timing_details(video_name):


    file_path = "videos/" + str(video_name) + "/timings.csv"
    df = pd.read_csv(file_path,na_filter=False)
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
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    model = replicate.models.get("pollinations/modnet")
    image = input_image
    output = model.predict(image=image)
    return output



def replace_background(video_name, foreground_image, background_image):

    background_image = Image.open(f"videos/{video_name}/assets/resources/backgrounds/{background_image}")
    foreground_image = Image.open(f"masked_image.png")
    background_image.paste(foreground_image, (0, 0), foreground_image)
    background_image.save(f"videos/{video_name}/replaced_bg.png")

    return (f"videos/{video_name}/replaced_bg.png")
    


def extract_all_frames(input_video, project_name, timing_details, time_per_frame):

    folder = 'videos/' + str(project_name) + '/assets/frames/1_selected'

    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))

    timing_details = get_timing_details(project_name)

    for i in timing_details:

        index_of_current_item = timing_details.index(i)
    
        time_of_frame = float(timing_details[index_of_current_item]["frame_time"])

        extract_frame_number = calculate_frame_number_at_time(input_video, time_of_frame, project_name)

        extract_frame(index_of_current_item, project_name, input_video, extract_frame_number,timing_details)



def extract_frame(frame_number, video_name, input_video, extract_frame_number,timing_details):

    input_video = "videos/" + str(video_name) + "/assets/resources/input_videos/" + str(input_video)

    input_video = cv2.VideoCapture(input_video)

    total_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)

    if extract_frame_number == total_frames:

        extract_frame_number = int(total_frames - 1)

    input_video.set(cv2.CAP_PROP_POS_FRAMES, extract_frame_number)

    ret, frame = input_video.read()

    df = pd.read_csv("videos/" + str(video_name) + "/timings.csv")

    if timing_details[frame_number]["frame_number"] == "":
    
        df.iloc[frame_number, [2]] = [extract_frame_number]

    df.to_csv("videos/" + str(video_name) + "/timings.csv", index=False)

    cv2.imwrite("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png", frame)

    img = Image.open("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")

    img.save("videos/" + video_name + "/assets/frames/1_selected/" + str(frame_number) + ".png")

    return str(frame_number) + ".png"


def touch_up_images(video_name, index_of_current_item, image):

    app_settings = get_app_settings()

    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("xinntao/gfpgan")    
    input_image = image
    # remove first and last characters from string
    print(input_image)
    output = model.predict(img=input_image)
    print(output)
  
    return output

def resize_image(video_name, image_number, new_width,new_height, image):

    response = r.get(image)
    image = Image.open(BytesIO(response.content))
    resized_image = image.resize((new_width, new_height))

    resized_image.save("videos/" + str(video_name) + "/temp_image.png")

    resized_image = upload_image("videos/" + str(video_name) + "/temp_image.png")

    os.remove("videos/" + str(video_name) + "/temp_image.png")

    return resized_image

def face_swap(video_name, index_of_current_item, source_image):

    app_settings = get_app_settings()
    
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

    model = replicate.models.get("arielreplicate/ghost_face_swap")

    version = model.versions.get("106df0aaf9690354379d8cd291ad337f6b3ea02fe07d90feb1dafd64820066fa")

    source_face = upload_image("videos/" + str(video_name) + "/face.png")

    target_face = source_image

    output = version.predict(source_path=source_face, target_path=target_face,use_sr=0)

    new_image = "videos/" + str(video_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png"
    
    try:

        urllib.request.urlretrieve(output, new_image)

    except Exception as e:

        print(e)
    return output

def prompt_model_stability(project_name, index_of_current_item, timing_details, source_image):

    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    index_of_current_item = int(index_of_current_item)
    prompt = timing_details[index_of_current_item]["prompt"]
    strength = timing_details[index_of_current_item]["strength"]
    model = replicate.models.get("cjwbw/stable-diffusion-img2img-v2.1")
    version = model.versions.get("650c347f19a96c8a0379db998c4cd092e0734534591b16a60df9942d11dec15b")
    input_image = source_image    
    output = version.predict(image=input_image, prompt_strength=str(strength), prompt=prompt, negative_prompt = timing_details[index_of_current_item]["negative_prompt"], width = project_settings["width"], height = project_settings["height"], guidance_scale = float(timing_details[index_of_current_item]["guidance_scale"]), seed = int(timing_details[index_of_current_item]["seed"]), num_inference_steps = int(timing_details[index_of_current_item]["num_inference_steps"]))
    new_image = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png" 

    return output[0]


def delete_frame(video_name, image_number):

    os.remove("videos/" + str(video_name) + "/assets/frames/1_selected/" + str(image_number) + ".png")

    df = pd.read_csv("videos/" + str(video_name) + "/timings.csv")
    
    for i in range(int(image_number)+1, len(os.listdir("videos/" + str(video_name) + "/assets/frames/1_selected"))):
            
        os.rename("videos/" + str(video_name) + "/assets/frames/1_selected/" + str(i) + ".png", "videos/" + str(video_name) + "/assets/frames/1_selected/" + str(i - 1) + ".png")

        df.iloc[i, [0]] = str(i - 1)

    # remove the row from the timings.csv file using pandas

    df = df.drop([int(image_number)])

    df.to_csv("videos/" + str(video_name) + "/timings.csv", index=False)

    




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
    print(model)
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
    output = version.predict(image=input_image, prompt=prompt, prompt_strength=strength, height = project_settings["height"], width = project_settings["width"], disable_safety_check=True, negative_prompt = negative_prompt, guidance_scale = guidance_scale, seed = seed, num_inference_steps = num_inference_steps)

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

def slice_part_of_video(video_name, video_number, video_start_percentage, video_end_percentage, slice_name):

    input_video = "videos/" + \
        str(video_name) + "/assets/videos/0_raw/" + str(video_number) + ".mp4"
    video_capture = cv2.VideoCapture(input_video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_duration_of_clip = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate
    start_time = float(video_start_percentage) * float(total_duration_of_clip)
    end_time = float(video_end_percentage) * float(total_duration_of_clip)
    clip = VideoFileClip(input_video).subclip(
        t_start=start_time, t_end=end_time)
    output_video = "videos/" + \
        str(video_name) + "/assets/videos/0_raw/" + str(slice_name) + ".mp4"
    clip.write_videofile(output_video, audio=False)

def update_video_speed(video_name, video_number, duration_of_static_time, total_duration_of_clip):


    input_video = "videos/" + \
        str(video_name) + "/assets/videos/0_raw/" + str(video_number) + ".mp4"

    slice_part_of_video(video_name, video_number, 0, 0.00000000001, "static")

    slice_part_of_video(video_name, video_number, 0, 1, "moving")

    video_capture = cv2.VideoCapture(
        "videos/" + str(video_name) + "/assets/videos/0_raw/static.mp4")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    total_duration_of_static = video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT) / frame_rate

    desired_speed_change_of_static = float(
        duration_of_static_time) / float(total_duration_of_static)

    update_slice_of_video_speed(
        video_name, "static.mp4", desired_speed_change_of_static)

    video_capture = cv2.VideoCapture(
        "videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4")

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    total_duration_of_moving = video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT) / frame_rate

    total_duration_of_moving = float(total_duration_of_moving)

    total_duration_of_clip = float(total_duration_of_clip)

    duration_of_static_time = float(duration_of_static_time)

    desired_speed_change_of_moving = (total_duration_of_clip - duration_of_static_time) / total_duration_of_moving

    update_slice_of_video_speed(video_name, "moving.mp4", desired_speed_change_of_moving)

    if duration_of_static_time == 0:
        
        # shutil.move("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4", "videos/" + str(video_name) + "/assets/videos/1_final/" + str(video_number) + ".mp4")
        os.rename("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4",
                  "videos/" + str(video_name) + "/assets/videos/1_final/" + str(video_number) + ".mp4")
        os.remove("videos/" + str(video_name) + "/assets/videos/0_raw/static.mp4")
    else:
        final_clip = concatenate_videoclips([VideoFileClip("videos/" + str(video_name) + "/assets/videos/0_raw/static.mp4"),                                        VideoFileClip("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4")])

        final_clip.write_videofile(
            "videos/" + str(video_name) + "/assets/videos/0_raw/full_output.mp4", fps=30)

        os.remove("videos/" + str(video_name) + "/assets/videos/0_raw/moving.mp4")
        os.remove("videos/" + str(video_name) + "/assets/videos/0_raw/static.mp4")
        os.rename("videos/" + str(video_name) + "/assets/videos/0_raw/full_output.mp4",
                "videos/" + str(video_name) + "/assets/videos/1_final/" + str(video_number) + ".mp4")

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

        df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")

        df.iloc[index_of_current_item, [15,16]] = [duration_of_static_time,total_duration_of_frame]
    
        df.to_csv("videos/" + str(project_name) + "/timings.csv", index=False)


def hair_swap(replicate_api_key, video_name, index_of_current_item,stablediffusionapi_com_api_key):

    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    model = replicate.models.get("cjwbw/style-your-hair")

    version = model.versions.get("c4c7e5a657e2e1abccd57625093522a9928edeccee77e3f55d57c664bcd96fa2")

    source_hair = upload_image("videos/" + str(video_name) + "/face.png")

    target_hair = upload_image("videos/" + str(video_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png")

    output = version.predict(source_image=source_hair, target_image=target_hair)

    new_image = "videos/" + str(video_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png"

    try:

        urllib.request.urlretrieve(output, new_image)

    except Exception as e:

        print(e)

def prompt_model_depth2img(strength,video_name, image_number, replicate_api_key, timing_details, source_image):

    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    prompt = timing_details[image_number]["prompt"]
    num_inference_steps = timing_details[image_number]["num_inference_steps"]
    guidance_scale = timing_details[image_number]["guidance_scale"]
    negative_prompt = timing_details[image_number]["negative_prompt"]
    model = replicate.models.get("jagilley/stable-diffusion-depth2img")
    version = model.versions.get("68f699d395bc7c17008283a7cef6d92edc832d8dc59eb41a6cafec7fc70b85bc")
    image = source_image

    output = version.predict(input_image=image, prompt_strength=str(strength), prompt=prompt, negative_prompt = negative_prompt, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale)
    
    return output[0]

def prompt_model_blip2(source_image, query):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]        
    model = replicate.models.get("salesforce/blip-2")
    version = model.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
    output = version.predict(image=source_image, question=query)
    print (output)
    return output

def facial_expression_recognition(source_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]        
    model = replicate.models.get("phamquiluan/facial-expression-recognition")
    version = model.versions.get("b16694d5bfed43612f1bfad7015cf2b7883b732651c383fe174d4b7783775ff5")
    output = version.predict(input_path=source_image)
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
    

def prompt_model_pix2pix(strength,video_name, image_number, timing_details, replicate_api_key, source_image):
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]
    image_number = int(image_number)
    prompt = timing_details[image_number]["prompt"]
    guidance_scale = float(timing_details[image_number]["guidance_scale"])
    seed = int(timing_details[image_number]["seed"])
    model = replicate.models.get("arielreplicate/instruct-pix2pix")
    version = model.versions.get("10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430")
    image = source_image
    output = version.predict(input_image=image, instruction_text=prompt, seed=seed, cfg_image=1.2, cfg_text = guidance_scale, resolution=704)    

    return output

def restyle_images(index_of_current_item,project_name, project_settings, timing_details, source_image):

    print("THIS IS THE INDEX OF THE CURRENT ITEM")
    print(index_of_current_item)

    index_of_current_item = int(index_of_current_item)
    
    prompt = timing_details[index_of_current_item]["prompt"]
    model_name = timing_details[index_of_current_item]["model_id"]
    strength = timing_details[index_of_current_item]["strength"]
    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)

    if model_name == "sd":
        output_url = prompt_model_stability(project_name, index_of_current_item,timing_details, source_image)
    elif model_name == "depth2img":    
        output_url = prompt_model_depth2img(strength,project_name, index_of_current_item,app_settings, timing_details, source_image)
    elif model_name == "pix2pix":
        output_url = prompt_model_pix2pix(strength,project_name, index_of_current_item, timing_details, app_settings, source_image)
    else:
        output_url = prompt_model_dreambooth(project_name, index_of_current_item, model_name, app_settings,timing_details, project_settings,source_image)

    add_image_variant(output_url, index_of_current_item, project_name, timing_details)




def character_pipeline(index_of_current_item, project_name, project_settings, timing_details, source_image):

    prompt = timing_details[index_of_current_item]["prompt"]
    model_name = timing_details[index_of_current_item]["model_id"]

    app_settings = get_app_settings()
    project_settings = get_project_settings(project_name)
    
    if "[expression]" in prompt:
        prompt_expression = facial_expression_recognition(source_image)
        prompt = prompt.replace("[expression]", prompt_expression)
        update_specific_timing_value(project_name, index_of_current_item, "prompt", prompt)
        timing_details = get_timing_details(project_name)

    if "[location]" in prompt:
        prompt_location = prompt_model_blip2(source_image, "What's surrounding the character?")
        prompt = prompt.replace("[location]", prompt_location)
        update_specific_timing_value(project_name, index_of_current_item, "prompt", prompt)
        timing_details = get_timing_details(project_name)
    

    # remove_background(project_name, index_of_current_item, "2_character_pipeline_completed", "yes")
    # replace_background(project_name, index_of_current_item, str(index_of_current_item) + ".png", "2_character_pipeline_completed")
    output_url = face_swap(project_name, index_of_current_item, source_image)
    # hair_swap(key_settings["replicate_com_api_key"],video_name,index_of_current_item,key_settings["stablediffusionapi_com_api_key"])
    output_url = touch_up_images(project_name, index_of_current_item, output_url)
    output_url = resize_image(project_name, index_of_current_item, int(project_settings["width"]),int(project_settings["height"]), output_url)
    output_url = prompt_model_dreambooth(project_name, index_of_current_item, model_name, app_settings,timing_details, project_settings,output_url)
    # remove_background(project_name, index_of_current_item, "2_character_pipeline_completed", "no")
    # replace_background(project_name, index_of_current_item, str(index_of_current_item) + ".png", "2_character_pipeline_completed")
    # prompt_model_stability("2_character_pipeline_completed",project_name, index_of_current_item,timing_details, "0.2")
    add_image_variant(output_url, index_of_current_item, project_name, timing_details)


def create_timings_row_at_frame_number(project_name, input_video, extract_frame_number, timing_details):

    frame_time = calculate_time_at_frame_number(input_video, float(extract_frame_number),project_name)

    df = pd.read_csv(f'videos/{project_name}/timings.csv')                

    last_index = len(timing_details)

    new_row = {'index_number': last_index, 'frame_time': frame_time, 'frame_number': extract_frame_number}

    df.loc[last_index] = new_row

    df.to_csv(f'videos/{project_name}/timings.csv', index=False)




def get_models():
    
    df = pd.read_csv('models.csv')

    models = df[df.columns[0]].tolist()

    return models




def update_source_image(project_name, index_of_current_item,source_image):

    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")

    df.iloc[index_of_current_item, [13]] = [source_image]

    # Convert the "primary_image" column to numeric
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



def update_timing_values(project_name, index_of_current_item,prompt, strength, model, character_pipeline,negative_prompt,guidance_scale,seed,num_inference_steps, source_image):

    df = pd.read_csv("videos/" + str(project_name) + "/timings.csv")

    print("HERE IT IS!!!")

    print(prompt,strength,model,character_pipeline,negative_prompt,guidance_scale,seed,num_inference_steps, source_image)
    df.iloc[index_of_current_item, [17,11,10,5,6,7,8,9,13]] = [prompt,strength,model,character_pipeline,negative_prompt,guidance_scale,seed,num_inference_steps, source_image]

    # Convert the "primary_image" column to numeric
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
        variants = variants[1:-1]
        variants = ast.literal_eval(variants)                            
        primary_image = int(timing_details[index_of_current_item]["primary_image"])
        source_image = variants[primary_image]
        list_of_images.append(source_image)
    print(list_of_images)
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


def create_depth_mask_image(image_url,layer):
    
    app_settings = get_app_settings()
    os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]            
    model = replicate.models.get("cjwbw/midas")
    version = model.versions.get("a6ba5798f04f80d3b314de0f0a62277f21ab3503c60c84d4817de83c5edfdae0")
    output = version.predict(image=image_url, model_type="dpt_beit_large_512")
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

    mask.save("mask.png") 

def execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, bg_image, prompt, negative_prompt, width, height, layer):
    if type_of_mask_selection == "Automated Background Selection":  
        removed_background = remove_background(project_name, bg_image)
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
            converted_image.save("mask.png")
            edited_image = inpainting(project_name, bg_image, prompt, negative_prompt)                                                                                                
                
    elif type_of_mask_selection == "Manual Background Selection":
        if type_of_mask_replacement == "Replace With Image":    
            response = r.get(bg_image)
            bg_img = Image.open(BytesIO(response.content))                        
            mask_img = Image.open("mask.png")                                
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
            im = Image.open("mask.png")
            mask = Image.new('RGB', (width, height), color = (255, 255, 255))
            mask.paste(im, (0, 0), im)                                    
            mask.save("mask.png")
            edited_image = inpainting(project_name, bg_image, prompt, negative_prompt)
    elif type_of_mask_selection == "Automated Layer Selection":
        create_depth_mask_image(bg_image,layer)
        if type_of_mask_replacement == "Replace With Image":
            mask = Image.open("mask.png").convert('1')

            response = r.get(bg_image)
            bg_img = Image.open(BytesIO(response.content)).convert('RGBA')    
            

            # Apply the mask on the background image
            masked_img = Image.composite(bg_img, Image.new('RGBA', bg_img.size, (0,0,0,0)), mask)

            # Save the result as masked_image.png
            masked_img.save("masked_image.png")
                                                                                                                  
            replace_background(project_name, "masked_image.png", background_image)
            edited_image = upload_image(f"videos/{project_name}/replaced_bg.png")
            
        elif type_of_mask_replacement == "Inpainting":
            edited_image = inpainting(project_name, bg_image, prompt, negative_prompt)
                        

    return edited_image


def main():
    
    def project_changed():
        st.session_state["project_changed"] = True

    if "project_changed" not in st.session_state:
        st.session_state["project_changed"] = False

    if st.session_state["project_changed"] == True:
        update_app_setting("previous_project", st.session_state["project_name"])
        st.session_state["project_changed"] = False

    app_settings = get_app_settings()
    st.sidebar.title("Banodoco")
    if app_settings["previous_project"] != "":
        st.session_state["project_name"] = project_name = app_settings["previous_project"]
        video_list = os.listdir("videos")
        st.session_state["index_of_project_name"] = video_list.index(project_name)
        st.session_state['project_set'] = 'Done'
    else:
        st.session_state["project_name"] = project_name
        st.session_state["index_of_project_name"] = ""            
    st.session_state["project_name"] = st.sidebar.selectbox("Select which project you'd like to work on:", os.listdir("videos"),index=st.session_state["index_of_project_name"], on_change=project_changed())    
    project_name = st.session_state["project_name"]


    
    if project_name == "":
        st.write("No projects found")
    else:        
        if not os.path.exists("videos/" + project_name + "/assets"):
            create_working_assets(project_name)
        timing_details = get_timing_details(project_name)
        st.session_state.stage = st.sidebar.radio("Select an option",
                                    ["App Settings","New Project","Project Settings","Custom Models","Key Frame Selection",
                                     "Frame Styling","Frame Editing","Frame Interpolation","Timing Adjustment","Video Rendering"])
        st.header(st.session_state.stage)   

     
        if st.session_state.stage == "Key Frame Selection":
            


            timing_details = get_timing_details(project_name)
            # if videos/{project_name}/assets/frames/1_selected/0.pngdoesn't exist, give a warning
                  
            project_settings = get_project_settings(project_name)
            images_list = [f for f in os.listdir(f'videos/{project_name}/assets/frames/0_extracted') if f.endswith('.png')]    
            images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
            st.sidebar.subheader("Extract key frames from video")                
            input_video_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/input_videos') if f.endswith('.mp4')]
            
            if project_settings["input_video"] != "": 
                input_video_index = input_video_list.index(project_settings["input_video"])                
                input_video = st.sidebar.selectbox("Input video:", input_video_list, index = input_video_index)
                st.sidebar.video(f'videos/{project_name}/assets/resources/input_videos/{input_video}')
            else:
                input_video = st.sidebar.selectbox("Input video:", input_video_list)

            with st.sidebar.expander("Upload new video", expanded=False):
                st.write("Upload a new video to the project")
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    with open(f'videos/{project_name}/assets/resources/input_videos/{uploaded_file.name}', 'wb') as f:
                        f.write(uploaded_file.getbuffer())
            type_of_extraction = project_settings["extraction_type"]
            types_of_extraction = ["Extract manually", "Regular intervals", "Extract from csv"]            
            type_of_extraction_index = types_of_extraction.index(type_of_extraction)            
            type_of_extraction = st.sidebar.radio("Choose type of key frame extraction", types_of_extraction, index = type_of_extraction_index)
            input_video_cv2 = cv2.VideoCapture(f'videos/{project_name}/assets/resources/input_videos/{input_video}')
            total_frames = input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = input_video_cv2.get(cv2.CAP_PROP_FPS)
            st.sidebar.caption(f"This video is {total_frames} frames long and has a framerate of {fps} fps.")

            if type_of_extraction == "Regular intervals":
                frequency_of_extraction = st.sidebar.slider("How frequently would you like to extract frames?", min_value=5, max_value=100, step=5, value = 10, help=f"This will extract frames at regular intervals. For example, if you choose 15 it'll extract every 15th frame.")
                if st.sidebar.checkbox("I understand that running this will remove all existing frames"):                    
                    if st.sidebar.button("Extract frames"):
                        update_project_setting("extraction_type", "Regular intervals",project_name)
                        update_project_setting("input_video", input_video,project_name)
                        number_of_extractions = int(total_frames/frequency_of_extraction)
                        remove_existing_timing(project_name)

                        for i in range (0, number_of_extractions):
                            
                            extract_frame_number = i * frequency_of_extraction
                            create_timings_row_at_frame_number(project_name, input_video, extract_frame_number, timing_details)
                            timing_details = get_timing_details(project_name)
                            extract_frame(i, project_name, input_video, extract_frame_number,timing_details)                                                                                                
                        st.experimental_rerun()
                else:                    
                    st.sidebar.button("Extract frames", disabled=True)

                    

            elif type_of_extraction == "Extract manually":
                granularity = st.sidebar.slider("Choose frame granularity", min_value=5, max_value=50, step=5, value = 10, help=f"This will extract frames for you to manually choose from. For example, if you choose 15 it'll extract every 15th frame.")
                if st.sidebar.checkbox("I understand that running this will remove all existing frames"):
                    if st.sidebar.button("Update granularity"):
                        update_project_setting("extraction_type", "Extract manually",project_name)
                        update_project_setting("input_video", input_video,project_name)
                        remove_existing_timing(project_name)
                        for f in os.listdir(f'videos/{project_name}/assets/frames/0_extracted'):
                            os.remove(f'videos/{project_name}/assets/frames/0_extracted/{f}')                        
                        for i in range(0, int(input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT)), int(granularity)):
                            input_video_cv2.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ret, frame = input_video_cv2.read()
                            cv2.imwrite(f"videos/{project_name}/assets/frames/0_extracted/" + str(i) + ".png", frame)
                            st.session_state['select_frames'] = []
                        cv2.imwrite(f"videos/{project_name}/assets/frames/0_extracted/" + str(int(float(total_frames))) + ".png", int(float(total_frames)))
                        st.experimental_rerun()
                else:
                    st.sidebar.button("Update granularity", disabled=True)


            elif type_of_extraction == "Extract from csv":
                st.sidebar.subheader("Re-extract key frames using existing timings file")
                st.sidebar.write("This will re-extract all frames based on the timings file. This is useful if you've changed the granularity of your key frames manually.")
                if st.sidebar.checkbox("I understand that running this will remove every existing frame"):
                    if st.sidebar.button("Re-extract frames"):
                        update_project_setting("extraction_type", "Extract from csv",project_name)
                        update_project_setting("input_video", input_video,project_name)
                        get_timing_details(project_name)                                            
                        for f in os.listdir(f'videos/{project_name}/assets/frames/0_extracted'):
                            os.remove(f'videos/{project_name}/assets/frames/0_extracted/{f}')
                        for i in timing_details:
                            index_of_current_item = timing_details.index(i)                                                
                            extract_frame(index_of_current_item, project_name, input_video, int(float(timing_details[index_of_current_item]['frame_number'])),timing_details)                                        
                else:
                    st.sidebar.button("Re-extract frames",disabled=True)                  
                                    
            timing_details = get_timing_details(project_name)
                    
            if not os.path.exists(f"videos/{project_name}/assets/frames/1_selected/0.png") and not os.path.exists(f"videos/{project_name}/assets/frames/1_selected/0.png"):                    
                st.info("You need to add  key frames first in the Key Frame Selection section.")
            else:         
                for image_name in timing_details:

                    index_of_current_item = timing_details.index(image_name)
                
                    image = Image.open(f'videos/{project_name}/assets/frames/1_selected/{index_of_current_item}.png')            
                    st.subheader(f'Image Name: {index_of_current_item}')                
                    st.image(image, use_column_width=True)
                    
                    col1, col2,col3, col4 = st.columns([1,1,1,1])

                    current_item =  str(index_of_current_item) + "_lad"

                    
                    with col1:
                        frame_time = round(float(timing_details[index_of_current_item]['frame_time']),2)                                    
                        st.markdown(f"Frame Time: {frame_time}")
                        
                    with col2:
                        # return frame number to 2 decimal places
                        frame_number = round(float(timing_details[index_of_current_item]['frame_number']),2)                                    
                        st.markdown(f"Frame Number: {frame_number}")

                

                    with col3:
                        if st.button(f"Reset Key Frame #{index_of_current_item}", help="This will reset the base key frame to the original unedited version. This will not affect the video."):
                            extract_frame(int(index_of_current_item), project_name, project_settings["input_video"], timing_details[index_of_current_item]["frame_number"],timing_details)
                            new_source_image = upload_image(f"videos/{project_name}/assets/frames/1_selected/{index_of_current_item}.png")
                            update_source_image(project_name, index_of_current_item, new_source_image)
                            st.experimental_rerun()
                            
                    with col4:                    
                        if st.button(f"Delete {index_of_current_item} Frame", disabled=False, help="This will delete the key frame from your project. This will not affect the video."):
                            delete_frame(project_name, index_of_current_item)
                            st.experimental_rerun()                    

                if type_of_extraction == "Extract manually":

                    st.subheader('Manually add key frames to your project')
                    st.write("Select a frame from the slider below and click 'Add Frame' it to the end of your project")
                    images = os.listdir(f"videos/{project_name}/assets/frames/0_extracted")
                    images = [i.replace(".png", "") for i in images]

                    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                                
                    if timing_details == []:
                        min_frames = 0
                    else:
                        length_of_timing_details = len(timing_details) - 1
                        min_frames= int(float(timing_details[length_of_timing_details]["frame_number"]))

                    max_frames = int((float(total_frames) / float(granularity))) * int(granularity)

                    if max_frames > int(float(total_frames)):
                        max_frames = int(float(total_frames))
            
                    slider = st.slider("Choose Frame", max_value= min_frames+ 100, min_value=min_frames,step=granularity, value = min_frames + granularity)



                    st.image(f"videos/{project_name}/assets/frames/0_extracted/{slider}.png")

                    if st.button(f"Add Frame {slider} to Project"):     
                        
                        last_index = len(timing_details)

                        shutil.copy(f"videos/{project_name}/assets/frames/0_extracted/{slider}.png", f"videos/{project_name}/assets/frames/1_selected/{last_index}.png")

                        create_timings_row_at_frame_number(project_name, input_video, slider,timing_details)
                        
                        st.experimental_rerun()

                
                st.header("Load existing selections for styling and editing:")
                st.write("This will load up the existing selections for styling and editing. This will update the source image if there's one already available.")

                if st.button("Load up existing selections"):
                        for i in timing_details:                    
                            index_of_current_item = timing_details.index(i)
                            source_image = upload_image(f"videos/{project_name}/assets/frames/1_selected/{index_of_current_item}.png")                        
                            update_source_image(project_name, index_of_current_item, source_image)
                        st.success("Loaded up existing selections!")
        

        elif st.session_state.stage == "App Settings":

            app_settings = get_app_settings()

            st.write("API Keys:")

            with st.expander("Reveal API Keys:"):

                replicate_com_api_key = st.text_input("replicate_com_api_key", value = app_settings["replicate_com_api_key"])
                aws_access_key_id = st.text_input("aws_access_key_id", value = app_settings["aws_access_key_id"])
                aws_secret_access_key = st.text_input("aws_secret_access_key", value = app_settings["aws_secret_access_key"])

            if st.button("Save Settings"):
                update_app_setting("replicate_com_api_key", replicate_com_api_key)
                update_app_setting("aws_access_key_id", aws_access_key_id)
                update_app_setting("aws_secret_access_key", aws_secret_access_key)
                st.experimental_rerun()

           
            
                


        elif st.session_state.stage == "New Project":
            new_project_name = st.text_input("Project Name", value="")
            width = st.selectbox("Select video width", options=["512","704","1024"], key="video_width")
            height = st.selectbox("Select video height", options=["512","704","1024"], key="video_height")
            input_type = st.radio("Select input type", options=["Video","Image"], key="input_type", help="This will determine whether you guide the AI with a video or a series of images. You can always change this later.")

            if st.button("Create New Project"):                         
                create_working_assets(new_project_name)
                project_name = new_project_name
                update_project_setting("width", width, project_name)
                update_project_setting("height", height, project_name)  
                update_project_setting("input_type", input_type, project_name)
                st.session_state["project_name"] = project_name
                st.session_state["project_set"] = "Yes"            
                st.success("Project created - you can select it on the left.")
                time.sleep(1)
                st.experimental_rerun()
                

                                                    
       
        elif st.session_state.stage == "Frame Styling":  
            # if project_name/assets/frames/1_selected/0.png or project_name/assets/frames/1_selected/1.png exist, then we can proceed to the next stage 
            
            
            project_settings = get_project_settings(project_name)

            if "strength" not in st.session_state:
                    
                    st.session_state['strength'] = project_settings["last_strength"]
                    st.session_state['prompt'] = project_settings["last_prompt"]
                    st.session_state['model'] = project_settings["last_model"]
                    st.session_state['run_character_pipeline'] = project_settings["last_character_pipeline"]
                    st.session_state['negative_prompt'] = project_settings["last_negative_prompt"]
                    st.session_state['guidance_scale'] = project_settings["last_guidance_scale"]
                    st.session_state['seed'] = project_settings["last_seed"]
                    st.session_state['num_inference_steps'] = project_settings["last_num_inference_steps"]
                    st.session_state['which_stage_to_run_on'] = project_settings["last_which_stage_to_run_on"]

            if timing_details[0]["source_image"] == "":
                st.info("You need to select and load key frames first in the Key Frame Selection section.")
                    
            else:
                if "which_image" not in st.session_state:
                    st.session_state['which_image'] = 0
                if timing_details[0]["alternative_images"] != "":
                    variants = ast.literal_eval(timing_details[st.session_state['which_image']]["alternative_images"][1:-1])
                    number_of_variants = len(variants)
                    current_variant = int(timing_details[st.session_state['which_image']]["primary_image"])

                    f1, f2, f3  = st.columns([1,3,1])
                    with f1:
                        st.session_state['which_image'] = st.number_input('Go to Frame:', 0, len(timing_details)-1,help=f"There are {len(timing_details)} key frames in this video.")                                                        
                        
                        variants = ast.literal_eval(timing_details[st.session_state['which_image']]["alternative_images"][1:-1])
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
                else:   
                    f1, f2, f3  = st.columns([1,3,1])
                    with f1:
                        st.session_state['which_image'] = st.number_input('Go to Frame:', 0, len(timing_details)-1,help=f"There are {len(timing_details)} key frames in this video.")   

                
                
            
                if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                    img2=variants[which_variant]
                else:
                    img2='https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'          
                image_comparison(starting_position=10,
                    img1=timing_details[st.session_state['which_image']]["source_image"],
                    img2=img2)
                
                detail1, detail2, detail3 = st.columns([1,1,1])
                with detail1:
                    st.number_input(f"How many variants?", min_value=1, max_value=6, value=1, key=f"number_of_variants_{st.session_state['which_image']}")
                with detail2:
                    st.write("")
                    st.write("")
                    if st.button(f"Generate Variants for {st.session_state['which_image']}", key=f"new_variations_{st.session_state['which_image']}",help="This will generate new variants based on the settings to the left."):
                        st.session_state['restyle_button'] = 'yes'
                        st.session_state['item_to_restyle'] = st.session_state['which_image']                        
                        st.experimental_rerun()                    
                with detail3:   
                    with st.expander("Show/Hide Previous Last Prompt Details", expanded=False):
                        print("a")
                        # st.write(f"Prompt: {st.session_state['which_image']['prompt']}")
                        # st.write(f"Strength: {st.session_state['which_image']['strength']}")
                        # st.write(f"Model: {st.session_state['which_image']['model_id']}")

                if st.button("Create gif of current main variants"):
                    
                    create_gif_preview(project_name, timing_details)
                    st.image(f"videos/{project_name}/preview_gif.gif", use_column_width=True)                
                    st.balloons()
                    
                    

                timing_details = get_timing_details(project_name)
                for i in timing_details:
                    print("----------------")
                    print(i)
                    print("----------------")
        
                

                if len(timing_details) == 0:
                    st.info("You first need to select key frames at the Key Frame Selection stage.")

                st.sidebar.header("Restyle Frames")
                                
                

                models = get_models()

                models.append('sd')
                models.append('depth2img')
                models.append('pix2pix')

                # find the index of last_model in models

                index_of_last_model = models.index(st.session_state['model'])

                st.session_state['model'] = st.sidebar.selectbox(f"Model", models,index=index_of_last_model)

                


                st.session_state['prompt'] = st.sidebar.text_area(f"Prompt", label_visibility="visible", value=st.session_state['prompt'])
                if st.session_state['model'] != "sd" and st.session_state['model'] != "depth2img" and st.session_state['model'] != "pix2pix":
                    model_details = get_model_details(st.session_state['model'])
                    st.sidebar.info(f"Must include '{model_details['keyword']}' to run this model")                                    
                else:
                    if st.session_state['model'] == "pix2pix":
                        st.sidebar.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")                    
                st.session_state['strength'] = st.sidebar.number_input(f"Batch strength", value=float(st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)
                if project_settings["last_character_pipeline"] == "No":
                    index_of_run_character_pipeline = 1
                else:
                    index_of_run_character_pipeline = 0

                with st.sidebar.expander("Advanced settings 😏"):
                    st.session_state['negative_prompt'] = st.text_area(f"Negative prompt", value=st.session_state['negative_prompt'], label_visibility="visible")
                    st.session_state['guidance_scale'] = st.number_input(f"Guidance scale", value=float(st.session_state['guidance_scale']))
                    st.session_state['seed'] = st.number_input(f"Seed", value=int(st.session_state['seed']))
                    st.session_state['num_inference_steps'] = st.number_input(f"Inference steps", value=int(st.session_state['num_inference_steps']))
                    

                
                st.session_state['character_pipeline'] = st.sidebar.radio("Run character pipeline?", options=["Yes", "No"], index=index_of_run_character_pipeline, horizontal=True)

                if st.session_state['character_pipeline'] == "Yes":
                    with st.sidebar.expander("Character pipeline instructions"):
                        st.markdown("## How to use the character pipeline")                
                        st.markdown("1. Create a fine-tined model in the Custom Model section of the app - you'll need to use this to generat a consistent character.")
                        st.markdown("2. It's best to include a detailed prompt. We recommend taking an input image and running it through a CLIP Interrogator")
                        st.markdown("3. Use [expression] and [location] tags to vary the expression and location of the character dynamically if that changes throughout the clip. Varying this in the prompt will make the character look more natural.")
                        st.markdown("4. In our experience, the best strength for coherent character transformations is 0.25.")
                
                            
                range_start = st.sidebar.slider('Update From', 0, len(timing_details) -1, 0)

                range_end = st.sidebar.slider('Update To', 0, len(timing_details) - 1, 1)

                if project_settings["last_which_stage_to_run_on"] == "Current Main Variant":
                    index_of_which_stage_to_run_on = 1
                else:
                    index_of_which_stage_to_run_on = 0

                st.session_state['which_stage_to_run_on'] = st.sidebar.radio("What stage of images would you like to run this on?", options=["Extracted Frames", "Current Main Variant"], horizontal=True, index = index_of_which_stage_to_run_on, help="Extracted frames means the original frames from the video.")

                promote_new_generation = st.sidebar.checkbox("Promote new generation to main variant", value=True, key="promote_new_generation_to_main_variant")

                range_end = range_end + 1

                project_settings = get_project_settings(project_name)

                app_settings = get_app_settings()

                if 'restyle_button' not in st.session_state:
                    st.session_state['restyle_button'] = ''
                    st.session_state['item_to_restyle'] = ''


                if range_start <= range_end:

                    if st.sidebar.button(f'Batch restyle') or st.session_state['restyle_button'] == 'yes':

                        print(st.session_state['strength'])

                        if st.session_state['restyle_button'] == 'yes':
                            range_start = int(st.session_state['item_to_restyle'])
                            range_end = range_start + 1
                            st.session_state['restyle_button'] = ''
                            st.session_state['item_to_restyle'] = ''

                        for i in range(range_start, range_end): 
                                                
                            index_of_current_item = i
                            get_model_details(st.session_state['model'])                    
                            st.session_state['prompt'] = st.session_state['prompt'].replace(",", ".")                              
                            st.session_state['prompt'] = st.session_state['prompt'].replace("\n", "")
                            update_project_setting("last_prompt", st.session_state['prompt'], project_name)
                            update_project_setting("last_strength", st.session_state['strength'],project_name)
                            update_project_setting("last_model", st.session_state['model'], project_name)
                            update_project_setting("last_character_pipeline", st.session_state['run_character_pipeline'], project_name)
                            update_project_setting("last_negative_prompt", st.session_state['negative_prompt'], project_name)
                            update_project_setting("last_guidance_scale", st.session_state['guidance_scale'], project_name)
                            update_project_setting("last_seed", st.session_state['seed'], project_name)
                            update_project_setting("last_num_inference_steps",  st.session_state['num_inference_steps'], project_name)   
                            update_project_setting("last_which_stage_to_run_on", st.session_state['which_stage_to_run_on'], project_name)                       
                            if timing_details[index_of_current_item]["source_image"] == "":
                                source_image = upload_image("videos/" + str(project_name) + "/assets/frames/1_selected/" + str(index_of_current_item) + ".png")
                            else:
                                source_image = timing_details[index_of_current_item]["source_image"]
                            update_timing_values(project_name, index_of_current_item, st.session_state['prompt'], st.session_state['strength'], st.session_state['model'],st.session_state['run_character_pipeline'], st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'], source_image)                                                    
                            timing_details = get_timing_details(project_name)
                            if st.session_state['which_stage_to_run_on'] == "Extracted Frames":
                                source_image = timing_details[index_of_current_item]["source_image"]
                            else:
                                variants = ast.literal_eval(timing_details[index_of_current_item]["alternative_images"][1:-1])
                                number_of_variants = len(variants)                       
                                primary_image = int(timing_details[index_of_current_item]["primary_image"])
                                source_image = variants[primary_image]
                                
                            if st.session_state['run_character_pipeline'] == "Yes":                        
                                character_pipeline(index_of_current_item, project_name, project_settings, timing_details, source_image)
                            else:                            
                                restyle_images(index_of_current_item, project_name, project_settings, timing_details, source_image)
                            if promote_new_generation == True:                              
                                timing_details = get_timing_details(project_name)                            
                                variants = ast.literal_eval(timing_details[index_of_current_item]["alternative_images"][1:-1])                                                                 
                                number_of_variants = len(variants)                   
                                if number_of_variants == 1:
                                    print("No new generation to promote")
                                else:                            
                                    promote_image_variant(index_of_current_item, project_name, number_of_variants - 1)
                                                    

                        st.experimental_rerun()

                else:
                        
                        st.sidebar.write("Select a valid range")
                   
                    
                

        elif st.session_state.stage == "Frame Interpolation":
            if len(timing_details) == 0:
                st.info("You first need to select key frames and restyle them.")
            else:
                st.write("This is the frame interpolation view")
                timing_details = get_timing_details(project_name)
                key_settings = get_app_settings()
                total_number_of_videos = len(timing_details) - 1

                interpolation_steps = st.slider("Number of interpolation steps", min_value=1, max_value=8, value=4)
                with st.expander("Unsure what to pick? Click to see what this means."):
                    st.write("Interpolation steps are the number of frames to generate between each frame. We recommend varying the number of interpolation steps roughly based on how long the gap between each frame is is.")
                    st.write("0.17 seconds = 2 steps")
                    st.write("0.3 seconds = 3 steps")
                    st.write("0.57 seconds = 4 steps")
                    st.write("1.1 seconds = 5 steps")
                    st.write("2.17 seconds = 6 steps")
                    st.write("4.3 seconds = 7 steps")
                    st.write("8.57 seconds = 8 steps")

                which_video = st.select_slider("Which video to interpolate", options=["All","Single"])
                delete_existing_videos = st.checkbox("Delete existing videos:")

                if which_video == "All":

                    if st.button("Interpolate All Videos"):
                        if delete_existing_videos == True:
                            for filename in os.listdir("videos/" + str(project_name) + "/assets/videos/0_raw"):
                                os.remove("videos/" + str(project_name) + "/assets/videos/0_raw/" + filename)

                        for i in range(0, total_number_of_videos):

                            index_of_current_item = i
                            
                            if not os.path.exists("videos/" + str(project_name) + "/assets/videos/0_raw/" + str(index_of_current_item) + ".mp4"):

                                if total_number_of_videos == index_of_current_item:

                                    current_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(index_of_current_item) + ".png"

                                    final_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(key_settings["ending_image"])

                                    prompt_interpolation_model(current_image_location, final_image_location, project_name, index_of_current_item,
                                                            interpolation_steps, key_settings["replicate_com_api_key"])

                                else:
                                    current_image_variants = timing_details[index_of_current_item]["alternative_images"]        
                                    current_image_variants = current_image_variants[1:-1]
                                    current_image_variants = ast.literal_eval(current_image_variants)
                                    current_image_number = timing_details[index_of_current_item]["primary_image"]
                                    current_image_location = current_image_variants[current_image_number]
                                    next_image_variants = timing_details[index_of_current_item+1]["alternative_images"]        
                                    next_image_variants = next_image_variants[1:-1]
                                    next_image_variants = ast.literal_eval(next_image_variants)
                                    next_image_number = timing_details[index_of_current_item+1]["primary_image"]
                                    next_image_location = next_image_variants[next_image_number]

                                    prompt_interpolation_model(current_image_location, next_image_location, project_name, index_of_current_item,
                                                            interpolation_steps, key_settings["replicate_com_api_key"])

                else:
                    specific_video = st.number_input("Which video to interpolate", min_value=0, max_value=total_number_of_videos, value=0)

                    if st.button("Interpolate this video"):
                        
                        current_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(specific_video) + ".png"

                        next_image_location = "videos/" + str(project_name) + "/assets/frames/2_character_pipeline_completed/" + str(specific_video+1) + ".png"

                        prompt_interpolation_model(current_image_location, next_image_location, project_name, specific_video,
                                                            interpolation_steps, key_settings["replicate_com_api_key"])


            

          

               


        elif st.session_state.stage == "Video Rendering":
            final_video_name = st.text_input("What would you like to name this video?")

            st.file_uploader("Attach music", type=["mp3"], help="Attach music to your video")

            delete_existing_videos = st.checkbox("Delete all the existing timings", value=True)

            if st.button("Render Video"):
                if delete_existing_videos == True:
                    for filename in os.listdir("videos/" + str(project_name) + "/assets/videos/1_final"):
                        os.remove("videos/" + str(project_name) + "/assets/videos/1_final/" + filename)
                timing_details = get_timing_details(project_name)
                total_number_of_videos = len(timing_details) - 2
                calculate_desired_duration_of_each_clip(timing_details, project_name)
                timing_details = get_timing_details(project_name)
        
                for i in timing_details:
                    
                    index_of_current_item = timing_details.index(i)

                    if index_of_current_item <= total_number_of_videos:

                        if not os.path.exists("videos/" + str(project_name) + "/assets/videos/1_final/" + str(index_of_current_item) + ".mp4"):

                            total_duration_of_clip = timing_details[index_of_current_item]['duration_of_clip']

                            total_duration_of_clip = float(total_duration_of_clip)

                            if index_of_current_item == total_number_of_videos:

                                total_duration_of_clip = timing_details[index_of_current_item]['duration_of_clip']
                                duration_of_static_time = float(timing_details[index_of_current_item]['static_time'])
                                duration_of_static_time = float(duration_of_static_time) / 2

                            elif index_of_current_item == 0:

                                duration_of_static_time = float(timing_details[index_of_current_item]['static_time'])

                            else:

                                duration_of_static_time = float(timing_details[index_of_current_item]['static_time'])

                             
                            update_video_speed(project_name, index_of_current_item, duration_of_static_time, total_duration_of_clip)

                video_list = []

                for i in timing_details:

                    index_of_current_item = timing_details.index(i)

                    if index_of_current_item <= total_number_of_videos:

                        index_of_current_item = timing_details.index(i)

                        video_list.append("videos/" + str(project_name) +
                                        "/assets/videos/1_final/" + str(index_of_current_item) + ".mp4")

                video_clips = [VideoFileClip(v) for v in video_list]

                finalclip = concatenate_videoclips(video_clips)

                # finalclip = finalclip.set_audio(AudioFileClip(
                #  "videos/" + video_name + "/assets/resources/music/" + key_settings["song"]))

                finalclip.write_videofile(f"videos/{project_name}/assets/videos/2_completed/{final_video_name}.mp4", fps=60,  audio_bitrate="1000k", bitrate="4000k", codec="libx264")

                video = VideoFileClip(f"videos/{project_name}/assets/videos/2_completed/{final_video_name}.mp4")

                st.experimental_rerun()

            # find all the .mp4 files in the folder

            video_list = [list_of_files for list_of_files in os.listdir(
                "videos/" + project_name + "/assets/videos/2_completed") if list_of_files.endswith('.mp4')]

            # sort the videos reverse chronologically

            video_dir = "videos/" + project_name + "/assets/videos/2_completed"

            video_list.sort(key=lambda f: int(re.sub('\D', '', f)))

            video_list = sorted(video_list, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)


            

            # list them on the page

            for video in video_list:

                st.subheader(video)       

                st.write(datetime.datetime.fromtimestamp(
                    os.path.getmtime("videos/" + project_name + "/assets/videos/2_completed/" + video)))

                st.video(f"videos/{project_name}/assets/videos/2_completed/{video}")

                # add a button to delete the video

                col1, col2 = st.columns(2)

                with col1:

                    if st.checkbox(f"Confirm {video} Deletion"):

                        if st.button(f"Delete {video}"):
                            os.remove("videos/" + project_name +
                                    "/assets/videos/2_completed/" + video)
                            st.experimental_rerun()
                    else:
                        st.button(f"Delete {video}",disabled=True)

                        

        elif st.session_state.stage == "Project Settings":

            st.write("Project Settings")
            

        elif st.session_state.stage == "Custom Models":
            
            app_settings = get_app_settings()

            tab1, tab2 = st.tabs(["Train New Model", "See Existing Models"])

            with tab1:

                images_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/training_data')]        
                images_list.sort(key=lambda f: int(re.sub('\D', '', f)))

                files_uploaded = ''

                uploaded_files = st.file_uploader("Add training images here", accept_multiple_files=True)
                if uploaded_files is not None:
                    for uploaded_file in uploaded_files:
                        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
                        st.write(file_details)
                        img = Image.open(uploaded_file)        
                        with open(os.path.join(f"videos/{project_name}/assets/resources/training_data",uploaded_file.name),"wb") as f: 
                            f.write(uploaded_file.getbuffer())         
                            st.success("Saved File") 
                            # apend the image to the list
                            images_list.append(uploaded_file.name)
                                                            
                images_for_model = []
                                 
                for image_name in images_list:
                    index_of_current_item = images_list.index(image_name)
                    st.subheader(f'{image_name}:')                        
                    image = Image.open(f'videos/{project_name}/assets/resources/training_data/{image_name}') 
                    st.image(image, width=400) 
                    yes = st.checkbox(f'Add {index_of_current_item} to model')    
                    if yes:
                        images_for_model.append(image_name)
                    else:
                        if index_of_current_item in images_for_model:
                            images_for_model.remove(index_of_current_item)
                
                st.sidebar.subheader("Train new model")
                model_name = st.sidebar.text_input("Model name:",value="", help="No spaces or special characters please")
                instance_prompt = st.sidebar.text_input("Trigger word:",value="", help = "This is the word that will trigger the model")
                class_prompt = st.sidebar.text_input("Describe what your prompts depict generally:",value="", help="This will help guide the model to learn what you want it to do")
                max_train_steps = st.sidebar.number_input("Max training steps:",value=2000, help=" The number of training steps to run. Fewer steps make it run faster but typically make it worse quality, and vice versa.")
                st.sidebar.write(f"You've selected {len(images_for_model)} images.")
                
                if len(images_for_model) < 5 or model_name == "" or instance_prompt == "" or class_prompt == "":
                    st.sidebar.write("Select at least 5 images and fill in all the fields to train a new model.")
                    st.sidebar.button("Train Model",disabled=True)
                else:
                    if st.sidebar.button("Train Model",disabled=False):
                        model_status = train_model(app_settings,images_for_model, instance_prompt,class_prompt,max_train_steps,model_name, project_name)
                        st.sidebar.success(model_status)
            
            with tab2:
                models = get_models()           
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
                        model_details["keyword"]
                    with col3:
                        model_details["id"]
                    with col4:
                        st.image(ast.literal_eval(model_details["training_images"])[0])
                    with col5:
                        st.image(ast.literal_eval(model_details["training_images"])[1])
                    with col6:
                        st.image(ast.literal_eval(model_details["training_images"])[2])
                    st.markdown("***")
                                        

                  
        elif st.session_state.stage == "Frame Editing":

            if timing_details[0]["source_image"] == "":
                st.info("You need to add  key frames first in the Key Frame Selection section.")

            else:

                timing_details = get_timing_details(project_name)
                project_settings = get_project_settings(project_name)

                #initiative value
                if "which_image" not in st.session_state:
                    st.session_state['which_image'] = 0
                
                def reset_new_image():
                    st.session_state['edited_image'] = ""
            
                    
                f1, f2, f3 = st.columns([1,2,1])
                with f1:
                    st.session_state['which_image'] = st.number_input('Go to Frame:', 0, len(timing_details)-1,help=f"There are {len(timing_details)} key frames in this video.", on_change=reset_new_image)
                    
                with f2:                
                    which_stage = st.radio('Select stage:', ["Unedited Key Frame", "Styled Key Frame"], horizontal=True, on_change=reset_new_image)
                    # st.session_state['which_image'] = st.slider('Select image to edit:', 0, len(timing_details)-1, st.session_state["which_image"])

                with f3:                
                    if which_stage == "Unedited Key Frame":     
                        st.write("")                     
                        if st.button("Reset Key Frame", help="This will reset the base key frame to the original unedited version. This will not affect the video."):
                            extract_frame(int(st.session_state['which_image']), project_name, project_settings["input_video"], timing_details[st.session_state['which_image']]["frame_number"],timing_details)
                            new_source_image = upload_image(f"videos/{project_name}/assets/frames/1_selected/{st.session_state['which_image']}.png")
                            update_source_image(project_name, st.session_state['which_image'], new_source_image)
                            st.experimental_rerun()
                                    
                if "edited_image" not in st.session_state:
                    st.session_state.edited_image = ""                        

                if which_stage == "Unedited Key Frame":
                    bg_image = timing_details[st.session_state['which_image']]["source_image"]
                elif which_stage == "Styled Key Frame":            
                    variants = ast.literal_eval(timing_details[st.session_state['which_image']]["alternative_images"][1:-1])
                    primary_image = timing_details[st.session_state['which_image']]["primary_image"]             
                    bg_image = variants[primary_image]
            
                width = int(project_settings["width"])
                height = int(project_settings["height"])

                
                st.sidebar.markdown("### Select Area To Edit:") 
                type_of_mask_selection = st.sidebar.radio("How would you like to select what to edit?", ["Automated Background Selection", "Automated Layer Selection", "Manual Background Selection"], horizontal=True)
            
                if "which_layer" not in st.session_state:
                    st.session_state['which_layer'] = "Background"

                if type_of_mask_selection == "Automated Layer Selection":
                    st.session_state['which_layer'] = st.sidebar.selectbox("Which layer would you like to replace?", ["Background", "Middleground", "Foreground"])


                if type_of_mask_selection == "Manual Background Selection":
                    if st.session_state['edited_image'] == "":
                        canvas_image = r.get(bg_image)
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
                            background_image=Image.open(BytesIO(canvas_image.content)),
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
                            im.save(f"mask.png", "PNG")
                    else:
                        image_comparison(
                            img1=bg_image,
                            img2=st.session_state['edited_image'], starting_position=10, label1="Original", label2="Edited")  
                        if st.button("Reset Canvas"):
                            st.session_state['edited_image'] = ""
                            st.experimental_rerun()
                
                elif type_of_mask_selection == "Automated Background Selection" or type_of_mask_selection == "Automated Layer Selection":
                    if st.session_state['edited_image'] == "":
                        st.image(bg_image)
                    else:
                        image_comparison(
                            img1=bg_image,
                            img2=st.session_state['edited_image'], starting_position=10, label1="Original", label2="Edited")                    
                        

            

                st.sidebar.markdown("### Edit Individual Image:") 
                
                type_of_mask_replacement = st.sidebar.radio("Select type of edit", ["Replace With Image", "Inpainting"], horizontal=True)            

                
                btn1, btn2 = st.sidebar.columns([1,1])

                if type_of_mask_replacement == "Replace With Image":
                    background_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/backgrounds') if f.endswith('.png')]                 
                    with btn1:
                        background_image = st.sidebar.selectbox("Range background", background_list)
                        if background_list != []:
                            st.image(f"videos/{project_name}/assets/resources/backgrounds/{background_image}", use_column_width=True)
                    with btn2:
                        uploaded_files = st.file_uploader("Add more background images here", accept_multiple_files=True)                    
                        if uploaded_files is not None:
                            for uploaded_file in uploaded_files:
                                file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
                                st.write(file_details)
                                img = Image.open(uploaded_file)        
                                with open(os.path.join(f"videos/{project_name}/assets/resources/backgrounds",uploaded_file.name),"wb") as f:                                 
                                    st.success("Your backgrounds are uploaded file, refresh the page to see them.")                     
                                    background_list.append(uploaded_file.name)
                elif type_of_mask_replacement == "Inpainting":
                    with btn1:
                        prompt = st.text_input("Prompt:", help="Describe the whole image, but focus on the details you want changed!")
                    with btn2:
                        negative_prompt = st.text_input("Negative Prompt:", help="Enter any things you want to make the model avoid!")

                edit1, edit2 = st.sidebar.columns(2)

                with edit1:
                    if st.button(f'Run Edit On Current Image'):
                        if type_of_mask_replacement == "Inpainting":
                            st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, "", bg_image, prompt, negative_prompt,width, height,st.session_state['which_layer'])
                        elif type_of_mask_replacement == "Replace With Image":
                            st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, bg_image, "", "",width, height,st.session_state['which_layer'])
                        st.experimental_rerun()
                with edit2:
                    if st.session_state['edited_image'] != "":                                     
                        if st.button("Promote Last Image", type="primary"):
                            if which_stage == "Unedited Key Frame":                        
                                update_source_image(project_name, st.session_state['which_image'], st.session_state['edited_image'])
                            elif which_stage == "Styled Key Frame":
                                number_of_image_variants = add_image_variant(st.session_state['edited_image'], st.session_state['which_image'], project_name, timing_details)
                                promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1)
                            st.session_state['edited_image'] = ""
                            st.success("Image promoted!")
                    else:
                        if st.button("Run Edit & Promote"):
                            if type_of_mask_replacement == "Inpainting":
                                st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, "", bg_image, prompt, negative_prompt,width, height,st.session_state['which_layer'])
                            elif type_of_mask_replacement == "Replace With Image":
                                st.session_state['edited_image'] = execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, bg_image, "", "",width, height,st.session_state['which_layer'])
                            if which_stage == "Unedited Key Frame":                        
                                update_source_image(project_name, st.session_state['which_image'], st.session_state['edited_image'])
                            elif which_stage == "Styled Key Frame":
                                number_of_image_variants = add_image_variant(st.session_state['edited_image'], st.session_state['which_image'], project_name, timing_details)
                                promote_image_variant(st.session_state['which_image'], project_name, number_of_image_variants - 1)
                            st.session_state['edited_image'] = ""
                            st.success("Image promoted!")
                            st.experimental_rerun()
                            
                            
                            
                        

                st.sidebar.markdown("### Batch Run Edits:")   
                st.sidebar.write("This will batch run the settings you have above on a batch of images.")     
                batch_run_from = st.sidebar.slider("From:", 1, 0, (0, len(timing_details)-1))
                batch_run_to = st.sidebar.slider("To:", 1, 0, (0, len(timing_details)-1))
                if which_stage == "Unedited Key Frame":
                    st.sidebar.warning("This will overwrite the source images in the range you select - you can always reset them if you wish.")
                elif which_stage == "Styled Key Frame":
                    make_primary_variant = st.sidebar.checkbox("Make primary variant", value=True, help="If you want to make the edited image the primary variant, tick this box. If you want to keep the original primary variant, untick this box.")          
                if st.sidebar.button("Batch Run Edit"):
                    for i in range(batch_run_from, batch_run_to):
                        if which_stage == "Unedited Key Frame":
                            bg_image = timing_details[i]["source_image"]
                            edited_image = execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, bg_image, prompt, negative_prompt, width, height,st.session_state['which_layer'])
                            update_source_image(project_name, st.session_state['which_image'], edited_image)
                        elif which_stage == "Styled Key Frame":
                            variants = ast.literal_eval(timing_details[st.session_state['which_image']]["alternative_images"][1:-1])
                            primary_image = timing_details[st.session_state['which_image']]["primary_image"]             
                            bg_image = variants[primary_image]   
                            edited_image = execute_image_edit(type_of_mask_selection, type_of_mask_replacement, project_name, background_image, bg_image, prompt, negative_prompt, width, height,st.session_state['which_layer'])   
                            number_of_image_variants = add_image_variant(edited_image, st.session_state['which_image'], project_name, timing_details)
                            if make_primary_variant == True:
                                promote_image_variant(i, project_name, number_of_image_variants-1)

                    



            
            
if __name__ == '__main__':
    main()

