import streamlit as st
import os
from PIL import Image
from streamlit_image_comparison import image_comparison
import cv2
import re
import csv
import pandas as pd
import replicate
import urllib
import requests as r
import base64
import time

def move_frame(project_name, index_of_current_item, distance_to_move, input_video,frame_time):

    frame_number = calculate_frame_number_at_time(input_video, frame_time) + distance_to_move
    print(frame_number)
    extract_frame(index_of_current_item, project_name, input_video, frame_number)
    st.experimental_rerun()


def swap_background():
    print('Run')
# Create a function for the main view


def get_key_settings(csv_file):

    key_settings = {}

    with open(csv_file, 'r') as f:

        lines = [line.split(',') for line in f.read().splitlines()]

    for i in range(1, 24):

        key_settings[lines[i][11]] = lines[i][12]

    return key_settings

def create_working_assets(video_name):

    os.mkdir("videos/" + video_name + "/assets")

    os.mkdir("videos/" + video_name + "/assets/frames")

    os.mkdir("videos/" + video_name + "/assets/frames/0_extracted")
    os.mkdir("videos/" + video_name + "/assets/frames/1_character_pipeline_completed")
    os.mkdir("videos/" + video_name + "/assets/frames/2_backdrop_pipeline_completed")

    os.mkdir("videos/" + video_name + "/assets/resources")

    os.mkdir("videos/" + video_name + "/assets/resources/backgrounds")
    os.mkdir("videos/" + video_name + "/assets/resources/masks")
    os.mkdir("videos/" + video_name + "/assets/resources/music")
    os.mkdir("videos/" + video_name + "/assets/resources/training_data")

    os.mkdir("videos/" + video_name + "/assets/videos")

    os.mkdir("videos/" + video_name + "/assets/videos/0_raw")
    os.mkdir("videos/" + video_name + "/assets/videos/1_final")

def update_key_setting(key, pair_value, project_name):
    
    csv_file_path = f'videos/{project_name}/settings.csv'
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        rows = []

        for row in csv_reader:
            if row[11] == key:            
                row_number = csv_reader.line_num - 2
                print(row_number)
                new_value = pair_value        
    
    # open the csv file in write mode
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Update the value on the 10th row of the 12th column
    df.iat[row_number, 12] = new_value

    # Save the updated DataFrame to a new CSV file
    df.to_csv(csv_file_path, index=False)


def get_timing_details(video_name):

    timing_details = []

    with open(("videos/" + str(video_name) + "/settings.csv"), 'r') as f:

        lines = [line.split(',') for line in f.read().splitlines()]

    number_of_rows = (len)(lines)


    for i in range(1, number_of_rows):        

        current_frame = {}

        current_frame["frame_time"] = lines[i][1]

        if current_frame["frame_time"] != "":
            
            current_frame["total_duration_of_clip"] = ""
            current_frame["model_id"] = lines[i][2]
            current_frame["prompt"] = lines[i][3]
            current_frame["duration_of_static_time"] = lines[i][4]
            current_frame["duration_of_morph_time"] = ""
            current_frame["background"] = lines[i][8]
            current_frame["mask"] = lines[i][9]
            current_frame["notes"] = lines[i][5]
            current_frame["whats_happening"] = lines[i][6]
            current_frame["shot"] = lines[i][7]

            timing_details.append(current_frame)

    return timing_details


def calculate_frame_number_at_time(input_video, time_of_frame):

    time_of_frame = float(time_of_frame)

    video = cv2.VideoCapture(input_video)

    frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(video.get(cv2.CAP_PROP_FPS))

    length_of_video = float(frame_count / fps)

    percentage_of_video = float(time_of_frame / length_of_video)

    frame_number = int(percentage_of_video * frame_count)

    if frame_number == 0:
        frame_number = 1

    return frame_number



def extract_all_frames(input_video, project_name, timing_details, time_per_frame):

    folder = 'videos/' + str(project_name) + '/assets/frames/0_extracted'

    for filename in os.listdir(folder):
        os.remove(os.path.join(folder, filename))

    update_key_setting('time_per_frame', time_per_frame, project_name)

    key_settings = get_key_settings("videos/" + str(project_name) + "/settings.csv")

    timing_details = get_timing_details(input_video,float(key_settings["time_per_frame"]), key_settings["model_id"],key_settings["master_prompt"], int(key_settings["static_time"]), key_settings["background_image"], key_settings["custom_or_stable"], project_name)

    for i in timing_details:

        index_of_current_item = timing_details.index(i)
    
        time_of_frame = float(timing_details[index_of_current_item]["frame_time"])

        extract_frame_number = calculate_frame_number_at_time(input_video, time_of_frame)

        extract_frame(index_of_current_item, project_name, input_video, extract_frame_number)

def extract_frame(frame_number, video_name, input_video_file, extract_frame_number):

    input_video_file = cv2.VideoCapture(input_video_file)

    input_video_file.set(cv2.CAP_PROP_POS_FRAMES, extract_frame_number)

    ret, frame = input_video_file.read()

    cv2.imwrite("videos/" + video_name + "/assets/frames/0_extracted/" + str(frame_number) + ".png", frame)

    img = Image.open("videos/" + video_name + "/assets/frames/0_extracted/" + str(frame_number) + ".png")

    img.save("videos/" + video_name + "/assets/frames/0_extracted/" + str(frame_number) + ".png")

    print(f"{frame_number}.png extracted!")

    return str(frame_number) + ".png"

def touch_up_images(video_name, replicate_api_key, index_of_current_item, type_of_touch_up, custom_touch_up_query, dreamstudio_ai_api_key, original_prompt):

    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    model = replicate.models.get("tencentarc/gfpgan")

    image = "videos/" + str(video_name) + "/assets/frames/1_character_pipeline_completed/" + str(index_of_current_item) + ".png"

    output = model.predict(img=open(image, "rb"))

    try:

        urllib.request.urlretrieve(output, image)

    except Exception as e:

        print("Error:")

def resize_image(video_name, image_number, new_width,new_height):

    image = Image.open("videos/" + str(video_name) + "/assets/frames/1_character_pipeline_completed/" + str(image_number) + ".png")

    resized_image = image.resize((new_width, new_height))

    resized_image.save("videos/" + str(video_name) + "/assets/frames/1_character_pipeline_completed/" + str(image_number) + ".png")

    return resized_image

def face_swap(replicate_api_key, video_name, index_of_current_item,stablediffusionapi_com_api_key):
    
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    model = replicate.models.get("arielreplicate/ghost_face_swap")

    version = model.versions.get("106df0aaf9690354379d8cd291ad337f6b3ea02fe07d90feb1dafd64820066fa")

    source_face = upload_image("videos/" + str(video_name) + "/face.png", stablediffusionapi_com_api_key)

    target_face = upload_image("videos/" + str(video_name) + "/assets/frames/0_extracted/" + str(index_of_current_item) + ".png", stablediffusionapi_com_api_key)

    output = version.predict(source_path=source_face, target_path=target_face,use_sr=0)

    new_image = "videos/" + str(video_name) + "/assets/frames/1_character_pipeline_completed/" + str(index_of_current_item) + ".png"
    
    try:

        urllib.request.urlretrieve(output, new_image)

    except Exception as e:

        print(e)

def prompt_model_stability(videoname, image_number, prompt, dreamstudio_ai_api_key, apend_to_prompt):

    os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

    os.environ['STABILITY_KEY'] = dreamstudio_ai_api_key

    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'],  # API Key reference.
        verbose=True,  # Print debug messages.
        engine="stable-diffusion-v1-5",
    )

    with open("videos/" + videoname + "/assets/frames/2_background_added/" + str(image_number) + ".png", "rb") as image:
        # extract binary from image
        image_bytes = image.read()

    # convert binary to base64
    img = Image.open(io.BytesIO(image_bytes))

    prompt = str(prompt) + "," + str(apend_to_prompt)

    response = stability_api.generate(
        prompt=prompt,
        init_image=img,
        start_schedule=0.6,
        steps=50,
        cfg_scale=10.0,
        width=1280,
        height=704,
        sampler=generation.SAMPLER_K_DPM_2_ANCESTRAL
    )

    print(response)

    for resp in response:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img2 = Image.open(io.BytesIO(artifact.binary))
                img2.save("videos/" + videoname +
                          "/assets/frames/3_restyled/" + str(image_number) + ".png")

def prompt_model_dreambooth(strength, folder_name,video_name, image_number, init_image, prompt, model_id, sd_api_key, apend_to_prompt, mask_image):


    sd_url = "https://stablediffusionapi.com/api/v4/dreambooth/img2img"

    init_image = upload_image("videos/" + str(video_name) + "/assets/frames/" + str(folder_name) + "/" + str(image_number) + ".png", sd_api_key)
        
    payload = {
        "key": sd_api_key,
        "prompt": "bukowski man. a close up of a person wearing a shirt. a character portrait. inspired by Paul Cadmus. behance contest winner. gigachad meme. in style of disney animation. smiling at the viewer. photography photorealistic. naturalistic technique. discord profile picture. blair armitage. silvio berlusconi. vladimir volego",
        # "prompt": str(prompt) + "," + str(apend_to_prompt),
        "width": "720",
        "height": "480",
        "samples": "1",
        "num_inference_steps": "20",
        "seed": "0",
        "guidance_scale": "7",
        "webhook": "0",
        "strength": strength,
        "track_id": "null",
        "init_image": init_image,
        "model_id": model_id

    }

    print(payload)

    completed = "false"

    response = r.post(sd_url, json=payload)

    while completed == "false":

        if response.json()["status"] == "processing":

            wait = int(response.json()["eta"])

            print("Processing, ETA: " + str(wait) + " seconds")

            time.sleep(wait)

            response = "https://stablediffusionapi.com/api/v3/dreambooth/fetch/" + str(response.json()["id"])

        elif response.json()["status"] == "success":

            time.sleep(3)

            output_url = response.json()["output"][0]

            image = r.get(output_url)

            with open("videos/" + str(video_name) + "/assets/frames/" + str(folder_name) + "/"+ str(image_number)+".png", "wb") as f:

                f.write(image.content)

            completed = "true"

        else:

            print("Something went wrong, trying again in 30 seconds.")

            print(response)

            time.sleep(30)

    return completed


def upload_image(image_location, sd_api_key):

    upload_success = "false"

    upload_attempts = 0

    while upload_success == "false":

        sd_api_key = sd_api_key

        sd_url = "https://stablediffusionapi.com/api/v3/base64_crop"

        # resize_image(video_name,image_number, 720, 480)

        with open(image_location, "rb") as image_file:

            image_bytes = image_file.read()

        encoded_string = base64.b64encode(image_bytes)

        upload_string = "data:image/png;base64," + \
            encoded_string.decode("utf-8").replace(" ", "")

        payload = {
            "key": sd_api_key,
            "image": upload_string,
            "crop": "false"
        }

        response = r.post(sd_url, json=payload)

        if response.json()["status"] == "success":

            upload_success = "true"

            print (response.json()["link"])

            return response.json()["link"]

        else:

            time.sleep(30)

            print(response.text)

def hair_swap(replicate_api_key, video_name, index_of_current_item,stablediffusionapi_com_api_key):

    os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    model = replicate.models.get("cjwbw/style-your-hair")

    version = model.versions.get("c4c7e5a657e2e1abccd57625093522a9928edeccee77e3f55d57c664bcd96fa2")

    source_hair = upload_image("videos/" + str(video_name) + "/face.png", stablediffusionapi_com_api_key)

    target_hair = upload_image("videos/" + str(video_name) + "/assets/frames/1_character_pipeline_completed/" + str(index_of_current_item) + ".png", stablediffusionapi_com_api_key)

    output = version.predict(source_image=source_hair, target_image=target_hair)

    new_image = "videos/" + str(video_name) + "/assets/frames/1_character_pipeline_completed/" + str(index_of_current_item) + ".png"

    try:

        urllib.request.urlretrieve(output, new_image)

    except Exception as e:

        print(e)

def restyle_images(strength, folder_name,video_name, frame_number, prompt, model_id, stablediffusionapi_com_api_key, dreamstudio_ai_api_key, apend_to_prompt, mask):

    if model_id == "sd":
        prompt_model_stability(strength, folder_name,video_name, frame_number,prompt, dreamstudio_ai_api_key, apend_to_prompt)

    else:
        prompt_model_dreambooth(strength, folder_name,video_name, frame_number, str(frame_number) + ".png", prompt, model_id, stablediffusionapi_com_api_key, apend_to_prompt, mask)



def character_pipeline(index_of_current_item, project_name, key_settings, timing_details):

    print("Running character pipeline on " + str(index_of_current_item) + ".png")

    face_swap(key_settings["replicate_com_api_key"], project_name, index_of_current_item, key_settings["stablediffusionapi_com_api_key"])

    touch_up_images(project_name, key_settings["replicate_com_api_key"], index_of_current_item, key_settings["type_of_touch_up"],key_settings["custom_touch_up_query"], key_settings["dreamstudio_ai_api_key"], timing_details[index_of_current_item]["prompt"])

    resize_image(project_name, index_of_current_item, 704,512)

    # hair_swap(key_settings["replicate_com_api_key"],video_name,index_of_current_item,key_settings["stablediffusionapi_com_api_key"])

    # INSERT CLOTHES SWAP

    restyle_images(0.25,"1_character_pipeline_completed",project_name, index_of_current_item, timing_details[index_of_current_item]["prompt"], timing_details[index_of_current_item]["model_id"], key_settings["stablediffusionapi_com_api_key"], key_settings["dreamstudio_ai_api_key"], key_settings["what_to_apend_to_each_prompt"], timing_details[index_of_current_item]["mask"])

    st.experimental_rerun()


def main():

    if 'stage' not in st.session_state:
        st.session_state['stage'] = ''

    header1, header2 = st.sidebar.columns([3, 1])

    header1.title("Banodoco")
    header2.button("Settings")

        
    project_name = "eyebrow_demo"
    project_name = st.sidebar.selectbox("Select an option", os.listdir("videos"),index=3)
    

    if project_name == "":

        st.write("No projects found")

    else:

        #key_settings = get_key_settings("videos/" + str(project_name) + "/settings.csv")

        if not os.path.exists("videos/" + project_name + "/assets"):

            create_working_assets(project_name)

        timing_details = get_timing_details(project_name)

        print(timing_details)

        # video = cv2.VideoCapture(input_video)

        # frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))

        st.session_state.stage = st.sidebar.radio("Select an option",
                                    ["Project Settings",
                                    "Train Model",
                                    "Key Frame Selection",
                                    "Background Replacement",
                                    "Frame Styling",
                                    "Frame Interpolation"])

        st.header(st.session_state.stage)
        
        if st.session_state.stage == "Key Frame Selection":

            images_list = [f for f in os.listdir(f'videos/{project_name}/assets/frames/0_extracted') if f.endswith('.png')]
    
            images_list.sort(key=lambda f: int(re.sub('\D', '', f)))


            if len(images_list) == 0:

                st.header("<------- Extract Key Frames Here")
                    
            else:                

                st.sidebar.subheader("Extract key frames from video")

                time_per_frame = st.sidebar.number_input("How many seconds per key frame?", value = 0.5)

                input_video_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/input_videos') if f.endswith('.mp4')]
                
                background_list = os.listdir(f'videos/{project_name}/assets/resources/backgrounds')
                

                input_video = st.sidebar.selectbox("Input video:", input_video_list)

                if st.sidebar.checkbox("I understand that running this will remove all existing frames"):

                    if st.sidebar.button("Extract Key Frames"):

                        replace_existing_timing(project_name, input_video, time_per_frame)

                        get_timing_details(project_name)

                        extract_all_frames(input_video, project_name, timing_details,time_per_frame)

                        st.experimental_rerun()
                else:

                    st.sidebar.button("Extract Key Frames", disabled=True)


                for image_name in images_list:
                
                    image = Image.open(f'videos/{project_name}/assets/frames/0_extracted/{image_name}')            
                    st.subheader(f'Image Name: {image_name}')                
                    st.image(image, use_column_width=True)
                    
                    col1, col2,col3, col4 = st.columns(4)

                    with col1:

                        if st.button(f'-10 Frames ({image_name})'):                                    
                            image_number = int(image_name.replace(".png", ""))
                            frame_time = timing_details[image_number]['frame_time']
                            move_frame(project_name, image_number, -10,input_video,frame_time)

                    with col2:

                        if st.button(f'-5 Frames ({image_name})'):
                            image_number = int(image_name.replace(".png", ""))
                            frame_time = timing_details[image_number]['frame_time']
                            move_frame(project_name, image_number, -5,input_video,frame_time)

                    with col3:

                        if st.button(f'+5 Frame ({image_name})'):
                            image_number = int(image_name.replace(".png", ""))
                            frame_time = timing_details[image_number]['frame_time']
                            move_frame(project_name, image_number, 5,input_video,frame_time)

                    with col4:

                        if st.button(f'+10 Frame ({image_name})'):
                            image_number = int(image_name.replace(".png", ""))
                            frame_time = timing_details[image_number]['frame_time']
                            move_frame(project_name, image_number, 10,input_video,frame_time)



        elif st.session_state.stage == "Background Replacement":
                      
            images_list = [f for f in os.listdir(f'videos/{project_name}/assets/frames/0_extracted') if f.endswith('.png')]

            images_list = sorted(images_list)

            images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
            
            background_list = os.listdir(f'videos/{project_name}/assets/resources/backgrounds')
            
            st.sidebar.header("Batch background replacement")

            range_start = st.sidebar.slider('Update From', 0, len(images_list), 1)

            range_end = st.sidebar.slider('Update To', 0, len(images_list), 1)

            background_image = st.sidebar.selectbox("Range background", background_list)

            st.sidebar.image(f"videos/{project_name}/assets/resources/backgrounds/{background_image}", use_column_width=True)

            if range_start <= range_end:

                if st.sidebar.button(f'Swap background'):

                    for i in range(range_start, range_end):
                        swap_background()

            else:
                    
                    st.sidebar.write("Select a valid range")

            uploaded_files = st.sidebar.file_uploader("Add more background images here", accept_multiple_files=True)
            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
                    st.write(file_details)
                    img = Image.open(uploaded_file)        
                    with open(os.path.join(f"videos/{project_name}/assets/resources/backgrounds",uploaded_file.name),"wb") as f: 
                        f.write(uploaded_file.getbuffer())         
                        st.success("Saved File") 
                        # apend the image to the list
                        images_list.append(uploaded_file.name)
            

            for image_name in images_list:
            
                image = Image.open(f'videos/{project_name}/assets/frames/0_extracted/{image_name}')            
            
                st.subheader(f'{image_name}')                

                st.image(image, use_column_width=True)
                                        


       
        elif st.session_state.stage == "Frame Styling":
        
            images_list = [f for f in os.listdir(f'videos/{project_name}/assets/frames/1_character_pipeline_completed') if f.endswith('.png')]
    
            images_list.sort(key=lambda f: int(re.sub('\D', '', f)))

            if len(images_list) == 0:
                st.write("No frames extracted yet")
                if st.button("Styling Frames"):
                    st.write("Styling frames")

            else:
                st.sidebar.header("Batch style")
                st.sidebar.text_input(f"Batch prompt", value="", placeholder=None, label_visibility="visible")
                st.sidebar.number_input(f"Batch strength")
                st.sidebar.selectbox(f"Batch model", ["SD", "style2", "style3"])                    

                range_start = st.sidebar.slider('Update From', 0, len(images_list), 1)

                range_end = st.sidebar.slider('Update To', 0, len(images_list), 1)

                if range_start <= range_end:

                    if st.sidebar.button(f'Batch restyle'):

                        for i in range(range_start, range_end):
                            character_pipeline(int(i), project_name, key_settings, timing_details)

                        st.experimental_rerun()

                else:
                        
                        st.sidebar.write("Select a valid range")
            
                for image_name in images_list:
                                
                    image = Image.open(f'videos/{project_name}/assets/frames/0_extracted/{image_name}')            
                    st.subheader(f'Image Name: {image_name}')            
                    image_comparison(
                        img1=f'videos/{project_name}/assets/frames/0_extracted/{image_name}',
                        img2=f'videos/{project_name}/assets/frames/1_character_pipeline_completed/{image_name}')
                    
                    option1, option2,option3,option4 = st.columns(4)

                    with option1:
                        st.text_input(f"Prompt - {image_name}", value="", placeholder=None, label_visibility="visible")

                    with option2:
                        st.number_input(f"Strength - {image_name}")

                    with option3:
                        st.selectbox(f"Model - {image_name}", ["SD", "style2", "style3"])                    

                    with option4:
                        st.text("")
                        st.text("")
                        if st.button(f'Re-run {image_name} styling'):
                            # remove .png from the image name
                            image_name = image_name[:-4]
                            character_pipeline(int(image_name), project_name, key_settings, timing_details)
                            st.experimental_rerun()



        elif st.session_state.stage == "Frame Interpolation":
            st.write("This is the frame interpolation view")
            if st.button("Interpolate Videos"):
                st.write("Interpolate Video")

        elif st.session_state.stage == "Video Speed":
            st.write("This is the video speed view")
            if st.button("Video Speed"):
                st.write("Speeding Videos Video")


        elif st.session_state.stage == "Project Settings":
            st.write("This is the settings view")

        elif st.session_state.stage == "Train Model":

            tab1, tab2 = st.tabs(["Train New Model", "See Existing Models"])

            with tab1:
                images_list = [f for f in os.listdir(f'videos/{project_name}/assets/resources/training_data') if f.endswith('.png')]
        
                images_list.sort(key=lambda f: int(re.sub('\D', '', f)))

                if len(images_list) == 0:
                    st.write("No frames extracted yet")
                
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
                    st.subheader(f'{index_of_current_item}:')                        
                    image = Image.open(f'videos/{project_name}/assets/resources/training_data/{image_name}') 
                    st.image(image, width=400) 
                    yes = st.checkbox(f'Add {index_of_current_item} to model')    

                    if yes:
                        images_for_model.append(index_of_current_item)
                    else:
                        if index_of_current_item in images_for_model:
                            images_for_model.remove(index_of_current_item)
                
                st.sidebar.subheader("Images for model")

                st.sidebar.write(f"You've selected {len(images_for_model)} image.")

                if len(images_for_model) < 7:
                    st.sidebar.write("Select at least 7 images for model training")
                    st.sidebar.button("Train Model",disabled=True)

                else:
                    st.sidebar.button("Train Model",disabled=False)
            
            with tab2:
                st.write("This is the tab 2")

                  

if __name__ == '__main__':
    main()

