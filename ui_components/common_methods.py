import ast
import os
import random
import string
import pandas as pd
import cv2

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

def get_timing_details(video_name):

    file_path = "videos/" + str(video_name) + "/timings.csv"
    df = pd.read_csv(file_path, na_filter=False)

    # Evaluate the alternative_images column and replace it with the evaluated list
    df['alternative_images'] = df['alternative_images'].fillna('').apply(lambda x: ast.literal_eval(x[1:-1]) if x != '' else '')
    return df.to_dict('records')

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

    df = pd.DataFrame(columns=['frame_time','frame_number','primary_image','alternative_images','custom_pipeline','negative_prompt','guidance_scale','seed','num_inference_steps','model_id','strength','notes','source_image','custom_models','adapter_type','duration_of_clip','interpolated_video','timing_video','prompt'])

    # df.loc[0] = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    df.to_csv(f'videos/{video_name}/timings.csv', index=False)

