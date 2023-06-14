from pathlib import Path
import os
import csv

# creates a file path if it's not already present
def create_file_path(path):
    if not path:
        return
    
    file = Path(path)
    if not file.is_file():
        last_slash_index = path.rfind('/')
        if last_slash_index != -1:
                directory_path = path[:last_slash_index]
                file_name = path[last_slash_index + 1:]
                
                # creating directory if not present
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
        else:
            directory_path = './'
            file_name = path

        # creating file
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'w') as f:
            pass
        
        # adding columns/rows in the file
        if file_name == 'timings.csv':
            data = [
                ['frame_time', 'frame_number', 'primary_image', 'alternative_images', 'custom_pipeline', 'negative_prompt', 'guidance_scale', 'seed', 'num_inference_steps',
                      'model_id', 'strength', 'notes', 'source_image', 'custom_models', 'adapter_type', 'clip_duration', 'interpolated_video', 'timed_clip', 'prompt', 'mask'],
            ]
        elif file_name == 'settings.csv':
            data = [
                ['key', 'value'],
                ['last_prompt', ''],
                ['default_model', 'controlnet'],
                ['last_strength', '0.5'],
                ['last_custom_pipeline', 'None'],
                ['audio', ''],
                ['input_type', 'Video'],
                ['input_video', ''],
                ['extraction_type', 'Regular intervals'],
                ['width', '704'],
                ['height', '512'],
                ['last_negative_prompt', '"nudity,  boobs, breasts, naked, nsfw"'],
                ['last_guidance_scale', '7.5'],
                ['last_seed', '0'],
                ['last_num_inference_steps', '100'],
                ['last_which_stage_to_run_on', 'Current Main Variants'],
                ['last_custom_models', '[]'],
                ['last_adapter_type', 'normal']
            ]
        elif file_name == 'app_settings.csv':
            data = [
                ['key', 'value'],
                ['replicate_com_api_key', ''],
                ['aws_access_key_id', ''],
                ['aws_secret_access_key', ''],
                ['previous_project', ''],
                ['replicate_username', ''],
                ['welcome_state', '0']
            ]
        elif file_name == 'log.csv':
            data = [
                ['model_name', 'model_version', 'total_inference_time', 'input_params', 'created_on'],
            ]

        
        if len(data):
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(data)

def copy_sample_assets(project_name):
    import shutil

    # copy sample video
    source = "sample_assets/input_videos/sample.mp4"
    dest = "videos/" + project_name + "/assets/resources/input_videos/sample.mp4"
    shutil.copyfile(source, dest)

    # copy selected frames
    select_samples_path = 'sample_assets/frames/selected_sample'
    file_list = os.listdir(select_samples_path)
    file_paths = []
    for item in file_list:
        item_path = os.path.join(select_samples_path, item)
        if os.path.isfile(item_path):
            file_paths.append(item_path)
    
    for idx in range(len(file_list)):
        source = file_paths[idx]
        dest = f"videos/{project_name}/assets/frames/1_selected/{file_list[idx]}"
        shutil.copyfile(source, dest)
    
    # copy timings file
    source = "sample_assets/frames/meta_data/timings.csv"
    dest = f"videos/{project_name}/timings.csv"
    shutil.copyfile(source, dest)

def create_working_assets(project_name):
    new_project = True
    if os.path.exists("videos/"+project_name):
        new_project = False

    directory_list = [
        # project specific files
        "videos/" + project_name,
        "videos/" + project_name + "/assets",
        "videos/" + project_name + "/assets/frames",
        "videos/" + project_name + "/assets/frames/0_extracted",
        "videos/" + project_name + "/assets/frames/1_selected",
        "videos/" + project_name + "/assets/frames/2_character_pipeline_completed",
        "videos/" + project_name + "/assets/frames/3_backdrop_pipeline_completed",
        "videos/" + project_name + "/assets/resources",
        "videos/" + project_name + "/assets/resources/backgrounds",
        "videos/" + project_name + "/assets/resources/masks",
        "videos/" + project_name + "/assets/resources/audio",
        "videos/" + project_name + "/assets/resources/input_videos",
        "videos/" + project_name + "/assets/resources/prompt_images",
        "videos/" + project_name + "/assets/videos",
        "videos/" + project_name + "/assets/videos/0_raw",
        "videos/" + project_name + "/assets/videos/1_final",
        "videos/" + project_name + "/assets/videos/2_completed",
        # app data
        "inference_log",
        # temp folder
        "videos/temp",
        "videos/temp/assets/videos/0_raw/"
    ]
    
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # copying sample assets for new project
    if new_project:
        copy_sample_assets(project_name)

    csv_file_list = [
        f'videos/{project_name}/settings.csv',
        f'videos/{project_name}/timings.csv',
        'inference_log/log.csv'
    ]

    for csv_file in csv_file_list:
        create_file_path(csv_file)

