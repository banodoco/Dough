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
                      'model_id', 'strength', 'notes', 'source_image', 'custom_models', 'adapter_type', 'duration_of_clip', 'interpolated_video', 'timing_video', 'prompt', 'mask'],
            ]
        elif file_name == 'settings.csv':
            data = [
                ['key', 'value'],
                ['last_prompt', ''],
                ['last_model', 'controlnet'],
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
                ['replicate_user_name', ''],
                ['welcome_state', '0']
            ]

        
        if len(data):
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(data)


def create_working_assets(video_name):
    directory_list = [
        "videos/" + video_name,
        "videos/" + video_name + "/assets",
        "videos/" + video_name + "/assets/frames",
        "videos/" + video_name + "/assets/frames/0_extracted",
        "videos/" + video_name + "/assets/frames/1_selected",
        "videos/" + video_name + "/assets/frames/2_character_pipeline_completed",
        "videos/" + video_name + "/assets/frames/3_backdrop_pipeline_completed",
        "videos/" + video_name + "/assets/resources",
        "videos/" + video_name + "/assets/resources/backgrounds",
        "videos/" + video_name + "/assets/resources/masks",
        "videos/" + video_name + "/assets/resources/audio",
        "videos/" + video_name + "/assets/resources/input_videos",
        "videos/" + video_name + "/assets/resources/prompt_images",
        "videos/" + video_name + "/assets/videos",
        "videos/" + video_name + "/assets/videos/0_raw",
        "videos/" + video_name + "/assets/videos/1_final",
        "videos/" + video_name + "/assets/videos/2_completed"
    ]
    
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    csv_file_list = [
        f'videos/{video_name}/settings.csv',
        f'videos/{video_name}/timings.csv'
    ]

    for csv_file in csv_file_list:
        create_file_path(csv_file)

