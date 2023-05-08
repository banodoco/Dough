import streamlit as st
from ui_components.common_methods import create_gif_preview, delete_frame, get_model_details, get_timing_details, promote_image_variant, trigger_restyling_process,add_image_variant,prompt_interpolation_model,update_speed_of_video_clip,create_timings_row_at_frame_number,extract_canny_lines,get_duration_from_video,get_audio_bytes_for_slice,add_audio_to_video_slice,convert_to_minutes_and_seconds,styling_sidebar,get_primary_variant_location,create_full_preview_video,back_and_forward_buttons,create_single_preview_video,resize_and_rotate_element,rotate_image,zoom_image
from repository.local_repo.csv_repo import get_app_settings, get_project_settings,update_specific_timing_value,update_project_setting
import uuid

project_name = "humanist_attenborough_1"
timing_details = get_timing_details(project_name)

rotation = 0
zoom = 1.0

for i in timing_details:
                                            
    index_of_current_item = timing_details.index(i)        
    if timing_details[index_of_current_item]["notes"] == "":
        current_location = timing_details[index_of_current_item]["source_image"]
        rotated_image = rotate_image(current_location,rotation)
        rotated_image.save("temp.png")        
        zoomed_image = zoom_image("temp.png", zoom)
        file_name = "temp/" + str(uuid.uuid4()) + ".png"  
        zoomed_image.save(file_name)                 
        update_specific_timing_value(project_name,index_of_current_item, "source_image", file_name)
        update_specific_timing_value(project_name,index_of_current_item, "notes", "rotation: " + str(rotation) + "| zoom: " + str(zoom))
        print("Index: " + str(index_of_current_item))
        print("Rotation: " + str(rotation))
        print("Zoom: " + str(zoom))        
    rotation = rotation + 0.5
    zoom = zoom + 0.01  
            
        





