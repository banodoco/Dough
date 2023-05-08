import streamlit as st
#from ui_components.common_methods import get_timing_details
from ui_components.common_methods import get_timing_details
import uuid
import os
from repository.local_repo.csv_repo import CSVProcessor, get_app_settings, get_project_settings, update_project_setting, update_specific_timing_value
import shutil

project_name = "humanist_attenborough_1"
timing_details = get_timing_details(project_name)

images = []

# add all the images in the /eyes folder to the images list

for image in os.listdir("eyes"):
    if image.endswith(".png"):
        images.append(image)

print(images)


for i in timing_details:
    index_of_current_item = timing_details.index(i)
    if timing_details[index_of_current_item]["source_image"] == "":
        image_to_use = images[0]
        # use shutil to move the image from the /eyes folder to the /temp folder
        shutil.move(f"eyes/{image_to_use}", "temp")
        new_image_path = f"temp/{image_to_use}"
        update_specific_timing_value(project_name,index_of_current_item, "source_image", new_image_path)
        # remove the image from the images list
        images.remove(image_to_use)
        

