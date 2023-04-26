import streamlit as st
import time
from repository.local_repo.csv_repo import get_app_settings, update_specific_timing_value
from ui_components.common_methods import calculate_desired_duration_of_each_clip, get_timing_details,create_individual_clip

def frame_interpolation_page(mainheader2, project_name):
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
                        update_specific_timing_value(project_name, index_of_current_item, "interpolation_steps", interpolation_steps)
                        timing_details = get_timing_details(project_name)
                    else:
                        update_specific_timing_value(project_name, index_of_current_item, "interpolation_steps", interpolation_steps)
                        timing_details = get_timing_details(project_name)

                    if timing_details[index_of_current_item]["interpolated_video"] == "":                                                        

                        if total_number_of_videos == index_of_current_item:
                                                        
                            video_location = create_individual_clip(index_of_current_item, project_name)
                            update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", video_location)

                        else:
                            

                            video_location =  create_individual_clip(index_of_current_item, project_name)
                
                            update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", video_location)
                            
                            
                st.success("All videos interpolated!")

        else:
            specific_video = st.number_input("Which video to interpolate", min_value=0, max_value=total_number_of_videos, value=0)

            if st.button("Interpolate this video"):

                if dynamic_interolation_steps == "Dynamic":
                    calculate_desired_duration_of_each_clip(timing_details,project_name)                    
                    interpolation_steps = calculate_dynamic_interpolations_steps(timing_details[specific_video]["duration_of_clip"])                            
                    update_specific_timing_value(project_name, index_of_current_item, "interpolation_steps", interpolation_steps)
                    timing_details = get_timing_details(project_name)                               

                video_location = create_individual_clip(index_of_current_item, project_name)
                
                update_specific_timing_value(project_name, index_of_current_item, "interpolated_video", video_location)
    


