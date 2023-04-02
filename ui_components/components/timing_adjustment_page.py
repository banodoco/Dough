import os
import re
import streamlit as st
import random
import time

from ui_components.common_methods import delete_frame, find_duration_of_clip, get_timing_details, render_video, update_specific_timing_value, update_video_speed

def timing_adjustment_page(project_name):
    timing_details = get_timing_details(project_name)
    total_number_of_videos = len(timing_details) - 1
    
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
                    