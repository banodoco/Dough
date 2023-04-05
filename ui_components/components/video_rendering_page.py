import streamlit as st
from ui_components.common_methods import attach_audio_element, get_timing_details, render_video
from repository.local_repo.csv_repo import get_project_settings, update_specific_timing_value
import random
import time
import os
import re

def video_rendering_page(mainheader2, project_name):
    with mainheader2:
        with st.expander("💡 How video rendering works"):
            st.info("This is simply pulling together the interpolated frames to deliver the final video. You can edit the timing if need be in in Tools > Timing Adjustment")

    timing_details = get_timing_details(project_name)
    project_settings = get_project_settings(project_name)

    disable_rendering = False
    for i in timing_details:
        if i["interpolated_video"] == "" and timing_details.index(i) != len(timing_details)-1 and disable_rendering == False:                    
            st.error("You need to interpolate all the videos before you can render the final video. If you delete frames or change the primary image, you will need to interpolate the video again.")
            disable_rendering = True
    parody_movie_names = ["The_Lord_of_the_Onion_Rings", "Jurassic_Pork", "Harry_Potter_and_the_Sorcerer_s_Kidney_Stone", "Star_Wars_The_Phantom_of_the_Oprah", "The_Silence_of_the_Yams", "The_Hunger_Pains", "Free_Willy_Wonka_and_the_Chocolate_Factory", "The_Da_Vinci_Chode", "Forrest_Dump", "The_Shawshank_Inebriation", "A_Clockwork_Orange_Juice", "The_Big_Lebowski_2_Dude_Where_s_My_Car", "The_Princess_Diaries_The_Dark_Knight_Rises", "Eternal_Sunshine_of_the_Spotless_Behind", "Rebel_Without_a_Clue", "The_Terminal_Dentist", "Dr_Strangelove_or_How_I_Learned_to_Stop_Worrying_and_Love_the_Bombastic", "The_Wolf_of_Sesame_Street", "The_Good_the_Bad_and_the_Fluffy", "The_Sound_of_Mucus", "Back_to_the_Fuchsia", "The_Curious_Case_of_Benjamin_s_Button", "The_Fellowship_of_the_Bing", "The_Texas_Chainsaw_Manicure",  "The_Iron_Manatee", "Night_of_the_Living_Bread", "Indiana_Jones_and_the_Temple_of_Groom", "Kill_Billiards", "The_Bourne_Redundancy", "The_SpongeBob_SquarePants_Movie_Sponge_Out_of_Water_and_Ideas","Planet_of_the_Snapes", "No_Country_for_Old_Yentas", "The_Expendable_Accountant", "The_Terminal_Illness", "A_Streetcar_Named_Retire", "The_Secret_Life_of_Walter_s_Mitty", "The_Hunger_Games_Catching_Foam", "The_Godfather_Part_Time_Job", "How_To_Kill_a_Mockingbird", "Star_Trek_III_The_Search_for_Spock_s_Missing_Sock", "Gone_with_the_Wind_Chimes", "Dr_No_Clue", "Ferris_Bueller_s_Day_Off_Sick", "Monty_Python_and_the_Holy_Fail", "A_Fistful_of_Quarters", "Willy_Wonka_and_the_Chocolate_Heartburn", "The_Good_the_Bad_and_the_Dandruff", "The_Princess_Bride_of_Frankenstein", "The_Wizard_of_Bras", "Pulp_Friction", "Die_Hard_with_a_Clipboard", "Indiana_Jones_and_the_Last_Audit", "Finding_Nemoy", "The_Silence_of_the_Lambs_The_Musical", "Titanic_2_The_Iceberg_Strikes_Back", "Fast_Times_at_Ridgemont_Mortuary", "The_Graduate_But_Only_Because_He_Has_an_Advanced_Degree", "Beauty_and_the_Yeast","The_Blair_Witch_Takes_Manhattan","Reservoir_Bitches","Die_Hard_with_a_Pension"]
    
    random_name = random.choice(parody_movie_names)

    final_video_name = st.text_input("What would you like to name this video?",value=random_name)

    attach_audio_element(project_name, project_settings,False)

    delete_existing_videos = st.checkbox("Delete all the existing timings", value=False)

    if st.button("Render New Video",disabled=disable_rendering):
        if delete_existing_videos == True:
            for i in timing_details:   
                index_of_current_item = timing_details.index(i)                                                             
                update_specific_timing_value(project_name, timing_details.index(i), "timing_video", "")
            timing_details = get_timing_details(project_name)
        
        render_video(project_name, final_video_name)
        st.success("Video rendered!")
        time.sleep(1.5)
        st.experimental_rerun()            
    

    video_list = [list_of_files for list_of_files in os.listdir(
        "videos/" + project_name + "/assets/videos/2_completed") if list_of_files.endswith('.mp4')]            

    video_dir = "videos/" + project_name + "/assets/videos/2_completed"

    video_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    video_list = sorted(video_list, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)                        
    import datetime
    for video in video_list:

        st.subheader(video)       

        st.write(datetime.datetime.fromtimestamp(
            os.path.getmtime("videos/" + project_name + "/assets/videos/2_completed/" + video)))

        st.video(f"videos/{project_name}/assets/videos/2_completed/{video}")
        
        col1, col2 = st.columns(2)

        with col1:

            if st.checkbox(f"Confirm {video} Deletion"):

                if st.button(f"Delete {video}"):
                    os.remove("videos/" + project_name +
                            "/assets/videos/2_completed/" + video)
                    st.experimental_rerun()
            else:
                st.button(f"Delete {video}",disabled=True)

                
