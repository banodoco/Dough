from typing import List
import datetime
import streamlit as st
from shared.constants import InternalFileTag, InternalFileType
from ui_components.common_methods import render_video
import random
import time
import os
import re
from ui_components.models import InternalFileObject, InternalFrameTimingObject
from ui_components.widgets.attach_audio_element import attach_audio_element

from utils.data_repo.data_repo import DataRepo


def video_rendering_page(mainheader2, project_uuid):
    data_repo = DataRepo()
    project = data_repo.get_project_from_uuid(project_uuid)
    project_name = project.name

    with mainheader2:
        with st.expander("💡 How video rendering works"):
            st.info("This is simply pulling together the interpolated frames to deliver the final video. You can edit the timing if need be in in Tools > Timing Adjustment")

    timing_details: List[InternalFrameTimingObject] = data_repo.get_timing_list_from_project(
        project_uuid)
    project_settings = data_repo.get_project_setting(project_uuid)

    parody_movie_names = ["The_Lord_of_the_Onion_Rings", "Jurassic_Pork", "Harry_Potter_and_the_Sorcerer_s_Kidney_Stone", "Star_Wars_The_Phantom_of_the_Oprah", "The_Silence_of_the_Yams", "The_Hunger_Pains", "Free_Willy_Wonka_and_the_Chocolate_Factory", "The_Da_Vinci_Chode", "Forrest_Dump", "The_Shawshank_Inebriation", "A_Clockwork_Orange_Juice", "The_Big_Lebowski_2_Dude_Where_s_My_Car", "The_Princess_Diaries_The_Dark_Knight_Rises", "Eternal_Sunshine_of_the_Spotless_Behind", "Rebel_Without_a_Clue", "The_Terminal_Dentist", "Dr_Strangelove_or_How_I_Learned_to_Stop_Worrying_and_Love_the_Bombastic", "The_Wolf_of_Sesame_Street", "The_Good_the_Bad_and_the_Fluffy", "The_Sound_of_Mucus", "Back_to_the_Fuchsia", "The_Curious_Case_of_Benjamin_s_Button", "The_Fellowship_of_the_Bing", "The_Texas_Chainsaw_Manicure",  "The_Iron_Manatee", "Night_of_the_Living_Bread", "Indiana_Jones_and_the_Temple_of_Groom", "Kill_Billiards", "The_Bourne_Redundancy", "The_SpongeBob_SquarePants_Movie_Sponge_Out_of_Water_and_Ideas",
                          "Planet_of_the_Snapes", "No_Country_for_Old_Yentas", "The_Expendable_Accountant", "The_Terminal_Illness", "A_Streetcar_Named_Retire", "The_Secret_Life_of_Walter_s_Mitty", "The_Hunger_Games_Catching_Foam", "The_Godfather_Part_Time_Job", "How_To_Kill_a_Mockingbird", "Star_Trek_III_The_Search_for_Spock_s_Missing_Sock", "Gone_with_the_Wind_Chimes", "Dr_No_Clue", "Ferris_Bueller_s_Day_Off_Sick", "Monty_Python_and_the_Holy_Fail", "A_Fistful_of_Quarters", "Willy_Wonka_and_the_Chocolate_Heartburn", "The_Good_the_Bad_and_the_Dandruff", "The_Princess_Bride_of_Frankenstein", "The_Wizard_of_Bras", "Pulp_Friction", "Die_Hard_with_a_Clipboard", "Indiana_Jones_and_the_Last_Audit", "Finding_Nemoy", "The_Silence_of_the_Lambs_The_Musical", "Titanic_2_The_Iceberg_Strikes_Back", "Fast_Times_at_Ridgemont_Mortuary", "The_Graduate_But_Only_Because_He_Has_an_Advanced_Degree", "Beauty_and_the_Yeast", "The_Blair_Witch_Takes_Manhattan", "Reservoir_Bitches", "Die_Hard_with_a_Pension"]
    random_name = random.choice(parody_movie_names)

    final_video_name = st.text_input(
        "What would you like to name this video?", value=random_name)

    attach_audio_element(project_uuid, False)

    delete_existing_timed_clips = st.checkbox(
        "Delete all the existing timing clips.", value=False)

    quality1, quality2 = st.columns([1, 2])

    with quality1:
        quality_of_video = st.radio("What quality would you like?", [
                                    'Preview', 'High-Quality'], horizontal=True)

    with quality2:
        if quality_of_video == "Preview":
            st.info("THIS MEANS")
        else:
            st.info("THIS MEANS")

    if st.button("Render New Video"):
        if delete_existing_timed_clips == True:
            for i in timing_details:
                data_repo.update_specific_timing(timing_details[i].uuid, timed_clip_id=None)
            timing_details = data_repo.get_timing_list_from_project(project_uuid)

        render_video(final_video_name, project_uuid, quality_of_video)
        st.success("Video rendered!")
        time.sleep(1.5)
        st.experimental_rerun()

    st.markdown("***")

    # video_list = [list_of_files for list_of_files in os.listdir(
    #     "videos/" + project_name + "/assets/videos/2_completed") if list_of_files.endswith('.mp4')]

    # TODO: only show completed videos
    video_list: List[InternalFileObject] = data_repo.get_all_file_list(InternalFileType.VIDEO.value, tag=InternalFileTag.GENERATED_VIDEO.value, project_id=project_uuid)
    video_list = sorted(video_list, key=lambda x: x.created_on, reverse=True)
    # video_dir = "videos/" + project_name + "/assets/videos/2_completed"

    # video_list.sort(key=lambda f: int(re.sub('\D', '', f)))

    # video_list = sorted(video_list, key=lambda x: os.path.getmtime(
    #     os.path.join(video_dir, x)), reverse=True)
    
    for video in video_list:
        st.subheader(video.name)

        try:
            st.write(datetime.datetime.fromisoformat(video.created_on))
        except Exception as e:
            st.write(datetime.datetime.strptime(video.created_on, '%Y-%m-%dT%H:%M:%S.%fZ'))

        st.video(video.location)

        col1, col2 = st.columns(2)

        with col1:
            if st.checkbox(f"Confirm {video.name} Deletion"):
                if st.button(f"Delete {video.name}"):
                    # removing locally
                    video_path = "videos/" + project_uuid + "/assets/videos/2_completed/" + video.name
                    if os.path.exists(video_path):
                        os.remove(video_path)

                    # removing from database
                    data_repo.delete_file_from_uuid(video.uuid)

                    st.experimental_rerun()
            else:
                st.button(f"Delete {video}", disabled=True)
