import streamlit as st
import time
from repository.local_repo.csv_repo import remove_existing_timing, update_specific_timing_value
from ui_components.common_methods import add_image_variant, get_timing_details, promote_image_variant

def batch_action_page(project_name):
    timing_details = get_timing_details(project_name)

    st.markdown("***")

    st.markdown("#### Make extracted key frames into completed key frames")
    st.write("This will move all the extracted key frames to completed key frames - good for if you don't want to make any changes to the key frames")
    if st.button("Move initial key frames to completed key frames"):
        for i in timing_details:
            index_of_current_item = timing_details.index(i)
            add_image_variant(timing_details[index_of_current_item]["source_image"], index_of_current_item, project_name, timing_details)
            promote_image_variant(index_of_current_item, project_name, 0)
        st.success("All initial key frames moved to completed key frames")

    st.markdown("***")
    
    st.markdown("#### Remove all existing timings")
    st.write("This will remove all the timings and key frames from the project")
    if st.button("Remove Existing Timings"):
        remove_existing_timing(project_name)

    st.markdown("***")
    
    st.markdown("#### Bulk adjust the timings")
    st.write("This will adjust the timings of all the key frames by the number of seconds you enter below")
    bulk_adjustment = st.number_input("What multiple would you like to adjust the timings by?", value=1.0)
    if st.button("Adjust Timings"):
        for i in timing_details:
            index_of_current_item = timing_details.index(i)
            new_frame_time = float(timing_details[index_of_current_item]["frame_time"]) * bulk_adjustment
            update_specific_timing_value(project_name, index_of_current_item, "frame_time", new_frame_time)
            
        st.success("Timings adjusted successfully!")
        time.sleep(1)
        st.experimental_rerun()