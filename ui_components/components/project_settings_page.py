import shutil
import streamlit as st
import os
import time

from repository.local_repo.csv_repo import get_project_settings, update_project_setting
from ui_components.common_methods import attach_audio_element


def project_settings_page(project_name):
    project_settings = get_project_settings(project_name)
    # make a list of all the files in videos/{project_name}/assets/resources/music
    
    attach_audio_element(project_name, project_settings,False)
    
    with st.expander("Version History"):
        

        version_name = st.text_input("What would you liket to call this version?", key="version_name")
        version_name = version_name.replace(" ", "_")

        if st.button("Make a copy of this project", key="copy_project"):                    
            shutil.copyfile(f"videos/{project_name}/timings.csv", f"videos/{project_name}/timings_{version_name}.csv")
            st.success("Project copied successfully!")             

        # list all the .csv files in that folder starting with timings_

        version_list = [list_of_files for list_of_files in os.listdir(
            "videos/" + project_name) if list_of_files.startswith('timings_')]
        
        header1, header2, header3 = st.columns([1,1,1])

        with header1:
            st.markdown("### Version Name")
        with header2:
            st.markdown("### Created On")
        with header3:
            st.markdown("### Restore Version")

        for i in version_list:
            col1, col2, col3 = st.columns([1,1,1])

            with col1:
                st.write(i)
            with col2:
                st.write(f"{time.ctime(os.path.getmtime(f'videos/{project_name}/{i}'))}")
            with col3:
                if st.button("Restore this version", key=f"restore_version_{i}"):
                    # change timings.csv to last_timings.csv
                    os.rename(f"videos/{project_name}/timings.csv", f"videos/{project_name}/timings_previous.csv")
                    # rename i to timings.csv
                    os.rename(f"videos/{project_name}/{i}", f"videos/{project_name}/timings.csv")
                    st.success("Version restored successfully! Just in case, the previous version has been saved as last_timings.csv")
                    time.sleep(2)
                    st.experimental_rerun()
    
    with st.expander("Frame Size"):
        st.write("Current Size = ", project_settings["width"], "x", project_settings["height"])
        width = st.selectbox("Select video width", options=["512","704","1024"], key="video_width")
        height = st.selectbox("Select video height", options=["512","704","1024"], key="video_height")
        if st.button("Save"):
            update_project_setting("width", width, project_name)
            update_project_setting("height", height, project_name)
            st.experimental_rerun()