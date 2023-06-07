import shutil
import streamlit as st
import os
import time

from ui_components.common_methods import attach_audio_element
from utils.data_repo.data_repo import DataRepo


def project_settings_page(project_uuid):
    data_repo = DataRepo()

    project = data_repo.get_project_from_uuid(project_uuid)
    project_settings = data_repo.get_project_setting(project_uuid)
    # make a list of all the files in videos/{project_name}/assets/resources/music

    project_name = project.name
    attach_audio_element(project_uuid, False)

    with st.expander("Version History"):
        version_name = st.text_input(
            "What would you liket to call this version?", key="version_name")
        version_name = version_name.replace(" ", "_")

        if st.button("Make a copy of this project", key="copy_project"):
            # shutil.copyfile(f"videos/{project_name}/timings.csv", f"videos/{project_name}/timings_{version_name}.csv")
            data_repo.create_backup(project_uuid, version_name)
            st.success("Project copied successfully!")

        # list all the .csv files in that folder starting with timings_
        # version_list = [list_of_files for list_of_files in os.listdir(
        #     "videos/" + project_name) if list_of_files.startswith('timings_')]
        version_list = data_repo.get_backup_list(project_uuid)

        header1, header2, header3 = st.columns([1, 1, 1])

        with header1:
            st.markdown("### Version Name")
        with header2:
            st.markdown("### Created On")
        with header3:
            st.markdown("### Restore Version")

        for backup in version_list:
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.write(backup.name)
            with col2:
                st.write(
                    # f"{time.ctime(os.path.getmtime(f'videos/{project_name}/{i}'))}")
                    f"{time.ctime(backup.created_on)}")
            with col3:
                if st.button("Restore this version", key=f"restore_version_{backup.name}"):
                    # change timings.csv to last_timings.csv
                    # os.rename(f"videos/{project_name}/timings.csv",
                    #           f"videos/{project_name}/timings_previous.csv")
                    # rename i to timings.csv
                    # make this copy the file instead using shutil os.rename(f"videos/{project_name}/{i}", f"videos/{project_name}/timings.csv")
                    # shutil.copyfile(
                    #     f"videos/{project_name}/{i}", f"videos/{project_name}/timings.csv")
                    restore_backup(backup.uuid)
                    st.success(
                        "Version restored successfully! Just in case, the previous version has been saved as last_timings.csv")
                    time.sleep(2)
                    st.experimental_rerun()

    with st.expander("Frame Size"):
        st.write("Current Size = ",
                 project_settings["width"], "x", project_settings["height"])
        width = st.selectbox("Select video width", options=[
                             "512", "683", "704", "1024"], key="video_width")
        height = st.selectbox("Select video height", options=[
                              "512", "704", "1024"], key="video_height")
        if st.button("Save"):
            data_repo.update_project_setting(project_uuid, width=width)
            data_repo.update_project_setting(project_uuid, height=height)
            st.experimental_rerun()
