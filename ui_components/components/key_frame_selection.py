import streamlit as st
import cv2
import os
import time
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from repository.local_repo.csv_repo import get_project_settings, remove_existing_timing, update_project_setting, update_specific_timing_value

from ui_components.common_methods import calculate_frame_number_at_time, create_timings_row_at_frame_number, create_video_without_interpolation, delete_frame, extract_frame, get_timing_details, preview_frame
from utils.media_processor.video import resize_video


def key_frame_selection_page(mainheader2, project_name):
    with mainheader2:
        with st.expander("💡 How key frame selection works"):
            st.info("Key Frame Selection is a process that allows you to select the frames that you want to style. These Key Frames act as the anchor points for your animations. On the left, you can bulk select these, while on the right, you can refine your choices, or manually select them.")
    timing_details = get_timing_details(project_name)
    project_settings = get_project_settings(project_name)

    st.sidebar.subheader("Upload new videos")
    st.sidebar.write(
        "Open the toggle below to upload and select new inputs video to use for this project.")

    if project_settings["input_video"] == "":
        st.sidebar.warning(
            "No input video selected - please select one below.")
    if project_settings["input_video"] != "":
        st.sidebar.success("Input video selected - you can change this below.")

    with st.sidebar.expander("Select input video", expanded=False):
        directory_path = f'videos/{project_name}/assets/resources/input_videos'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        input_video_list = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.mov', '.MOV', '.avi'))]
        if project_settings["input_video"] != "":
            input_video_index = input_video_list.index(
                project_settings["input_video"])
            input_video = st.selectbox(
                "Input video:", input_video_list, index=input_video_index)
            input_video_cv2 = cv2.VideoCapture(
                f'videos/{project_name}/assets/resources/input_videos/{input_video}')
            total_frames = input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = input_video_cv2.get(cv2.CAP_PROP_FPS)
            # duration to 2 decimal places
            duration = round(total_frames / fps, 2)

            preview1, preview2, preview3 = st.columns([1, 1, 1])
            with preview1:
                st.image(preview_frame(project_name,
                         input_video, total_frames * 0.25))
            with preview2:
                st.image(preview_frame(project_name,
                         input_video, total_frames * 0.5))
            with preview3:
                st.image(preview_frame(project_name,
                         input_video, total_frames * 0.75))
            st.caption(
                f"This video is {duration} seconds long, and has {total_frames} frames.")
            # st.video(f'videos/{project_name}/assets/resources/input_videos/{input_video}')
        else:
            input_video = st.selectbox("Input video:", input_video_list)

        if st.button("Update Video"):
            update_project_setting("input_video", input_video, project_name)
            st.experimental_rerun()
        st.markdown("***")
        st.subheader("Upload new video")
        width = int(project_settings["width"])
        height = int(project_settings["height"])

        uploaded_file = st.file_uploader("Choose a file")
        keep_audio = st.checkbox("Keep audio from original video.")
        resize_this_video = st.checkbox(
            "Resize video to match project settings: " + str(width) + "px x " + str(height) + "px", value=True)

        if st.button("Upload new video"):
            video_path = f'videos/{project_name}/assets/resources/input_videos/{uploaded_file.name}'
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            width = int(project_settings["width"])
            height = int(project_settings["height"])
            if resize_this_video == True:
                resize_video(input_path=video_path,
                             output_path=video_path, width=width, height=height)
            st.success("Video uploaded successfully")
            if keep_audio == True:
                clip = VideoFileClip(
                    f'videos/{project_name}/assets/resources/input_videos/{uploaded_file.name}')
                clip.audio.write_audiofile(
                    f'videos/{project_name}/assets/resources/audio/extracted_audio.mp3')
                update_project_setting(
                    "audio", "extracted_audio.mp3", project_name)
            update_project_setting("input_video", input_video, project_name)
            project_settings = get_project_settings(project_name)
            time.sleep(1)
            st.experimental_rerun()

    st.sidebar.subheader("Bulk extract key frames from video")
    with st.sidebar.expander("💡 Learn about bulk extraction vs manual selection", expanded=False):
        st.info("You can use either of the options below to extract key frames from your video in bulk, or you can use the manual key on the bottom right. If need be, you can also refine your key frame selection after bulk extraction using by clicking into the single frame view.")
    types_of_extraction = ["Regular intervals", "Extract from csv"]

    type_of_extraction = st.sidebar.radio(
        "Choose type of key frame extraction", types_of_extraction)
    input_video_cv2 = cv2.VideoCapture(
        f'videos/{project_name}/assets/resources/input_videos/{input_video}')
    total_frames = input_video_cv2.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = input_video_cv2.get(cv2.CAP_PROP_FPS)
    st.sidebar.caption(
        f"This video is {total_frames} frames long and has a framerate of {fps} fps.")

    if type_of_extraction == "Regular intervals":
        frequency_of_extraction = st.sidebar.slider("How frequently would you like to extract frames?", min_value=1, max_value=120, step=1,
                                                    value=10, help=f"This will extract frames at regular intervals. For example, if you choose 15 it'll extract every 15th frame.")
        if st.sidebar.checkbox("I understand that running this will remove all existing frames and styling."):
            if st.sidebar.button("Extract frames"):
                update_project_setting(
                    "extraction_type", "Regular intervals", project_name)
                update_project_setting(
                    "input_video", input_video, project_name)
                number_of_extractions = int(
                    total_frames/frequency_of_extraction)
                remove_existing_timing(project_name)

                for i in range(0, number_of_extractions):
                    timing_details = get_timing_details(project_name)
                    extract_frame_number = i * frequency_of_extraction
                    last_index = len(timing_details)
                    create_timings_row_at_frame_number(
                        project_name, input_video, extract_frame_number, timing_details, last_index)
                    timing_details = get_timing_details(project_name)
                    extract_frame(i, project_name, input_video,
                                  extract_frame_number, timing_details)
                st.experimental_rerun()
        else:
            st.sidebar.button("Extract frames", disabled=True)
    elif type_of_extraction == "Extract manually":
        st.sidebar.info(
            "On the right, you'll see a toggle to choose which frames to extract. You can also use the slider to choose the granularity of the frames you want to extract.")
    elif type_of_extraction == "Extract from csv":
        st.sidebar.subheader(
            "Re-extract key frames using existing timings file")
        st.sidebar.write(
            "This will re-extract all frames based on the timings file. This is useful if you've changed the granularity of your key frames manually.")
        if st.sidebar.checkbox("I understand that running this will remove every existing frame"):
            if st.sidebar.button("Re-extract frames"):
                update_project_setting(
                    "extraction_type", "Extract from csv", project_name)
                update_project_setting(
                    "input_video", input_video, project_name)
                get_timing_details(project_name)
                for i in timing_details:
                    index_of_current_item = timing_details.index(i)
                    extract_frame_number = calculate_frame_number_at_time(
                        input_video, timing_details[index_of_current_item]["frame_time"], project_name)
                    extract_frame(index_of_current_item, project_name,
                                  input_video, extract_frame_number, timing_details)

        else:
            st.sidebar.button("Re-extract frames", disabled=True)

    if len(timing_details) == 0:
        st.info("Once you've added key frames, they'll appear here.")
    else:
        # which_image_value is the current keyframe number
        if "which_image_value" not in st.session_state:
            st.session_state['which_image_value'] = 0

        timing_details = get_timing_details(project_name)

        # 0 -> list view, 1 -> single view
        if 'key_frame_view_type_index' not in st.session_state:
            st.session_state['key_frame_view_type_index'] = 0

        view_types = ["List View", "Single Frame"]

        st.session_state['key_frame_view_type'] = st.radio(
            "View type:", view_types, key="which_view_type", horizontal=True, index=st.session_state['key_frame_view_type_index'])

        # if displayed_view != selected_view then rerun()
        if view_types.index(st.session_state['key_frame_view_type']) != st.session_state['key_frame_view_type_index']:
            st.session_state['key_frame_view_type_index'] = view_types.index(
                st.session_state['key_frame_view_type'])
            st.experimental_rerun()

        if st.session_state['key_frame_view_type'] == "Single Frame":
            header1, header2, header3 = st.columns([1, 1, 1])
            with header1:
                st.session_state['which_image'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", min_value=0, max_value=len(
                    timing_details)-1, step=1, value=st.session_state['which_image_value'], key="which_image_checker")
                if st.session_state['which_image_value'] != st.session_state['which_image']:
                    st.session_state['which_image_value'] = st.session_state['which_image']
                    st.experimental_rerun()
                index_of_current_item = st.session_state['which_image']

            with header3:
                st.write("")

            slider1, slider2 = st.columns([6, 12])
            # make a slider for choosing the frame to extract, starting from the previous frame number, and ending at the next frame number
            if index_of_current_item == 0:
                min_frames = 0
            else:
                min_frames = int(
                    float(timing_details[index_of_current_item-1]['frame_number'])) + 1

            if index_of_current_item == len(timing_details)-1:
                max_frames = int(total_frames) - 2
            else:
                max_frames = int(
                    float(timing_details[index_of_current_item+1]['frame_number'])) - 1

            with slider1:
                st.markdown(
                    f"Frame # for Key Frame {index_of_current_item}: {timing_details[index_of_current_item]['frame_number']}")
                # show frame time to the nearest 2 decimal places
                st.markdown(
                    f"Frame time: {round(float(timing_details[index_of_current_item]['frame_time']),2)}")

                if st.button("Delete current key frame"):
                    delete_frame(project_name, index_of_current_item)
                    timing_details = get_timing_details(project_name)
                    st.experimental_rerun()

            with slider2:
                if timing_details[index_of_current_item]["frame_number"] - 1 == timing_details[index_of_current_item-1]["frame_number"]\
                        and timing_details[index_of_current_item]["frame_number"] + 1 == timing_details[index_of_current_item+1]["frame_number"]:
                    st.warning(
                        "There's nowhere to move this frame due to it being 1 frame away from both the next and previous frame.")
                    new_frame_number = int(
                        float(timing_details[index_of_current_item]['frame_number']))
                else:
                    new_frame_number = st.slider(f"Choose which frame to preview for Key Frame #{index_of_current_item}:", min_value=min_frames, max_value=max_frames, step=1, value=int(
                        float(timing_details[index_of_current_item]['frame_number'])))

            preview1, preview2 = st.columns([1, 2])

            with preview1:
                if index_of_current_item == 0:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                else:
                    st.write("Previous frame:")
                    st.image(timing_details[index_of_current_item-1]
                             ["source_image"], use_column_width=True)

                if index_of_current_item == len(timing_details)-1:
                    st.write("")
                else:
                    st.write("Next frame:")
                    st.image(timing_details[index_of_current_item+1]
                             ["source_image"], use_column_width=True)
            with preview2:
                flag1, flag2 = st.columns([1, 1])
                with flag1:
                    st.write("Preview frame:")
                    st.write("")
                with flag2:
                    if new_frame_number == int(float(timing_details[index_of_current_item]['frame_number'])):
                        st.info(f"This is the current frame")
                    else:
                        st.info(
                            f"{timing_details[index_of_current_item]['frame_number']} is the current frame")

                st.image(preview_frame(project_name,
                         input_video, new_frame_number))

                bottom1, bottom2 = st.columns([1, 1])
                if new_frame_number != int(float(timing_details[index_of_current_item]['frame_number'])):

                    with bottom1:
                        if st.button("Update this frame to here"):
                            update_specific_timing_value(
                                project_name, index_of_current_item, "frame_number", new_frame_number)
                            timing_details = get_timing_details(project_name)
                            extract_frame(index_of_current_item, project_name,
                                          input_video, new_frame_number, timing_details)
                            st.experimental_rerun()
                    with bottom2:
                        if st.button("Add new key frame at this time"):
                            if new_frame_number > int(float(timing_details[index_of_current_item]['frame_number'])):
                                created_row = create_timings_row_at_frame_number(
                                    project_name, input_video, new_frame_number, timing_details, index_of_current_item+1)
                                timing_details = get_timing_details(
                                    project_name)
                                extract_frame(
                                    created_row, project_name, input_video, new_frame_number, timing_details)
                            elif new_frame_number < int(float(timing_details[index_of_current_item]['frame_number'])):
                                created_row = create_timings_row_at_frame_number(
                                    project_name, input_video, new_frame_number, timing_details, index_of_current_item)
                                timing_details = get_timing_details(
                                    project_name)
                                extract_frame(
                                    created_row, project_name, input_video, new_frame_number, timing_details)
                            timing_details = get_timing_details(project_name)
                            st.session_state['which_image_value'] = created_row
                            st.experimental_rerun()
                else:
                    with bottom1:
                        st.button("Update key frame to this frame #",
                                  disabled=True, help="This is the current frame.")
                    with bottom2:
                        st.button("Add new frame at this time",
                                  disabled=True, help="This is the current frame.")

        elif st.session_state['key_frame_view_type'] == "List View":
            for image_name in timing_details:

                index_of_current_item = timing_details.index(image_name)

                # if image starts with http
                if image_name["source_image"].startswith("http"):
                    image = timing_details[index_of_current_item]["source_image"]
                else:
                    image = Image.open(
                        timing_details[index_of_current_item]["source_image"])

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.subheader(f'Image Name: {index_of_current_item}')
                col2.empty()
                with col3:
                    if st.button("Delete this keyframe", key=f'{index_of_current_item}'):
                        delete_frame(project_name, index_of_current_item)
                        timing_details = get_timing_details(project_name)
                        st.experimental_rerun()

                st.image(image, use_column_width=True)

                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

                with col1:
                    frame_time = round(
                        float(timing_details[index_of_current_item]['frame_time']), 2)
                    st.markdown(f"Frame Time: {frame_time}")

                with col2:
                    # return frame number to 2 decimal places
                    frame_number = round(
                        float(timing_details[index_of_current_item]['frame_number']), 2)
                    st.markdown(f"Frame Number: {frame_number}")

                with col4:
                    if st.button(f"Jump to single frame view for #{index_of_current_item}", help="This will switch to a Single Frame view type and open this individual image."):
                        st.session_state['which_image_value'] = index_of_current_item
                        st.session_state['key_frame_view_type'] = "Single View"
                        st.session_state['key_frame_view_type_index'] = 1
                        st.session_state['open_manual_extractor'] = False
                        st.experimental_rerun()

        st.markdown("***")

    if 'open_manual_extractor' not in st.session_state:
        st.session_state['open_manual_extractor'] = True

    open_manual_extractor = st.checkbox(
        "Open manual Key Frame extractor", value=st.session_state['open_manual_extractor'])

    if open_manual_extractor is True:
        if project_settings["input_video"] == "":
            st.info(
                "You need to add an input video on the left before you can add key frames.")
        else:
            manual1, manual2 = st.columns([3, 1])
            with manual1:
                st.subheader('Add key frames to the end of your video:')
                st.write(
                    "Select a frame from the slider below and click 'Add Frame' it to the end of your project.")
                # if there are >10 frames, and show_current_key_frames == "Yes", show an info
                if len(timing_details) > 10 and st.session_state['key_frame_view_type'] == "List View":
                    st.info("You have over 10 frames visible. To keep the frame selector running fast, we recommend hiding the currently selected key frames by selecting 'No' in the 'Show currently selected key frames' section at the top of the page.")
            with manual2:
                st.write("")
                granularity = st.number_input("Choose selector granularity", min_value=1, max_value=50, step=1, value=1,
                                              help=f"This will extract frames for you to manually choose from. For example, if you choose 15 it'll extract every 15th frame.")

            if timing_details == []:
                min_frames = 0
            else:
                length_of_timing_details = len(timing_details) - 1
                min_frames = int(
                    float(timing_details[length_of_timing_details]["frame_number"]))

            max_frames = min_frames + 100

            if max_frames > int(float(total_frames)):
                max_frames = int(float(total_frames)) - 2

            slider = st.slider("Choose frame:", max_value=max_frames,
                               min_value=min_frames, step=granularity, value=min_frames)

            st.image(preview_frame(project_name, input_video, slider),
                     use_column_width=True)

            if st.button(f"Add Frame {slider} to Project"):
                last_index = len(timing_details)
                created_row = create_timings_row_at_frame_number(
                    project_name, input_video, slider, timing_details, last_index)
                timing_details = get_timing_details(project_name)
                extract_frame(created_row, project_name,
                              input_video, slider, timing_details)
                st.experimental_rerun()
        st.markdown("***")
        st.subheader("Make preview video at current timings")
        if st.button("Make Preview Video"):
            create_video_without_interpolation(timing_details, "preview")
            st.video(f'videos/{project_name}/preview.mp4')
