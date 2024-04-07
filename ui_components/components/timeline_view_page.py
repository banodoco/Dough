import time
from shared.constants import COMFY_BASE_PATH
import streamlit as st
import os
import requests
import shutil
from zipfile import ZipFile
from io import BytesIO
from ui_components.constants import CreativeProcessType
from ui_components.methods.video_methods import upscale_video
from ui_components.widgets.timeline_view import timeline_view
from ui_components.components.explorer_page import gallery_image_view
from utils import st_memory
from utils.data_repo.data_repo import DataRepo
from ui_components.widgets.sidebar_logger import sidebar_logger
from ui_components.components.explorer_page import generate_images_element

def timeline_view_page(shot_uuid: str, h2):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    project_uuid = shot.project.uuid
    project = data_repo.get_project_from_uuid(project_uuid)

    with st.sidebar:
        views = CreativeProcessType.value_list()

        if "view" not in st.session_state:
            st.session_state["view"] = views[0]

        st.write("")    

        with st.expander("🔍 Generation log", expanded=True):
            # if st_memory.toggle("Open", value=True, key="generaton_log_toggle"):
            sidebar_logger(st.session_state["shot_uuid"])
        
        st.write("")

        with st.expander("📋 Explorer shortlist",expanded=True):
            if st_memory.toggle("Open", value=True, key="explorer_shortlist_toggle"):
                gallery_image_view(shot.project.uuid, shortlist=True, view=["add_and_remove_from_shortlist","add_to_any_shot"])
        
    st.markdown(f"#### :green[{st.session_state['main_view_type']}] > :red[{st.session_state['page']}]")
    st.markdown("***")
    slider1, slider2, slider3 = st.columns([2,1,1])
    with slider1:
        st.markdown(f"### 🪄 '{project.name}' timeline")
        st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")

    shot_list = data_repo.get_shot_list(project_uuid)
    main_clip_list = []
    for shot in shot_list:
        if shot.main_clip and shot.main_clip.location:
            main_clip_list.append(shot.main_clip.location)
    
    with slider3:
        with st.expander("Export all main variants", expanded=False):
            if not len(main_clip_list):
                st.info("No videos available in the project.")
            
            else:
                if st.button('Prepare videos for download'):
                    temp_dir = 'temp_main_variants'
                    os.makedirs(temp_dir, exist_ok=True)
                    zip_data = BytesIO()
                    st.info("Preparing videos for download. This may take a while.")                    
                    time.sleep(0.4)
                    try:
                        for idx, shot in enumerate(shot_list):
                            if shot.main_clip and shot.main_clip.location:
                                # Prepend the shot number (idx + 1) to the filename
                                file_name = f'{idx + 1:03d}_{shot.name}.mp4'  # Using :03d to ensure the number is zero-padded to 3 digits
                                file_path = os.path.join(temp_dir, file_name)
                                if shot.main_clip.location.startswith('http'):
                                    response = requests.get(shot.main_clip.location)
                                    with open(file_path, 'wb') as f:
                                        f.write(response.content)
                                else:
                                    shutil.copyfile(shot.main_clip.location, file_path)

                        with ZipFile(zip_data, 'w') as zipf:
                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    zipf.write(os.path.join(root, file), file)

                        st.download_button(
                            label="Download Main Variant Videos zip",
                            data=zip_data.getvalue(),
                            file_name="main_variant_videos.zip",
                            mime='application/zip',
                            key="main_variant_download",
                            use_container_width=True,
                            type="primary"
                        )
                    finally:
                        shutil.rmtree(temp_dir)
    
    with slider2:
        with st.expander("Bulk upscale", expanded=False):
            def upscale_settings():
                checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
                all_files = os.listdir(checkpoints_dir)
                if len(all_files) == 0:
                    st.info("No models found in the checkpoints directory")
                    styling_model = "None"
                else:
                    # Filter files to only include those with .safetensors and .ckpt extensions
                    model_files = [file for file in all_files if file.endswith('.safetensors') or file.endswith('.ckpt')]
                    # drop all files that contain xl
                    model_files = [file for file in model_files if "xl" not in file]
                    # model_files.insert(0, "None")  # Add "None" option at the beginning
                    styling_model = st.selectbox("Styling model", model_files, key="styling_model")

                type_of_upscaler = st.selectbox("Type of upscaler", ["Dreamy", "Realistic", "Anime", "Cartoon"], key="type_of_upscaler")
                upscale_by = st.slider("Upscale by", min_value=1.0, max_value=3.0, step=0.1, key="upscale_by", value=2.0)
                strength_of_upscale = st.slider("Strength of upscale", min_value=1.0, max_value=3.0, step=0.1, key="strength_of_upscale", value=2.0)
                set_upscaled_to_main_variant = st.checkbox("Set upscaled to main variant", key="set_upscaled_to_main_variant", value=True)
                
                return styling_model, type_of_upscaler, upscale_by, strength_of_upscale, set_upscaled_to_main_variant
            
            if not len(main_clip_list):
                st.info("No videos to upscale")
            else:
                styling_model, upscaler_type, upscale_factor, upscale_strength, promote_to_main_variant = upscale_settings()
                if st.button("Upscale All Main Variants"):
                    for shot in shot_list:
                        if shot.main_clip and shot.main_clip.location:
                            upscale_video(
                                shot.uuid,
                                styling_model, 
                                upscaler_type, 
                                upscale_factor, 
                                upscale_strength, 
                                promote_to_main_variant
                            )
                            
    # start_time = time.time()
    timeline_view(st.session_state["shot_uuid"], st.session_state['view'])
    st.markdown("### ✨ Generate frames")
    st.write("##### _\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_")
    
    # end_time = time.time()
    # print("///////////////// timeline laoded in: ", end_time - start_time)
    generate_images_element(position='explorer', project_uuid=project_uuid, timing_uuid=None, shot_uuid=None)
    # end_time = time.time()
    # print("///////////////// generate img laoded in: ", end_time - start_time)
    gallery_image_view(project_uuid,False,view=['add_and_remove_from_shortlist','view_inference_details','shot_chooser','add_to_any_shot'])
    # end_time = time.time()
    # print("///////////////// gallery laoded in: ", end_time - start_time)