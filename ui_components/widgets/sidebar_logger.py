import streamlit as st

from shared.constants import InferenceParamType, InferenceStatus
from ui_components.widgets.frame_movement_widgets import jump_to_single_frame_view_button
import json
import math
from ui_components.widgets.frame_selector import update_current_frame_index

from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.replicate.constants import REPLICATE_MODEL

def sidebar_logger(project_uuid):
    data_repo = DataRepo()

    timing_details = data_repo.get_timing_list_from_project(project_uuid=project_uuid)

    a1, _, a3 = st.columns([1, 0.2, 1])
    
    refresh_disabled = False # not any(log.status in [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value] for log in log_list)

    if a1.button("Refresh log", disabled=refresh_disabled): st.rerun()
    # a3.button("Jump to full log view")

    status_option = st.radio("Statuses to display:", options=["All", "In Progress", "Succeeded", "Failed"], key="status_option", index=0, horizontal=True)

    status_list = None
    if status_option == "In Progress":
        status_list = [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value]
    elif status_option == "Succeeded":
        status_list = [InferenceStatus.COMPLETED.value]
    elif status_option == "Failed":
        status_list = [InferenceStatus.FAILED.value]

    b1, b2 = st.columns([1, 1])

    project_setting = data_repo.get_project_setting(project_uuid)
    
    page_number = b1.number_input('Page number', min_value=1, max_value=project_setting.total_log_pages, value=1, step=1)
    items_per_page = b2.slider("Items per page", min_value=1, max_value=20, value=5, step=1)
    log_list, total_page_count = data_repo.get_all_inference_log_list(
        project_id=project_uuid, 
        page=page_number, 
        data_per_page=items_per_page, 
        status_list=status_list
    )
    
    if project_setting.total_log_pages != total_page_count:
        project_setting.total_log_pages = total_page_count
        st.rerun()
    
    st.write("Total page count: ", total_page_count)
    # display_list = log_list[(page_number - 1) * items_per_page : page_number * items_per_page]                

    if log_list and len(log_list):
        file_list = data_repo.get_file_list_from_log_uuid_list([log.uuid for log in log_list])
        log_file_dict = {}
        for file in file_list:
            log_file_dict[str(file.inference_log.uuid)] = file

        st.markdown("---")

        for _, log in enumerate(log_list):
            origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, None)
            if not log.status:
                continue
            
            output_url = None
            if log.uuid in log_file_dict:
                output_url = log_file_dict[log.uuid].location

            c1, c2, c3 = st.columns([1, 1 if output_url else 0.01, 1])

            with c1:                
                input_params = json.loads(log.input_params)
                st.caption(f"Prompt:")
                prompt = input_params.get('prompt', 'No prompt found')                
                st.write(f'"{prompt[:30]}..."' if len(prompt) > 30 else f'"{prompt}"')
                st.caption(f"Model:")
                st.write(json.loads(log.output_details)['model_name'].split('/')[-1])
                            
            with c2:
                if output_url:                                              
                    if output_url.endswith('png') or output_url.endswith('jpg') or output_url.endswith('jpeg') or output_url.endswith('gif'):
                        st.image(output_url)
                    elif output_url.endswith('mp4'):
                        st.video(output_url, format='mp4', start_time=0)
                    else:
                        st.info("No data to display")         
        
            with c3:
                if log.status == InferenceStatus.COMPLETED.value:
                    st.success("Completed")
                elif log.status == InferenceStatus.FAILED.value:
                    st.warning("Failed")
                elif log.status == InferenceStatus.QUEUED.value:
                    st.info("Queued")
                elif log.status == InferenceStatus.IN_PROGRESS.value:
                    st.info("In progress")
                elif log.status == InferenceStatus.CANCELED.value:
                    st.warning("Canceled")
                
                if output_url and 'timing_uuid' in origin_data:
                    timing = data_repo.get_timing_from_uuid(origin_data['timing_uuid'])
                    if timing and st.session_state['frame_styling_view_type'] != "Timeline":
                        jump_to_single_frame_view_button(timing.aux_frame_index + 1, timing_details, 'sidebar_'+str(log.uuid))     

                    else:
                        if st.session_state['frame_styling_view_type'] != "Explorer":
                            if st.button(f"Jump to explorer", key=str(log.uuid)):
                                # TODO: fix this
                                st.session_state['main_view_type'] = "Creative Process"
                                st.session_state['frame_styling_view_type_index'] = 0
                                st.session_state['frame_styling_view_type'] = "Explorer"
                                st.session_state['change_view_type'] = False
                                st.rerun()
                
                
            st.markdown("---")