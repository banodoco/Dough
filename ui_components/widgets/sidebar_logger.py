import streamlit as st

from shared.constants import InferenceParamType, InferenceStatus

import json
import math

def sidebar_logger(data_repo, project_uuid):
    a1, _, a3 = st.columns([1, 0.2, 1])
    
    log_list = data_repo.get_all_inference_log_list(project_uuid)
    refresh_disabled = not any(log.status in [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value] for log in log_list)

    if a1.button("Refresh log", disabled=refresh_disabled): st.rerun()
    a3.button("Jump to full log view")

    # Add radio button for status selection
    status_option = st.radio("Statuses to display:", options=["All", "In Progress", "Succeeded", "Failed"], key="status_option", index=0, horizontal=True)

    # Filter log_list based on selected status
    if status_option == "In Progress":
        log_list = [log for log in log_list if log.status in [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value]]
    elif status_option == "Succeeded":
        log_list = [log for log in log_list if log.status == InferenceStatus.COMPLETED.value]
    elif status_option == "Failed":
        log_list = [log for log in log_list if log.status == InferenceStatus.FAILED.value]

    b1, b2 = st.columns([1, 1])
    items_per_page = b2.slider("Items per page", min_value=1, max_value=20, value=5, step=1)
    page_number = b1.number_input('Page number', min_value=1, max_value=math.ceil(len(log_list) / items_per_page), value=1, step=1)
    
    log_list = log_list[::-1][(page_number - 1) * items_per_page : page_number * items_per_page]                

    st.markdown("---")

    for idx, log in enumerate(log_list):  
                                
        origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, None)
        if not log.status or not origin_data:
            continue
        
        output_url = None
        output_data = json.loads(log.output_details)
        if 'output' in output_data and output_data['output']:
            output_url = output_data['output'][0] if isinstance(output_data['output'], list) else output_data['output']                        
        
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
            
            if output_url:
                if st.button(f"Jump to frame {idx}"):
                    st.info("Fix this.")
            
            # if st.button("Delete", key=f"delete_{log.uuid}"):
            #    data_repo.update_inference_log(log.uuid, status="")
            #    st.rerun()
        
        st.markdown("---")