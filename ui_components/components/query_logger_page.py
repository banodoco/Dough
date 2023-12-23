import json
import streamlit as st
from ui_components.constants import DefaultTimingStyleParams
from utils.common_utils import get_current_user

from utils.data_repo.data_repo import DataRepo

def query_logger_page():
    st.header("Inference Log list")

    data_repo = DataRepo()
    current_user = get_current_user()
    b1, b2 = st.columns([1, 1])

    total_log_table_pages = st.session_state['total_log_table_pages'] if 'total_log_table_pages' in st.session_state else DefaultTimingStyleParams.total_log_table_pages
    page_number = b1.number_input('Page number', min_value=1, max_value=total_log_table_pages, value=1, step=1)
    inference_log_list, total_page_count = data_repo.get_all_inference_log_list(
        page=page_number,
        data_per_page=100
    )

    if total_log_table_pages != total_page_count:
        st.session_state['total_log_table_pages'] = total_page_count
        st.rerun()

    data = {
        'Project': [],
        'Prompt': [],
        'Model': [],
        'Inference time (sec)': [],
        'Cost ($)': [],
        'Status': []
    }

    for log in inference_log_list:
        data['Project'].append(log.project.name)
        prompt = json.loads(log.input_params).get('prompt', '') if log.input_params else ''
        data['Prompt'].append(prompt)
        model_name = json.loads(log.output_details).get('model_name', '') if log.output_details else ''
        data['Model'].append(model_name)
        data['Inference time (sec)'].append(round(log.total_inference_time, 3))
        data['Cost ($)'].append(round(log.total_inference_time * 0.004, 3))
        data['Status'].append(log.status)

    
    st.table(data=data)