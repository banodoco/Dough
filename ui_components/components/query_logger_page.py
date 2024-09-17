import json
import time
import requests
import streamlit as st
from ui_components.constants import DefaultTimingStyleParams
from utils.common_utils import get_current_user
from shared.constants import ServerType
from utils.data_repo.api_repo import APIRepo
from utils.data_repo.data_repo import DataRepo
from utils.state_refresh import refresh_app


def query_logger_page():
    st.markdown("##### Inference log")

    data_repo = DataRepo()
    api_repo = APIRepo()

    credits_remaining = 0
    credit_data_timeout = 2 * 60  # 2 mins
    if (
        "user_credit_data" in st.session_state
        and st.session_state["user_credit_data"]
        and time.time() - st.session_state["user_credit_data"]["created_on"] <= credit_data_timeout
    ):
        credits_remaining = st.session_state["user_credit_data"]["balance"]
    else:
        response = api_repo.get_cur_user()
        credits_remaining = response.get("payload", {}).get("data", 0).get("total_credits", 0)
        st.session_state["user_credit_data"] = {
            "balance": credits_remaining,
            "created_on": time.time(),
        }

    c01, c02, _ = st.columns([1, 1, 2])

    credits_remaining = round(credits_remaining, 3)
    with c01:
        st.write(f"### Credit Balance: {credits_remaining}")

    with c02:
        if st.button("Refresh Credits"):
            st.session_state["user_credit_data"] = None
            refresh_app()

    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        credits_to_buy = st.number_input(
            label="Credits to Buy",
            key="credit_btn",
            min_value=1,
            step=1,
        )

    with c2:
        st.write("")
        st.write("")
        if st.button("Generate payment link"):
            st.write("Please click on the link below to make the payment")
            st.write("www.google.com")

    b1, b2 = st.columns([1, 0.2])

    total_log_table_pages = (
        st.session_state["total_log_table_pages"]
        if "total_log_table_pages" in st.session_state
        else DefaultTimingStyleParams.total_log_table_pages
    )
    list_of_pages = [i for i in range(1, total_log_table_pages + 1)]
    page_number = b1.radio(
        "Select page:", options=list_of_pages, key="inference_log_page_number", index=0, horizontal=True
    )
    # page_number = b1.number_input('Page number', min_value=1, max_value=total_log_table_pages, value=1, step=1)
    inference_log_list, total_page_count = data_repo.get_all_inference_log_list(
        page=page_number, data_per_page=100
    )

    if total_log_table_pages != total_page_count:
        st.session_state["total_log_table_pages"] = total_page_count
        refresh_app()

    data = {
        "Project": [],
        "Prompt": [],
        "Model": [],
        "Inference time (sec)": [],
        "Credits": [],
        "Status": [],
    }

    # if SERVER != ServerType.DEVELOPMENT.value:
    #     data[] = []

    if inference_log_list and len(inference_log_list):
        for log in inference_log_list:
            data["Project"].append(log.project.name)
            prompt = json.loads(log.input_params).get("prompt", "") if log.input_params else ""
            data["Prompt"].append(prompt)
            model_name = log.model_name
            data["Model"].append(model_name)
            data["Inference time (sec)"].append(round(log.total_inference_time, 3))
            data["Credits"].append(round(log.credits_used, 3))
            data["Status"].append(log.status)

        st.table(data=data)
        st.markdown("***")
    else:
        st.info("No logs present")
