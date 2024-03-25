import time
import streamlit as st

from shared.constants import (
    AppSubPage,
    CreativeProcessPage,
    InferenceParamType,
    InferenceStatus,
    InferenceType,
    InternalFileTag,
    InternalFileType,
)
from ui_components.widgets.display_element import individual_video_display_element
from ui_components.widgets.frame_movement_widgets import jump_to_single_frame_view_button
import json
import math
from ui_components.widgets.frame_selector import update_current_frame_index

from utils.data_repo.data_repo import DataRepo
from utils.ml_processor.constants import ML_MODEL, MODEL_FILTERS


def sidebar_logger(shot_uuid):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    refresh_disabled = False  # not any(log.status in [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value] for log in log_list)
    z0, z1, z2 = st.columns([0.75, 0.75, 0.75])
    if z0.button(
        "Refresh log",
        disabled=refresh_disabled,
        type="primary",
        use_container_width=True,
        help="You can also press 'r' on your keyboard to refresh.",
    ):
        st.rerun()
    if z1.button(
        "Run backlog", help="This will run all the generations in the backlog.", use_container_width=True
    ):
        log_list_filter_data = {
            "project_id": shot.project.uuid,
            "page": 1,
            "data_per_page": 100,
            "status_list": [InferenceStatus.BACKLOG.value],
        }

        backlog_log_list, backlog_list_total_page_count = data_repo.get_all_inference_log_list(
            **log_list_filter_data
        )

        if backlog_log_list and len(backlog_log_list):
            status = data_repo.update_inference_log_list(
                [l.uuid for l in backlog_log_list], status=InferenceStatus.QUEUED.value
            )
            if status:
                st.success("success")
                time.sleep(0.7)
                st.rerun()
        else:
            st.info("No backlogs")
            time.sleep(0.7)
            st.rerun()
    y1, y2 = st.columns([1, 1])
    with y1:
        display_options = ["In Progress", "All", "Succeeded", "Failed", "Backlog"]
        if "status_optn_index" not in st.session_state:
            st.session_state["status_optn_index"] = 0

        status_option = st.radio(
            "Statuses to display:",
            options=display_options,
            key="status_option",
            index=st.session_state["status_optn_index"],
            horizontal=True,
        )

        st.session_state["status_optn_index"] = display_options.index(status_option)

    status_list = None
    if status_option == "In Progress":
        status_list = [InferenceStatus.QUEUED.value, InferenceStatus.IN_PROGRESS.value]
    elif status_option == "Succeeded":
        status_list = [InferenceStatus.COMPLETED.value]
    elif status_option == "Failed":
        status_list = [InferenceStatus.FAILED.value]
    elif status_option == "Backlog":
        status_list = [InferenceStatus.BACKLOG.value]

    project_setting = data_repo.get_project_setting(shot.project.uuid)
    with y2:

        selected_option = st.selectbox(
            "Which model to show:", ["All"] + [m.display_name() for m in MODEL_FILTERS]
        )
    page_number = y2.number_input(
        "Page number:", min_value=1, max_value=project_setting.total_log_pages, value=1, step=1
    )
    items_per_page = 5
    # items_per_page = z2.slider("Items per page", min_value=1, max_value=20, value=5, step=1)

    log_filter_data = {
        "project_id": shot.project.uuid,
        "page": page_number,
        "data_per_page": items_per_page,
        "status_list": status_list,
    }

    if selected_option != "All":
        log_filter_data["model_name_list"] = [
            selected_option
        ]  # multiple models can be entered here for filtering if needed

    log_list, total_page_count = data_repo.get_all_inference_log_list(**log_filter_data)

    if project_setting.total_log_pages != total_page_count:
        project_setting.total_log_pages = total_page_count
        st.rerun()
    with z2:
        if total_page_count > 1:
            st.caption(f"Total page count: {total_page_count}")
    # display_list = log_list[(page_number - 1) * items_per_page : page_number * items_per_page]

    if log_list and len(log_list):
        file_list = data_repo.get_file_list_from_log_uuid_list([log.uuid for log in log_list])
        log_file_dict = {}
        for file in file_list:
            log_file_dict[str(file.inference_log.uuid)] = file

        # st.markdown("---")
        for _, log in enumerate(log_list):
            origin_data = json.loads(log.input_params).get(InferenceParamType.ORIGIN_DATA.value, None)
            if not log.status:
                continue
            
            inference_type = origin_data.get("inference_type", "")
            output_url = None
            if log.uuid in log_file_dict:
                output_url = log_file_dict[log.uuid].location

            c0, c1, c2, c3, c4 = st.columns(
                [0.5, 0.5, 0.7 if output_url else 0.01, 1, 0.01 if output_url else 1]
            )
            with c0:
                input_params = json.loads(log.input_params)
                prompt = input_params.get("prompt", "No prompt found")
                st.caption(f"Prompt: \"{prompt[:5] + '...' if len(prompt) > 30 else prompt}\"")
                if inference_type == InferenceType.FRAME_INTERPOLATION.value:
                    video_inference_image_grid(origin_data)
                else:
                    st.caption("-\-\-\-\-\-\-\-\-")
            
            with c1:
                try:
                    model_name = json.loads(log.output_details)["model_name"].split("/")[-1]
                except Exception as e:
                    model_name = "Unavailable"
                st.caption(f"Model: {model_name}")

            with c2:
                if output_url:
                    if (
                        output_url.endswith("png")
                        or output_url.endswith("jpg")
                        or output_url.endswith("jpeg")
                        or output_url.endswith("gif")
                    ):
                        st.image(output_url)
                    elif output_url.endswith("mp4"):
                        individual_video_display_element(output_url)
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
                elif log.status == InferenceStatus.BACKLOG.value:
                    st.warning("Backlog")

                """
                log_file = log_file_dict[log.uuid] if log.uuid in log_file_dict else None
                if log_file:
                    if log_file.type == InternalFileType.IMAGE.value and log_file.tag != InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value:
                        if st.button("Add to shortlist ➕", key=f"sidebar_shortlist_{log_file.uuid}",use_container_width=True, help="Add to shortlist"):
                            data_repo.update_file(log_file.uuid, tag=InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value)
                            st.success("Added To Shortlist")
                            time.sleep(0.3)
                            st.rerun()
                """

                if log.status in [InferenceStatus.QUEUED.value, InferenceStatus.BACKLOG.value]:
                    if st.button(
                        "Cancel", key=f"cancel_gen_{log.uuid}", use_container_width=True, help="Cancel"
                    ):
                        err_msg = "Generation has already started"
                        success_msg = "Generation cancelled"
                        # fetching the current status as this could have been already started
                        log = data_repo.get_inference_log_from_uuid(log.uuid)
                        cur_status = log.status
                        if cur_status not in [InferenceStatus.QUEUED.value, InferenceStatus.BACKLOG.value]:
                            st.error(err_msg)
                            time.sleep(0.7)
                            st.rerun()
                        else:
                            res = data_repo.update_inference_log(
                                uuid=log.uuid, status=InferenceStatus.CANCELED.value
                            )
                            if not res:
                                st.error(err_msg)
                            else:
                                st.success(success_msg)
                            time.sleep(0.7)
                            st.rerun()

                if output_url and origin_data:
                    if inference_type == InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value:
                        timing = data_repo.get_timing_from_uuid(origin_data.get("timing_uuid"))
                        if timing and st.session_state["frame_styling_view_type"] != "Timeline":
                            jump_to_single_frame_view_button(
                                timing.aux_frame_index + 1, timing_list, "sidebar_" + str(log.uuid)
                            )

                    elif inference_type == InferenceType.GALLERY_IMAGE_GENERATION.value:
                        pass
                        # if st.session_state['page'] != "Explore":
                        #     if st.button(f"Jump to explorer", key=str(log.uuid)):
                        #         # TODO: fix this
                        #         st.session_state['main_view_type'] = "Creative Process"
                        #         st.session_state['frame_styling_view_type_index'] = 0
                        #         st.session_state['frame_styling_view_type'] = "Explorer"
                        #         st.rerun()

                    elif inference_type == InferenceType.FRAME_INTERPOLATION.value:
                        jump_to_shot_button(origin_data.get('shot_uuid', ''), log.uuid)

def video_inference_image_grid(origin_data):
    if origin_data:
        if 'settings' in origin_data and 'file_uuid_list' in origin_data['settings'] \
            and origin_data['settings']['file_uuid_list']:
            data_repo = DataRepo()
            total_size = len(origin_data['settings']['file_uuid_list'])
            file_uuid_list = origin_data['settings']['file_uuid_list'][:3]
            image_list, _ = data_repo.get_all_file_list(uuid__in=file_uuid_list, file_type=InternalFileType.IMAGE.value)    # extra element for displaying pending count
            
            num_images = len(image_list)
            for index in range(num_images + 1):
                if index < num_images and image_list[index]:
                    st.image(image_list[index].location, width=30)
                else:
                    pending_count = total_size - len(image_list)
                    if pending_count:
                        st.info('+' + str(pending_count))

def jump_to_shot_button(shot_uuid, log_uuid):
    if shot_uuid:
        if st.button("Jump to Shot", key=f"sidebar_btn_{shot_uuid}_{log_uuid}"):
            st.session_state['current_frame_sidebar_selector'] = 0
            st.session_state['page'] = CreativeProcessPage.ANIMATE_SHOT.value
            st.session_state['current_subpage'] = AppSubPage.ANIMATE_SHOT.value
            st.session_state["shot_uuid"] = shot_uuid
            st.rerun()