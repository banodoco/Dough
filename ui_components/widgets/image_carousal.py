import streamlit as st
from st_clickable_images import clickable_images

from ui_components.constants import WorkflowStageType
from ui_components.models import InternalShotObject
from utils.data_repo.data_repo import DataRepo


def display_image(timing_uuid, stage=None, clickable=False):
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)
    timing_idx = timing.aux_frame_index

    # if it's less than 0 or greater than the number in timing_details, show nothing
    if not timing:
        st.write("no images")

    else:
        if stage == WorkflowStageType.STYLED.value:
            image = timing.primary_image_location
        elif stage == WorkflowStageType.SOURCE.value:
            image = timing.source_image.location if timing.source_image else ""

        if image != "":
            if clickable is True:
                if "counter" not in st.session_state:
                    st.session_state["counter"] = 0

                import base64

                if image.startswith("http"):
                    st.write("")
                else:
                    with open(image, "rb") as image:
                        st.write("")
                        encoded = base64.b64encode(image.read()).decode()
                        image = f"data:image/jpeg;base64,{encoded}"

                st.session_state[f"{timing_idx}_{stage}_clicked"] = clickable_images(
                    [image],
                    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                    img_style={"max-width": "100%", "height": "auto", "cursor": "pointer"},
                    key=f"{timing_idx}_{stage}_image_{st.session_state['counter']}",
                )

                if st.session_state[f"{timing_idx}_{stage}_clicked"] == 0:
                    timing_details = data_repo.get_timing_list_from_shot(timing.shot.uuid)
                    st.session_state["current_frame_uuid"] = timing_details[timing_idx].uuid
                    st.session_state["current_frame_index"] = timing_idx + 1
                    st.session_state["prev_frame_index"] = timing_idx + 1
                    # st.session_state['frame_styling_view_type_index'] = 0
                    st.session_state["frame_styling_view_type"] = "Individual"
                    st.session_state["counter"] += 1
                    st.rerun()

            elif clickable is False:
                st.image(image, use_column_width=True)
        else:
            st.error(f"No {stage} image found for #{timing_idx + 1}")


def carousal_of_images_element(shot_uuid, stage=WorkflowStageType.STYLED.value):
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    timing_list = shot.timing_list

    header1, header2, header3, header4, header5 = st.columns([1, 1, 1, 1, 1])

    current_frame_uuid = st.session_state["current_frame_uuid"]
    current_timing = data_repo.get_timing_from_uuid(current_frame_uuid)

    with header1:
        if current_timing.aux_frame_index - 2 >= 0:
            prev_2_timing = data_repo.get_timing_from_frame_number(
                shot_uuid, current_timing.aux_frame_index - 2
            )

            if prev_2_timing:
                display_image(prev_2_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{prev_2_timing.aux_frame_index + 1}")

    with header2:
        if current_timing.aux_frame_index - 1 >= 0:
            prev_timing = data_repo.get_timing_from_frame_number(
                shot_uuid, current_timing.aux_frame_index - 1
            )
            if prev_timing:
                display_image(prev_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{prev_timing.aux_frame_index + 1}")

    with header3:

        timing = data_repo.get_timing_from_uuid(current_frame_uuid)
        display_image(timing.uuid, stage=stage, clickable=True)
        st.success(f"#{current_timing.aux_frame_index + 1}")
    with header4:
        if current_timing.aux_frame_index + 1 <= len(timing_list):
            next_timing = data_repo.get_timing_from_frame_number(
                shot_uuid, current_timing.aux_frame_index + 1
            )
            if next_timing:
                display_image(next_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{next_timing.aux_frame_index + 1}")

    with header5:
        if current_timing.aux_frame_index + 2 <= len(timing_list):
            next_2_timing = data_repo.get_timing_from_frame_number(
                shot_uuid, current_timing.aux_frame_index + 2
            )
            if next_2_timing:
                display_image(next_2_timing.uuid, stage=stage, clickable=True)
                st.info(f"#{next_2_timing.aux_frame_index + 1}")
