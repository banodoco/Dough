from utils.data_repo.data_repo import DataRepo
import streamlit as st
import time
from utils.state_refresh import refresh_app


def duplicate_shot_button(shot_uuid, position="shot_view"):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    if st.button(
        "Duplicate shot",
        key=f"duplicate_btn_{shot.uuid}_{position}",
        help="This will duplicate this shot.",
        use_container_width=True,
    ):
        data_repo.duplicate_shot(shot.uuid)
        st.success("Shot duplicated successfully")
        time.sleep(0.3)
        refresh_app()
