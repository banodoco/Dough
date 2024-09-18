import json
import os
import socket
import time

import requests
import streamlit as st
from shared.constants import (
    HOSTED_BACKGROUND_RUNNER_MODE,
    SERVER,
    SERVER_URL,
    InferenceStatus,
    InternalFileType,
    InternalResponse,
    ServerType,
)
from shared.utils import validate_token
from ui_components.models import InternalUserObject
from utils.common_decorators import log_time

from utils.data_repo.data_repo import DataRepo
from utils.state_refresh import refresh_app
from utils.third_party_auth.google.google_auth import get_auth_provider


# APIRepo connects to the hosted backend of Banodoco
class APIRepo:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIRepo, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_base_url()
        self._setup_urls()

        self.data_repo = DataRepo()
        self.auth_provider = get_auth_provider()

    def _load_base_url(self):
        # SERVER_URL = os.getenv("SERVER_URL", "")
        # if not SERVER_URL.startswith("http"):
        #     # connecting through service discovery
        #     self.base_url = "http://" + socket.gethostbyname(SERVER_URL) + ":8080"
        self.base_url = SERVER_URL

    def _setup_urls(self):
        # user
        self.USER_OP_URL = "/v1/user/op"

        # payment
        self.PAYMENT_OP_URL = "/v1/payment/stripe-link"

        # inference log
        self.LOG_URL = "/v1/inference/log"

        # file
        self.FILE_URL = "/v1/user/file"

        # payment
        self.STRIPE_PAYMENT_URL = "/v1/payment/stripe-link"

    def is_user_logged_in(self):
        auth_token, _ = self._get_auth_token()
        return True if auth_token else False

    def logout(self):
        st.session_state["auth_token"] = ""
        st.session_state["refresh_auth_token"] = ""
        self.set_auth_token("", "", user={"name": "", "email": ""})
        refresh_app()

    def _get_auth_token(self, validate_through_db=False):
        """
        it fetches the auth from db and validates it, and stores it in the session_state for fast access.
        the token in the session_state is refreshed every 5 mins, incase it expires.
        """
        data_repo = DataRepo()

        auth_token, refresh_auth_token = None, None
        app_secrets = None
        auth_timeout = 5 * 60  # 5 mins timeout

        if (
            "last_auth_refresh" in st.session_state
            and time.time() - st.session_state["last_auth_refresh"] > auth_timeout
        ):
            st.session_state["last_auth_refresh"] = time.time()
            st.session_state["auth_token"] = ""
            st.session_state["refresh_auth_token"] = ""

        if "auth_token" in st.session_state and st.session_state["auth_token"]:
            auth_token = st.session_state["auth_token"]
            refresh_auth_token = st.session_state.get("refresh_auth_token", None)

        else:
            user: InternalUserObject = data_repo.get_first_active_user()
            if not user:
                return None, None

            app_secrets = data_repo.get_app_secrets_from_user_uuid(user.uuid)
            if not ("aws_access_key" in app_secrets and app_secrets["aws_access_key"]):
                return None, None

            auth_token, refresh_auth_token = app_secrets["aws_access_key"], app_secrets["aws_secret_key"]
            # if loading in the session state for the first time, then we validate against the db
            validate_through_db = True

        auth_token, refresh_auth_token = validate_token(auth_token, refresh_auth_token, validate_through_db)
        if auth_token:
            # updating session state
            st.session_state["auth_token"] = auth_token
            st.session_state["refresh_auth_token"] = refresh_auth_token

            # updating db if the token has changed
            if app_secrets and app_secrets["aws_access_key"] != auth_token:
                self.set_auth_token(auth_token, refresh_auth_token)

            return auth_token, refresh_auth_token

        else:
            self.set_auth_token("", "")
            st.rerun()  # refreshing the app when auth token is not present so that the user can login
            return None, None

    def set_auth_token(self, auth_token, refresh_token, user=None):
        """
        TODO: right now, auth_token is being stored in aws_access_key and refresh_token is being stored
        in aws_secret_key. Have to do proper code cleanup + fresh migrations.
        """

        data_repo = DataRepo()
        data_repo.update_app_setting(**{"aws_access_key": auth_token, "aws_secret_access_key": refresh_token})

        if not auth_token:
            # clearing data
            st.session_state["auth_token"] = ""
            st.session_state["refresh_auth_token"] = ""

        if user:
            current_user = data_repo.get_first_active_user()
            data_repo.update_user(
                current_user.uuid,
                name=user.get("name", ""),
                email=user.get("email", ""),
            )
        return

    ################### base http methods
    def _get_headers(self, content_type="application/json"):
        auth_token, _ = self._get_auth_token()

        headers = {}
        headers["Authorization"] = f"Bearer {auth_token}"
        if content_type:
            headers["Content-Type"] = content_type

        return headers

    # @log_time
    def http_get(self, url, params=None):
        self._load_base_url()
        res = requests.get(self.base_url + url, params=params, headers=self._get_headers())
        return res.json()

    # @log_time
    def http_post(self, url, data={}, file_content=None):
        self._load_base_url()
        if file_content:
            files = {"file": file_content}
            res = requests.post(self.base_url + url, data=data, files=files, headers=self._get_headers(None))
        else:
            res = requests.post(self.base_url + url, json=data, headers=self._get_headers())

        return res.json()

    # @log_time
    def http_put(self, url, data=None):
        self._load_base_url()
        res = requests.put(self.base_url + url, json=data, headers=self._get_headers())
        return res.json()

    # @log_time
    def http_delete(self, url, params=None):
        self._load_base_url()
        res = requests.delete(self.base_url + url, params=params, headers=self._get_headers())
        return res.json()

    #########################################

    def update_log_status(self, log_uuid, **kwargs):
        data = kwargs
        data["uuid"] = log_uuid

        return self.http_put(self.LOG_URL, data)

    def create_log(self, log_data, model_name="llama3"):
        credits_remaining = self.get_user_credits()
        if not credits_remaining:
            st.error("Insufficient credits")
            return

        # this also creates log in the local database
        # this should only be used when the results return immediately
        res = self.http_post(self.LOG_URL, log_data)
        status = res.get("payload", {}).get("data", {}).get("status", InferenceStatus.FAILED.value)
        output = "" if status == InferenceStatus.FAILED.value else res["payload"]["data"]["output_details"]
        if status:
            total_time = res["payload"]["data"]["total_inference_time"]
            total_credits_used = res["payload"]["data"]["total_credits_used"]
        else:
            total_time, total_credits_used = 0, 0

        log_data = log_data = {
            "project_id": st.session_state["project_uuid"],
            "model_id": None,
            "input_params": json.dumps(log_data),
            "output_details": output,
            "total_inference_time": total_time,
            "credits_used": total_credits_used,
            "status": (
                InferenceStatus.COMPLETED.value
                if status == InferenceStatus.COMPLETED.value
                else InferenceStatus.FAILED.value
            ),
            "model_name": model_name,
            "generation_source": "",
            "generation_tag": "",
        }

        inference_log = self.data_repo.create_inference_log(**log_data)
        return inference_log

    def get_cur_user(self):
        return self.http_get(self.USER_OP_URL)

    def get_signed_url(self, file_info):
        try:
            response = self.http_post(self.FILE_URL, data=file_info)
            return response["payload"]["data"]["signed_url"], response["payload"]["data"]["public_url"]
        except requests.RequestException as e:
            print(f"Failed to get signed URL for {file_info}: {str(e)}")
            return None, None

    def get_user_credits(self):
        """
        this is a soft check on the credit (as it uses a cached value). the final
        check is done when the task is picked for processing, the db has the updated value at that time
        """
        credits_remaining = 0
        credit_data_timeout = 2 * 60  # 2 mins
        if (
            "user_credit_data" in st.session_state
            and st.session_state["user_credit_data"]
            and time.time() - st.session_state["user_credit_data"]["created_on"] <= credit_data_timeout
        ):
            credits_remaining = st.session_state["user_credit_data"]["balance"]
        else:
            response = self.get_cur_user()
            credits_remaining = response.get("payload", {}).get("data", 0).get("total_credits", 0)
            st.session_state["user_credit_data"] = {
                "balance": credits_remaining,
                "created_on": time.time(),
            }

        return credits_remaining

    def generate_payment_link(self, amount):
        res = self.http_get(self.PAYMENT_OP_URL, params={"total_amount": amount})
        return res["payload"]["data"] if res["status"] else ""
