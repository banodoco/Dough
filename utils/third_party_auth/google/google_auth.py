from abc import ABC
from dotenv import load_dotenv
import requests

from shared.constants import SERVER_URL
from shared.logging.logging import AppLogger

load_dotenv()

logger = AppLogger()


def get_auth_provider():
    return GoogleAuthProvider()  # TODO: add test provider for development


class AuthProvider(ABC):
    def get_auth_url(self, *args, **kwargs):
        pass

    def verify_auth_details(self, *args, **kwargs):
        pass


# TODO: make this implementation 'proper'
class GoogleAuthProvider(AuthProvider):
    def __init__(self):
        self.url = f"{SERVER_URL}/v1/authentication/google"

    def get_auth_url(self, redirect_uri):
        params = {"redirect_uri": redirect_uri}

        response = requests.get(self.url, params=params)

        if response.status_code == 200:
            data = response.json()
            print(data)
            auth_url = data["payload"]["data"]["url"]
            return f"""<a target='_self' href='{auth_url}'> Google login -> </a>"""
        else:
            print(f"Error: {response.status_code} - {response.text}")

        return None

    def verify_auth_details(self, auth_details=None):
        response = requests.post(self.url, json=auth_details, headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            data = response.json()
            if not data["status"]:
                return None, None, None

            print("response found: ", data)
            user = {"name": data["payload"]["user"]["name"], "email": data["payload"]["user"]["email"]}
            return data["payload"]["token"], data["payload"]["refresh_token"], user
        else:
            logger.error("auth verification failed:", response.text)

        return None, None, None
