import datetime
import time
import jwt
import requests
from shared.constants import SERVER_URL, InternalResponse
import urllib.parse


def execute_shell_command(command: str):
    import subprocess

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print("Error:\n", result.stderr)
    return InternalResponse(result.stdout, "success", result.returncode == 0)


def is_online_file_path(file_path):
    parsed = urllib.parse.urlparse(file_path)
    return parsed.scheme in ("http", "https", "ftp")


def is_url_valid(url):
    try:
        response = requests.head(url, allow_redirects=True)
        final_response = response.history[-1] if response.history else response

        return final_response.status_code in [200, 201, 307]  # TODO: handle all possible status codes
    except Exception as e:
        return False


def get_file_type(url):
    try:
        response = requests.head(url)
        content_type = response.headers.get("Content-Type")

        if content_type and "image" in content_type:
            return "image"
        elif content_type and "video" in content_type:
            return "video"
        else:
            return "unknown"
    except Exception as e:
        print("Error:", e)
        return "unknown"


def generate_fresh_token(refresh_token):
    if not refresh_token:
        return None, None

    url = f"{SERVER_URL}/v1/authentication/refresh"

    payload = {}
    headers = {"Authorization": f"Bearer {refresh_token}"}

    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code == 200:
        data = response.json()
        return data["payload"]["token"], data["payload"]["refresh_token"]

    return None, None


def validate_token_through_db(token, refresh_token):
    url = f"{SERVER_URL}/v1/user/op"

    payload = {}
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code == 200:
        data = response.json()
        return token, refresh_token
    else:
        return generate_fresh_token(refresh_token)


def validate_token(
    token,
    refresh_token,
    validate_through_db=False,
):
    # returns a fresh token if the old one has expired
    # returns None if the token has expired or can't be renewed
    if not token:
        return None, None

    try:
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        exp = decoded_token.get("exp")

        if exp is None:
            return token, refresh_token

        now = time.time()
        if exp > now:
            if not validate_through_db:
                return token, refresh_token
            else:
                return validate_token_through_db(token, refresh_token)
        else:
            return generate_fresh_token(refresh_token)

    except jwt.ExpiredSignatureError:
        print("expired token, trying to refresh...")
        return generate_fresh_token(refresh_token)
    except Exception as e:
        print("error validating the jwt: ", str(e))
        return None, None
