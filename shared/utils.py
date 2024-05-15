import requests
from shared.constants import InternalResponse
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
