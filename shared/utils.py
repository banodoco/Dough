from shared.constants import InternalResponse
import urllib.parse


def execute_shell_command(command: str):
    import subprocess

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print("Error:\n", result.stderr)
    return InternalResponse(result.stdout, 'success', result.returncode == 0)

def is_online_file_path(file_path):
    parsed = urllib.parse.urlparse(file_path)
    return parsed.scheme in ('http', 'https', 'ftp')