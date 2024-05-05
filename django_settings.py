import os
from pathlib import Path
import sys

sys.path.append("../")


from dotenv import load_dotenv

from shared.constants import HOSTED_BACKGROUND_RUNNER_MODE, LOCAL_DATABASE_NAME, SERVER, ServerType


load_dotenv()

if SERVER == ServerType.DEVELOPMENT.value:
    DB_LOCATION = LOCAL_DATABASE_NAME
else:
    DB_LOCATION = ""

BASE_DIR = Path(__file__).resolve().parent.parent

if HOSTED_BACKGROUND_RUNNER_MODE in [False, "False"]:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": DB_LOCATION,
        }
    }
else:
    import boto3

    ssm = boto3.client("ssm", region_name="ap-south-1")
    DB_NAME = ssm.get_parameter(Name="/backend/banodoco/db/name")["Parameter"]["Value"]
    DB_USER = ssm.get_parameter(Name="/backend/banodoco/db/user")["Parameter"]["Value"]
    DB_PASS = ssm.get_parameter(Name="/backend/banodoco/db/password")["Parameter"]["Value"]
    DB_HOST = ssm.get_parameter(Name="/backend/banodoco/db/host")["Parameter"]["Value"]
    DB_PORT = ssm.get_parameter(Name="/backend/banodoco/db/port")["Parameter"]["Value"]

    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": DB_NAME,
            "USER": DB_USER,
            "PASSWORD": DB_PASS,
            "HOST": DB_HOST,
            "PORT": DB_PORT,
        }
    }

INSTALLED_APPS = ("backend",)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

SECRET_KEY = "4e&6aw+(5&cg^_!05r(&7_#dghg_pdgopq(yk)xa^bog7j)^*j"
