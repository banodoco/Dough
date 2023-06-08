import os
from pathlib import Path
import sys

sys.path.append('../')


from dotenv import load_dotenv

from shared.constants import LOCAL_DATABASE_NAME, SERVER, ServerType


load_dotenv()

if SERVER != ServerType.DEVELOPMENT.value:
    DB_LOCATION = LOCAL_DATABASE_NAME
else:
    DB_LOCATION = ''

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': DB_LOCATION,
    }
}

INSTALLED_APPS = ('backend',)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

SECRET_KEY = '4e&6aw+(5&cg^_!05r(&7_#dghg_pdgopq(yk)xa^bog7j)^*j'