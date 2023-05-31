import os
from dotenv import load_dotenv

from banodoco_settings import SERVER, ServerType
load_dotenv()

if SERVER == ServerType.DEVELOPMENT.value:
    DB_LOCATION = './banodoco-local.db'
else:
    # TODO: add ssm connection here
    pass


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': DB_LOCATION,
    }
}

INSTALLED_APPS = (
    'backend',
)
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

SECRET_KEY = '4e&6aw+(5&cg^_!05r(&7_#dghg_pdgopq(yk)xa^bog7j)^*j'