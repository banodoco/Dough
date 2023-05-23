import os

DB_LOCATION = '../banodoc-local.db'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': DB_LOCATION,
    }
}

INSTALLED_APPS = (
    'database',
)
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

SECRET_KEY = '4e&6aw+(5&cg^_!05r(&7_#dghg_pdgopq(yk)xa^bog7j)^*j'