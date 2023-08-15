# streamlit state constants
from utils.enum import ExtendedEnum


LOGGED_USER = 'logged_user'
AUTH_TOKEN = 'auth_details'

class ImageStage(ExtendedEnum):
    SOURCE_IMAGE = 'Source Image'
    MAIN_VARIANT = 'Main Variant'
    NONE = 'None'