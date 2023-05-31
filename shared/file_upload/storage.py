from banodoco_settings import SERVER, ServerType
from backend.constants import InternalFileType

def store_file(file_name,  file_type: InternalFileType, file=None, file_location=None):
    if SERVER == ServerType.PRODUCTION.value:
        # upload to s3
        pass
    else:
        if file_type == InternalFileType.IMAGE.value:
            if file:
                # save file locally
                pass
            elif file_location:
                pass
