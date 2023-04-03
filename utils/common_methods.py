from pathlib import Path
import os

# creates a file path if it's not already present
def create_file_path(path):
    if not path:
        return
    
    file = Path(path)
    if not file.is_file():
        last_slash_index = path.rfind('/')
        if last_slash_index != -1:
            directory_path = path[:last_slash_index]
            file_name = path[last_slash_index + 1:]
            
            # creating directory if not present
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            
            # creating file
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'w') as f:
                pass
