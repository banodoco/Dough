import json
import os


MOTION_LORA_DB = 'data.json'

def is_file_present(filename):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, filename)
    return os.path.isfile(file_path)

def write_to_motion_lora_local_db(update_data):
    data_store = MOTION_LORA_DB
            
    data = {}
    if os.path.exists(data_store):
        try:
            with open(data_store, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
        except Exception as e:
            pass
    
    for key, value in update_data.items():
        data[key] = value
    
    data = json.dumps(data, indent=4)
    with open(data_store, 'w', encoding='utf-8') as file:
        file.write(data)
        
def read_from_motion_lora_local_db(key = None):
    data_store = MOTION_LORA_DB
    
    data = {}
    if os.path.exists(data_store):
        with open(data_store, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
            
    return data[key] if key in data else data