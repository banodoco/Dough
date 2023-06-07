import json
import os

def is_file_present(filename):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, filename)
    return os.path.isfile(file_path)

def get_current_user():
    data_store = 'data.json'
    # check if the local storage json is present
    if not is_file_present(data_store):
        with open(data_store, 'w') as file:
            json.dump({}, file, indent=4)

    # if current user is not set then pick the first user in the db
    with open(data_store, 'r') as file:
        data = json.load(file)
    
    if 'current_user' not in data:
        from utils.data_repo.data_repo import DataRepo
        data_repo = DataRepo()
        data['current_user'] = data_repo.get_first_active_user()
        
    with open(data_store, 'w') as file:
        json.dump(data, file, indent=4)

    return data_store['curent_user']

def get_current_user_uuid():
    current_user = get_current_user()
    if current_user and 'uuid' in current_user:
        return current_user['uuid']
    else: 
        return None