import json
import os

def is_file_present(filename):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_directory, filename)
    return os.path.isfile(file_path)

# def get_current_user():
#     logger = AppLogger()
#     data_store = 'data.json'

#     if not is_file_present(data_store):
#         with open(data_store, 'w') as file:
#             json.dump({}, file, indent=4)


#     data = {}
#     try:
#         with open(data_store, 'r') as file:
#             data = json.loads(file.read())
#     except Exception as e:
#         logger.log(LoggingType.ERROR, 'user not found in local storage')

    
#     if not ( data and 'current_user' in data):
#         from utils.data_repo.data_repo import DataRepo
#         data_repo = DataRepo()
#         user = data_repo.get_first_active_user()
#         data = {}
#         data['current_user'] = user.to_json() if user else None
        
#     with open(data_store, 'w') as file:
#         json.dump(data, file, indent=4)

#     with open(data_store, 'r') as file:
#         data = json.loads(file.read())

#     logger.log(LoggingType.DEBUG, 'user found in local storage' + str(data))
#     return json.loads(data['current_user']) if data['current_user'] else None

# def get_current_user_uuid():
#     current_user = get_current_user()
#     if current_user and 'uuid' in current_user:
#         return current_user['uuid']
#     else: 
#         return None