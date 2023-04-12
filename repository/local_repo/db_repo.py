from dataclasses import dataclass
import sqlite3 as sl
from typing import List
import uuid
from pathlib import Path

LOCAL_DB_FILENAME = 'banodoco-local.db'
MODEL_TABLE = 'model'
USER_TABLE = 'user'
INFERENCE_LOG_TABLE = 'inference_log'
MODEL_PARAM_MAP_TABLE = 'model_param_map'


def setup_database():
    local_db_path = LOCAL_DB_FILENAME
    file = Path(local_db_path)

    if file.is_file():
        return
    
    print("setting up database")
    
    # connect to database
    con = sl.connect(LOCAL_DB_FILENAME)

    # creating tables
    with con:
        con.execute("""
            PRAGMA foreign_keys = ON;
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS user (
                uuid VARCHAR(50) NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255),
                email VARCHAR(255),
                password VARCHAR(255)
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS model (
                uuid VARCHAR(50) NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255),
                version VARCHAR(255),
                replicate_url VARCHAR(255)
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS model_param_map (
                uuid VARCHAR(50) NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                standard_param_key VARCHAR(255) NOT NULL,
                model_param_key VARCHAR(255) NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model(id)
            );
        """)

        '''
            user_id can be null initially
        '''
        con.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                uuid VARCHAR(50) NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                user_id INTEGER,
                input_params TEXT,
                output_details TEXT,
                created_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                total_inference_time INT DEFAULT 0,
                FOREIGN KEY (model_id) REFERENCES model(id)
                FOREIGN KEY (user_id) REFERENCES user(id)
            );
        """)

@dataclass
class Model:
    uuid: str
    id: int
    name: str
    version: str
    replicate_url: str

class DBRepo:
    def __init__(self):
        self.conn = sl.connect(LOCAL_DB_FILENAME)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def insert_data(self, table, **kwargs):
        # adding uuid
        kwargs['uuid'] = str(uuid.uuid4())

        columns = ', '.join(kwargs.keys())
        values = tuple(kwargs.values())
        placeholders = ', '.join(['?'] * len(values))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(query, values)
        self.conn.commit()

    def insert_multiple_data(self, sql_query, data_list):
        self.cursor.executemany(sql_query, data_list)
        self.conn.commit()

    def fetch_all_data_from_query(self, sql_query):
        self.cursor.execute(sql_query)
        return self.cursor.fetchall()
    
    def fetch_first_data_from_query(self, sql_query):
        self.cursor.execute(sql_query)
        return self.cursor.fetchone()
    
    def fetch_all_model_list_by_name_and_version(self, model_name, model_version) -> List[Model]:
        query = f"SELECT * FROM model WHERE name = '{model_name}' AND version = '{model_version}'"
        model_list = self.fetch_all_data_from_query(query)
        res = []
        for model in model_list:
            res.append(Model(uuid=model[0], id=model[1], name=model[2], version=model[3], replicate_url=model[4]))
        return res
    
    def fetch_first_model_by_name_and_version(self, model_name, model_version) -> Model:
        query = f"SELECT * FROM model WHERE name = '{model_name}' AND version = '{model_version}'"
        model = self.fetch_first_data_from_query(query)
        if model:
            model = Model(uuid=model[0], id=model[1], name=model[2], version=model[3], replicate_url=model[4])
        
        return model

    def log_inference_data_in_local_db(self, payload):
        '''
        payload = {
            'model_name': model.name,
            'model_version': model.version,
            'total_inference_time': time_taken,
            'input_params': data_str,
            'created_on': int(time.time())
        }
        '''

        user_id = None
        model = self.fetch_first_model_by_name_and_version(payload['model_name'], payload['model_version'])
        if not model:
            # TODO: Add replicate url also
            model_data = {
                'name': payload['model_name'],
                'version': payload['model_version']
            }

            self.insert_data(MODEL_TABLE, **model_data)
            model = self.fetch_first_model_by_name_and_version(payload['model_name'], payload['model_version'])
            model_id = model.id
        else:
            model_id = model.id
        
        log_data = {
            'model_id': model_id,
            'user_id': user_id,
            'input_params': payload['input_params'],
            'output_details': None,
            'total_inference_time': round(payload['total_inference_time'], 2)
        }

        self.insert_data(INFERENCE_LOG_TABLE, **log_data)
