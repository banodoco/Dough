import sqlite3 as sl
from pathlib import Path

def setup_database():
    local_db_path = "banodoco-local.db"
    file = Path(local_db_path)

    if file.is_file():
        return
    
    print("setting up database")
    
    # connect to database
    con = sl.connect("banodoco-local.db")

    # creating tables
    with con:
        con.execute("""
            PRAGMA foreign_keys = ON;
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid CHAR(50) NOT NULL,
                name VARCHAR(255),
                email VARCHAR(255),
                password VARCHAR(255)
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid CHAR(50) NOT NULL,
                name VARCHAR(255),
                replicate_url VARCHAR(255)
            );
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS model_param_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                standard_param_key VARCHAR(255) NOT NULL,
                model_param_key VARCHAR(255) NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model(id)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                input_params TEXT NOT NULL,
                output_details TEXT NOT NULL,
                created_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                total_inference_time INT DEFAULT 0,
                FOREIGN KEY (model_id) REFERENCES model(id)
            );
        """)


class DBRepo:
    def __init__(self):
        pass

    # TODO: separate this in individual methods
    def insert_data_list(self, sql_query, data):
        con = sl.connect("banodoco-local.db")
        with con:
            con.executemany(sql_query, data)
        
        con.close()

    def fetch_data_list(self, sql_query):
        con = sl.connect("banodoco-local.db")
        data = None
        with con:
            data = con.execute(sql_query)
        
        con.close()
        return data


    