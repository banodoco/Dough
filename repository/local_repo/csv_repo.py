import pandas as pd
import csv

from utils.common_methods import create_file_path

class CSVProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        create_file_path(self.file_path)
    
    # returns the entire csv file data in json format
    def get_all_json_data(self, na_filter=False):
        res = {}
        data = pd.read_csv(self.file_path, na_filter=na_filter)

        for i, row in data.iterrows():
            res[row['key']] = row['value']
        return res
    
    # updates a single key value pair
    def update_csv_data(self, key, value):
        with open(self.file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)

            rows = []
            for row in csv_reader:
                if row[0] == key:
                    row_number = csv_reader.line_num - 2
                    new_value = value

        df = pd.read_csv(self.file_path)
        df.iat[row_number, 1] = new_value
        df.to_csv(self.file_path, index=False)

    # clear the entire csv file
    def clear_all_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(df.index[0:])
        df.to_csv(self.file_path, index=False)
    
    # TODO: create a separate interface for this later
    def update_specific_timing_value(self, index_of_current_item, parameter, value):
        df = pd.read_csv(self.file_path)
        
        try:
            col_index = df.columns.get_loc(parameter)
        except KeyError:
            raise ValueError(f"Invalid parameter: {parameter}")
        
        df.iloc[index_of_current_item, col_index] = value
        numeric_cols = ["primary_image", "seed", "num_inference_steps"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
            df[col].fillna(0, inplace=True)
            df[col] = df[col].astype(int)
        
        df.to_csv(self.file_path, index=False)


def get_project_settings(project_name):
    csv_client = CSVProcessor(f'videos/{project_name}/settings.csv')
    return csv_client.get_all_json_data()

def update_project_setting(key, value, project_name):
    csv_client = CSVProcessor(f'videos/{project_name}/settings.csv')
    csv_client.update_csv_data(key, value)

def remove_existing_timing(project_name):
    csv_client = CSVProcessor("videos/" + str(project_name) + "/timings.csv")
    csv_client.clear_all_data()

def get_app_settings():
    csv_client = CSVProcessor("app_settings.csv")
    return csv_client.get_all_json_data()

def update_app_settings(key, value):
    csv_client = CSVProcessor("app_settings.csv")
    csv_client.update_csv_data(key, value)

def update_specific_timing_value(project_name, index_of_current_item, parameter, value):
    csv_client = CSVProcessor(f"videos/{project_name}/timings.csv")
    csv_client.update_specific_timing_value(index_of_current_item, parameter, value)
