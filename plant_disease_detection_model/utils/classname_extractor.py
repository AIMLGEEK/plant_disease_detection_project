import json

def get_class_names_from_directory(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
