import os
import json

import pandas as pd

from utils import directoryManager as dm


def create_empty_json(json_path):
    dm.create_feature_json_dir(json_path)
    with open(json_path, "w", newline='') as file:
        pass


def find_feature_json(json_path):
    if not os.path.isfile(json_path):
        create_empty_json(json_path)
    return json_path


def write_features_to_json_file(json_path, wav_path, features):
    json_path = json_path.replace('.wav', '.json')
    find_feature_json(json_path)
    entry = []
    entry.append([wav_path, features, len(features)])
    json_file = pd.DataFrame(entry, columns=['wav_path', 'features', 'feature count'])
    dm.check_if_file_exists_then_remove(json_path)
    json_file.to_json(json_path)
    print(json_path)