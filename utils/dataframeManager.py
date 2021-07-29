import os
import json
import logging

import pandas as pd

from utils import directoryManager as dm
from config import SYSTEM

"""
DataframeManager is used when working with a dataframe/json file containing information for the training / predicting phase.
This file shall only contain methods that create, load, save dataframes to json as also output content of said files.
"""

"""
    create_librosa_dataframe (creates the dataframe which links all wav files to the corresponding json files containing 
    their features)
"""
def create_librosa_dataframe(speaker_ids):
    logging.info("creating librosa dataframe... ")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_name = speaker_id + '\\' + file
            all_features.append([speaker_id, file_name])
    features_dataframe = pd.DataFrame(all_features, columns=['speaker_id', 'file_name'])
    dataframe_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
    save_dataframe_to_json_file(features_dataframe, dataframe_path)
    return features_dataframe


"""
    create_psf_dataframe (creates the dataframe which links all wav files to the corresponding json files containing     
    their features)
"""
def create_psf_dataframe(speaker_ids):
    logging.info("creating psf dataframe... ")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_name = speaker_id + '\\' + file
            all_features.append([speaker_id, file_name])
    features_dataframe = pd.DataFrame(all_features, columns=['speaker_id', 'file_name'])
    dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
    save_dataframe_to_json_file(features_dataframe, dataframe_path)
    return features_dataframe

def create_feature_json(json_path, rows):
    dm.create_feature_json_dir(json_path)
    with open(json_path, 'w', newline='') as file:
        writer = json.writer(file)
        writer.writerow(["file_name", "features"])
        writer.writerows(rows)


def find_feature_json(json_path):
    if not os.path.isfile(json_path):
        create_feature_json(json_path, [])
    return json_path


def write_features_to_json_file(json_path, wav_path, features):
    json_path = json_path.replace('.wav', '.json')
    find_feature_json(json_path)
    entry = []
    entry.append([wav_path, features, len(features)])
    json_file = pd.DataFrame(entry, columns=['wav_path', 'features', 'feature count'])
    dm.check_if_file_exists_then_remove(json_path)
    json_file.to_json(json_path)


def save_dataframe_to_json_file(dataframe, path):
    dm.check_if_file_exists_then_remove(path)
    dataframe.to_json(path)


def load_dataframe_from_path(path):
    dataframe = pd.read_json(path)
    return dataframe


def get_data_for_training(m_type, speaker_ids):
    return get_svm_data_for_training(speaker_ids[0]) if m_type == 'svm' else get_gmm_data_for_training(speaker_ids)


def get_gmm_data_for_training(speaker_ids):
    t = []
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)
    return get_training_files(t, SYSTEM.FEATURE_TYPE)


def get_svm_data_for_training(speaker_id):
    t = []
    y = []
    speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)
            is_speaker = 1 if id == speaker_id else 0
            y.append(is_speaker)
    return get_training_files(t, SYSTEM.FEATURE_TYPE), y


def get_training_files(t, f_type):
    print("F_TYPE: ", f_type)
    dataframe_path = dm.get_all_data_path() + '\\' + f_type + '-dataframe.json'
    dataframe = load_dataframe_from_path(dataframe_path)
    training_files = []
    for element in t:
        parts = dataframe.loc[dataframe['file_name'] == element].file_name.array[0].split('\\')
        file_path = parts[0] + '\\' + parts[1] + '\\' + f_type + '\\' + parts[2].replace('.wav', '.json')
        path = dm.get_all_wav_path() + '\\' + file_path
        file_features = load_dataframe_from_path(path)
        features = file_features.features[0]
        training_files.append(features)

    return training_files
    # return util.get_correct_array_form(training_files)
