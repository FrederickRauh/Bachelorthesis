import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm
from utils import util


def create_librosa_dataframe(speaker_ids):
    print("creating librosa dataframe... ")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            features = flib.load_features_from_json(file_path)
            file_name = speaker_id + '\\' + file
            all_features.append([features, speaker_id, file_name])
    features_dataframe = pd.DataFrame(all_features, columns=['feature', 'speaker_id', 'file_name'])
    dataframe_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
    save_dataframe_to_json_file(features_dataframe, dataframe_path)
    return features_dataframe


def create_psf_dataframe(speaker_ids):
    print("creating psf dataframe... ")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            features = fpsf.load_features_from_json(file_path)
            file_name = speaker_id + '\\' + file
            all_features.append([features, speaker_id, file_name])

    features_dataframe = pd.DataFrame(all_features, columns=['feature', 'speaker_id', 'file_name'])
    dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
    save_dataframe_to_json_file(features_dataframe, dataframe_path)
    return features_dataframe


def save_dataframe_to_csv_file(dataframe):
    dataframe_path = dm.get_all_data_path() + '\\' + 'dataframe.csv'
    dm.check_if_file_exists_then_remove(dataframe_path)
    dataframe.to_csv(dataframe_path)


def save_dataframe_to_json_file(dataframe, path):
    dm.check_if_file_exists_then_remove(path)
    dataframe.to_json(path)


def load_dataframe():
    dataframe_path = dm.get_all_data_path() + '\\' + 'dataframe.csv'
    return pd.read_csv(dataframe_path)


def load_dataframe_from_path(path):
    return pd.read_json(path)


def get_data_for_training_from_dataframe(type, speaker_id, dataframe):
    t = []
    y = []
    speaker_ids = [speaker_id]
    if type == 'svm':
        speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)
            if type == 'svm':
                is_speaker = 0
                if id == speaker_id:
                    is_speaker = 1
                y.append(is_speaker)
    if type == 'svm':
        return get_training_files(dataframe, t), y
    return get_training_files(dataframe, t)


def get_training_files(dataframe, t):
    training_files = []
    for element in t:
        training_features = dataframe.loc[dataframe['file_name'] == element].feature.array[0]['0']
        training_files.append(training_features)
    return util.get_correct_array_form(training_files)