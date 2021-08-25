import numpy as np
import pandas as pd

from configparser import ConfigParser

from utils import directoryManager as dm, util
from utils.dataframeManager import load_dataframe_from_path

file = 'config.ini'
config = ConfigParser()
config.read(file)
index = config.getint("system", "training_files")

def get_data_for_training(m_type, speaker_ids, feature_type):
    y = []
    if m_type == 'svm':
        files, y = get_svm_data_for_training(speaker_ids[0], feature_type)
    if m_type == 'gmm':
        files = get_gmm_data_for_training(speaker_ids, feature_type)
    if m_type.__contains__('gmm-ubm'):
        files = get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type)

    return np.asarray(files), y


def get_gmm_data_for_training(speaker_ids, feature_type):
    t = []
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        if index > 0:
            adjusted_index = index * 6
            wav_files = wav_files[:int(adjusted_index)]
        for wav_file in wav_files:
            file = rf'{id}/{wav_file}'
            t.append(file)

    return get_training_files(t, feature_type)


def get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type):
    t = []
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        if index > 0:
            adjusted_index = index * 10
            ubm_index = (int(adjusted_index) / 10)
            wav_files = wav_files[:int(adjusted_index)]
        if m_type == 'gmm-ubm-ubm':
            wav_files = wav_files[:int(ubm_index)]
        else:
            wav_files = wav_files[int(ubm_index):]
        for wav_file in wav_files:
            file = rf'{id}/{wav_file}'
            t.append(file)
    return get_training_files(t, feature_type)


def get_svm_data_for_training(speaker_id, feature_type):
    t = []
    y = []
    y_new = []
    speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        # only get at max 20 files from Speaker, to avoid to much training data
        if index > 0:
            wav_files = wav_files[:int(index)]
        for wav_file in wav_files:
            file = rf'{id}/{wav_file}'
            t.append(file)
            is_speaker = 1 if id == speaker_id else 0
            for x in range(399):
                y_new.append(is_speaker)
            # y.append(is_speaker)

    training_files = []
    training_features = []

    for element in t:
        element = element.replace('\\', '/')
        parts = element.split('/')
        ending = parts[2].replace('.wav', '.json')
        file_path = rf'{parts[0]}/{parts[1]}/{feature_type}/{ending}'
        path = rf'{dm.get_all_wav_path()}/{file_path}'
        file_features = load_dataframe_from_path(path)
        features = file_features.features[0]
        features = np.array(features)
        for vector in features:
            training_features.append(vector)

    return training_features, y_new


def get_training_files(t, feature_type):
    # training_files = []
    length_of_file = 0

    training_features = np.asarray(())

    for element in t:
        element = element.replace('\\', '/')
        parts = element.split('/')
        ending = parts[2].replace('.wav', '.json')
        file_path = rf'{parts[0]}/{parts[1]}/{feature_type}/{ending}'
        path = rf'{dm.get_all_wav_path()}/{file_path}'
        file_features = load_dataframe_from_path(path)
        features = file_features.features[0]
        features = np.asarray(features)

        # training_files.append(features)

        for vector in features:
            if training_features.size == 0:
                training_features = vector
            else:
                training_features = np.vstack((training_features, vector))
    # return util.get_correct_array_form(training_files)
    return training_features


def get_test_files_and_extra_data(speaker_ids):
    test_files = load_test_files(speaker_ids)
    extra_data = [[test_files]]
    extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
    return test_files, extra_data_object


def load_test_files(speaker_ids):
    index = config.getint("system", 'testing_files') * (-1)
    files = []
    for speaker_id in speaker_ids:
        wav_files = dm.get_wav_files(speaker_id)[index:]
        for wav_file in wav_files:
            wav_file = rf'{dm.get_all_wav_path()}/{speaker_id}/{wav_file}'
            files.append(wav_file)
    return np.asarray(files)


def get_attack_files_and_extra_data(speaker_ids):
    files = []
    for speaker_id in speaker_ids:
        attack_files = dm.get_attack_files(speaker_id)
        for attack_file in attack_files:
            attack_file = rf'{dm.get_all_wav_path()}/{speaker_id}/{attack_file}'
            files.append(attack_file)
    test_files = np.asarray(files)
    extra_data = [[test_files]]
    extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
    return test_files, extra_data_object