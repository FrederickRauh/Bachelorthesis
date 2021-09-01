import math

import numpy as np
import pandas as pd

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, util
from utils.dataframeManager import load_dataframe_from_path

file = 'config.ini'
config = ConfigParser()
config.read(file)
training_index = config.getfloat("training_testing", "training_files")
test_index = config.getfloat("training_testing", "testing_files")


def get_percentage_of_audio(speaker_id, wav_files, percentage):
    min_length = (am.get_length_of_minimal() * percentage)
    x = min_length % 4
    min_length -= x
    files = []
    time_used = 0
    for wav_file in wav_files:
        if min_length > time_used:
            file = wav_file.replace('\\', '/')
            parts = file.split('/')
            ending = parts[1]
            file_path = rf'{speaker_id}/{parts[0]}/{ending}'
            path = rf'{dm.get_all_wav_path()}/{file_path}'
            length_of_file = am.get_audio_length(path)
            modulo = length_of_file % 4
            length_of_file -= modulo
            time_used += length_of_file
            files.append(file_path)

    return files


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
        if training_index < 1:
            files = get_percentage_of_audio(id, wav_files, training_index)
            t = files
            duration = 0
            for file in t:
                file = rf'{dm.get_all_wav_path()}/{file}'
                duration += am.get_audio_length(file)

        else:
            adjusted_index = int(training_index)
            wav_files = wav_files[:adjusted_index]
            for wav_file in wav_files:
                file = rf'{id}/{wav_file}'
                t.append(file)
    if training_index < 1:
        return get_percentage_of_training_files(t, feature_type, training_index)
    else:
        return get_training_files(t, feature_type)


def get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type):
    t = []
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        # get percentage of test files
        if training_index < 1:
            temp = get_percentage_of_audio(id, wav_files, training_index)
            temp_vectors = get_percentage_of_training_files(temp, feature_type, training_index)
            length = len(temp_vectors)
            ubm_index = int(math.floor(length * 0.1))
            if m_type == 'gmm-ubm-ubm':
                temp_files = temp_vectors[:ubm_index]
            else:
                temp_files = temp_vectors[ubm_index:]
            for vector in temp_files:
                t.append(vector)
        else:
            adjusted_index = int(training_index)
            wav_files = wav_files[:adjusted_index]
            # divide into 2 parts, 10% of files for ubm, 90% for gmm
            length = float(len(wav_files))
            ubm_index = (length / 10)
            if m_type == 'gmm-ubm-ubm':
                wav_files = wav_files[:int(ubm_index)]
            else:
                wav_files = wav_files[int(ubm_index):]
            for wav_file in wav_files:
                file = rf'{id}/{wav_file}'
                t.append(file)
    if training_index < 1:
        return t
    else:
        return get_training_files(t, feature_type)


def get_training_files(t, feature_type):
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
        for feature in features:
            feature = np.asarray(feature)
            for vector in feature:
                if training_features.size == 0:
                    training_features = vector
                else:
                    training_features = np.vstack((training_features, vector))
    # return util.get_correct_array_form(training_files)
    return training_features


def get_percentage_of_training_files(files, feature_type, percentage):
    training_features = np.asarray(())

    min_length = (am.get_length_of_minimal() * percentage)
    x = min_length % 4
    min_length = (min_length - x) / 4 * 399

    t = 0
    for element in files:
        element = element.replace('\\', '/')
        parts = element.split('/')
        ending = parts[2].replace('.wav', '.json')
        file_path = rf'{parts[0]}/{parts[1]}/{feature_type}/{ending}'
        path = rf'{dm.get_all_wav_path()}/{file_path}'
        file_features = load_dataframe_from_path(path)
        features = file_features.features[0]
        features = np.asarray(features)
        t += len(features)
        for feature in features:
            feature = np.asarray(feature)
            if min_length > 0:
                for vector in feature:
                    if training_features.size == 0:
                        training_features = vector
                    else:
                        training_features = np.vstack((training_features, vector))
            min_length -= 399
    # return util.get_correct_array_form(training_files)
    return training_features


def get_svm_data_for_training(speaker_id, feature_type):
    total_files = []
    y = []
    y_new = []
    speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        # only get at max 20 files from Speaker, to avoid to much training data
        if training_index < 1:
            adjusted_index = int(util.get_percent_index(len(wav_files), training_index, True))
        else:
            adjusted_index = int(training_index)
        wav_files = wav_files[:adjusted_index]

        for wav_file in wav_files:
            file = rf'{id}/{wav_file}'
            total_files.append(file)
            is_speaker = 1 if id == speaker_id else 0
            y.append(is_speaker)

    training_features = []

    for x in range(len(total_files)):
        element = total_files[x]
        element = element.replace('\\', '/')
        parts = element.split('/')
        ending = parts[2].replace('.wav', '.json')
        file_path = rf'{parts[0]}/{parts[1]}/{feature_type}/{ending}'
        path = rf'{dm.get_all_wav_path()}/{file_path}'
        file_features = load_dataframe_from_path(path)
        features = file_features.features[0]
        features = np.array(features)
        for feature in features:
            for i in range(399):
                y_new.append(y[x])
            feature = np.array(feature)
            for vector in feature:
                training_features.append(vector)

    return training_features, y_new


def get_test_files_and_extra_data(speaker_ids):
    test_files = load_test_files(speaker_ids)
    extra_data = [[test_files]]
    extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
    return test_files, extra_data_object


def load_test_files(speaker_ids):
    files = []
    for speaker_id in speaker_ids:
        x_files = dm.get_wav_files(speaker_id)
        if test_index < 1:
            adjusted_index = int(util.get_percent_index(len(x_files), test_index, False))
        else:
            adjusted_index = int(test_index * (-1))
        wav_files = x_files[adjusted_index:]

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
