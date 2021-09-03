import logging
import math
import random

import numpy as np
import pandas as pd

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, util
from utils.dataframeManager import load_dataframe_from_path

file = 'config.ini'
config = ConfigParser()
config.read(file)
training_index = config.getfloat("training_testing", "training_files")
ubm_percentage = config.getfloat("training_testing", "ubm_amount")
test_index = config.getfloat("training_testing", "testing_files")
equal_percentage = config.getboolean("training_testing", "equal_percentage")
min_amount = 0
if equal_percentage and training_index < 1:
    min_amount = am.get_length_of_least_audio()


def get_percentage_of_audio_signal(speaker_id, wav_files, length, feature_type):
    temp = 0
    temp_files = []
    for wav_file in wav_files:
        if temp < length:
            file_path = rf'{dm.get_all_wav_path()}/{speaker_id}/{wav_file}'
            length_of_file = am.get_audio_length(file_path)
            temp += (length_of_file - (length_of_file % 4))
            temp_files.append(file_path)

    training_features = np.asarray(())

    vector_amount = length / 4 * 399
    for temp_file in temp_files:
        temp_file = temp_file.replace('\\', '/').replace('.wav', '.json')
        parts = temp_file.split('/')
        temp_path = ''
        for part in parts:
            if not part == parts[len(parts) - 1]:
                temp_path += rf'{part}/'
            else:
                temp_path += rf'{feature_type}/{part}'
        features = load_dataframe_from_path(temp_path)
        features = np.asarray(features.features[0])
        for feature in features:
            if vector_amount > 0:
                if training_features.size == 0:
                    training_features = feature
                else:
                    training_features = np.vstack((training_features, feature))
                vector_amount -= len(feature)

    return training_features


def get_equal_percentage(speaker_id, percentage, feature_type):
    # min_amount is in seconds
    min_per = min_amount * percentage
    length = min_per - (min_per % 4)
    return get_percentage_of_audio_signal(speaker_id, dm.get_wav_files(speaker_id), length, feature_type)


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
            if equal_percentage:
                return get_equal_percentage(id, training_index, feature_type)
            else:
                adjusted_index = int(util.get_percent_index(len(wav_files), training_index, True))
        else:
            adjusted_index = int(training_index)

        wav_files = wav_files[:adjusted_index]
        for wav_file in wav_files:
            file = rf'{id}/{wav_file}'
            t.append(file)

    return get_training_files(t, feature_type)


def get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type):
    t = []
    if training_index < 1 and equal_percentage:
        training_features = np.asarray(())
        ubm_index = 0
        for id in speaker_ids:
            temp_ = get_equal_percentage(id, training_index, feature_type)
            length = len(temp_)
            if ubm_index == 0:
                ubm_index = (math.ceil(math.floor(length * ubm_percentage) / 399) * 399) - 1
            # random.shuffle(temp_)
            if m_type == 'gmm-ubm-ubm':
                temp_ = temp_[:ubm_index]
            else:
                temp_ = temp_[ubm_index:]
            if training_features.size == 0:
                training_features = temp_
            else:
                training_features = np.vstack((training_features, temp_))
        return training_features
    else:
        for id in speaker_ids:
            wav_files = dm.get_wav_files(id)
            # get percentage of test files
            if training_index < 1:
                adjusted_index = int(util.get_percent_index(len(wav_files), training_index, True))
            else:
                adjusted_index = int(training_index)
            wav_files = wav_files[:adjusted_index]
            # divide into 2 parts, 10% of files for ubm, 90% for gmm
            ubm_index = (float(len(wav_files)) * ubm_percentage)
            if m_type == 'gmm-ubm-ubm':
                wav_files = wav_files[:int(ubm_index)]
            else:
                wav_files = wav_files[int(ubm_index):]
            for wav_file in wav_files:
                file = rf'{id}/{wav_file}'
                t.append(file)
        return get_training_files(t, feature_type)


def get_training_files(t, feature_type):
    duration = 0
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
        duration += (len(features) * 4)
        for feature in features:
            # feature = np.asarray(feature)
            # for vector in feature:
            if training_features.size == 0:
                training_features = feature
            else:
                training_features = np.vstack((training_features, feature))
    # return util.get_correct_array_form(training_files)
    return training_features


def get_svm_data_for_training(speaker_id, feature_type):
    total_files = []
    y = []
    y_new = []
    speaker_ids = dm.get_all_ids()
    if training_index < 1 and equal_percentage:
        training_features = np.asarray(())
        for id in speaker_ids:
            if training_index < 1:
                if equal_percentage:
                    features = get_equal_percentage(id, training_index, feature_type)
                    is_speaker = 1 if id == speaker_id else 0
                    y_new += len(features) * [is_speaker]
                    features = np.asarray(([features]))
                    for feature in features:
                        if training_features.size == 0:
                            training_features = feature
                        else:
                            training_features = np.vstack((training_features, feature))
        return training_features, y_new


    else:
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

        training_features = np.asarray(())

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
                y_new += [y[x]] * 399

                feature = np.array(feature)
                if training_features.size == 0:
                    training_features = feature
                else:
                    training_features = np.vstack((training_features, feature))

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
