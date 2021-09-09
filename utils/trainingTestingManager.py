import configparser
import json
import logging
import math

import numpy as np
import pandas as pd

from configparser import ConfigParser

from utils import directoryManager as dm, util
from utils.dataframeManager import load_dataframe_from_path

global training_files
training_files = None
global training_length
training_length = None

file = 'config.ini'
config = ConfigParser()
config.read(file)
ubm_percentage = config.getfloat("training_testing", "ubm_amount")
test_index = config.getfloat("training_testing", "testing_files")

try:
    training_length = json.loads(config.get("training_testing", "training_files"))
except configparser.NoOptionError:
    logging.info(f"No training_file amount specified")

try:
    training_length = json.loads(config.get("training_testing", "training_length"))
except configparser.NoOptionError:
    logging.info(f"No training_length specified")


def get_data_for_training(m_type, speaker_ids, feature_type, length=None, files=None):
    if length:
        global training_length
        training_length = float(length)
    if files:
        global training_files
        training_files = int(files)
    y = []
    if m_type == 'svm':
        features, y = get_svm_data_for_training(speaker_ids[0], feature_type)
    if m_type == 'gmm':
        features = get_gmm_data_for_training(speaker_ids[0], feature_type)
    if m_type.__contains__('gmm-ubm'):
        features = get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type)

    return np.asarray(features), y


def get_gmm_data_for_training(speaker_id, feature_type):
    wav_files = dm.get_wav_files(speaker_id)
    if training_length:
        training_features = np.asarray([])
        time_left = training_length
        for wav_file in wav_files:
            wav_file_path = rf'{speaker_id}/{wav_file}'
            features = retrieve_feature_vectors_from_file(wav_file_path, feature_type)
            features = np.asarray(features)
            for feature in features:
                if time_left >= 4:
                    if training_features.size == 0:
                        training_features = feature
                    else:
                        training_features = np.vstack((training_features, feature))
                    time_left -= 4
        return training_features
    else:
        if training_files:
            training_features = np.asarray(())
            for x in range(training_files):
                wav_file_path = rf'{speaker_id}/{wav_files[x]}'
                features = retrieve_feature_vectors_from_file(wav_file_path, feature_type)
                features = np.asarray(features)
                for feature in features:
                    if training_features.size == 0:
                        training_features = feature
                    else:
                        training_features = np.vstack((training_features, feature))
        return training_features
    return []


def get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type):
    training_features = np.asarray([])
    ubm_index = 0
    for speaker_id in speaker_ids:
        temp_features = np.asarray([])
        wav_files = dm.get_wav_files(speaker_id)
        if training_length:
            time_left = training_length * 2
            for wav_file in wav_files:
                wav_file_path = rf'{speaker_id}/{wav_file}'
                features = retrieve_feature_vectors_from_file(wav_file_path, feature_type)
                features = np.asarray(features)
                for feature in features:
                    if time_left >= 4:
                        if temp_features.size == 0:
                            temp_features = feature
                        else:
                            temp_features = np.vstack((temp_features, feature))
                        time_left -= 4
        else:
            if training_files:
                audio_files = training_files * 2
                for x in range(audio_files):
                    wav_file_path = rf'{speaker_id}/{wav_files[x]}'
                    features = retrieve_feature_vectors_from_file(wav_file_path, feature_type)
                    features = np.asarray(features)
                    for feature in features:
                        if temp_features.size == 0:
                            temp_features = feature
                        else:
                            temp_features = np.vstack((temp_features, feature))

        length = len(temp_features)
        if ubm_index == 0:
            ubm_index = math.floor(length * ubm_percentage)
            ubm_index = (ubm_index) - (ubm_index % 399)
        if m_type == 'gmm-ubm-ubm':
            temp_features = temp_features[:ubm_index]
        else:
            temp_features = temp_features[ubm_index:]
        if training_features.size == 0:
            training_features = temp_features
        else:
            training_features = np.vstack((training_features, temp_features))
    return training_features


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
    y_new = []
    training_features = np.asarray([])
    speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        if training_length:
            temp_features = np.asarray([])
            time_left = training_length
            for wav_file in wav_files:
                wav_file_path = rf'{id}/{wav_file}'
                features = retrieve_feature_vectors_from_file(wav_file_path, feature_type)
                features = np.asarray(features)
                is_speaker = 1 if id == speaker_id else 0
                for feature in features:
                    if time_left >= 4:
                        y_new += 399 * [is_speaker]
                        if temp_features.size == 0:
                            temp_features = feature
                        else:
                            temp_features = np.vstack((temp_features, feature))
                        time_left -= 4
        else:
            if training_files:
                temp_features = np.asarray(())
                for x in range(training_files):
                    wav_file_path = rf'{id}/{wav_files[x]}'
                    features = retrieve_feature_vectors_from_file(wav_file_path, feature_type)
                    features = np.asarray(features)
                    is_speaker = 1 if id == speaker_id else 0
                    y_new += len(features) * [is_speaker]
                    for feature in features:
                        if temp_features.size == 0:
                            temp_features = feature
                        else:
                            temp_features = np.vstack((temp_features, feature))

        if training_features.size == 0:
            training_features = temp_features
        else:
            training_features = np.vstack((training_features, temp_features))

    return training_features


def retrieve_feature_vectors_from_file(wav_file, feature_type):
    parts = wav_file.split('/')
    ending = parts[2].replace('.wav', '.json')
    file_path = rf'{dm.get_all_wav_path()}/{parts[0]}/{parts[1]}/{feature_type}/{ending}'
    dataframe = load_dataframe_from_path(file_path)
    features = dataframe.features[0]
    return features


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


def get_attack_files_and_extra_data(speaker_ids, replay_type):
    files = []
    for speaker_id in speaker_ids:
        attack_files = dm.get_attack_files(speaker_id)
        for attack_file in attack_files:
            if attack_file.__contains__(replay_type):
                attack_file = rf'{dm.get_all_wav_path()}/{speaker_id}/{attack_file}'
                files.append(attack_file)
    test_files = np.asarray(files)
    extra_data = [[test_files]]
    extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
    return test_files, extra_data_object
