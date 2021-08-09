import numpy as np

from utils import directoryManager as dm, util
from config import CONFIG as config
from utils.dataframeManager import load_dataframe_from_path


def get_data_for_training(m_type, speaker_ids, feature_type=config.FEATURE_TYPE):
    y = []
    if m_type == 'svm':
        files, y = get_svm_data_for_training(speaker_ids[0], feature_type)
    if m_type == 'gmm':
        files = get_gmm_data_for_training(speaker_ids, feature_type)
    if m_type.__contains__('gmm-ubm'):
        files = get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type)
    return np.asarray(files), y


def get_gmm_data_for_training(speaker_ids, feature_type=config.FEATURE_TYPE):
    t = []
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        wav_files = wav_files[:len(wav_files) - 10]
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)

    return get_training_files(t, feature_type)


def get_gmm_ubm_data_for_training(speaker_ids, m_type, feature_type=config.FEATURE_TYPE):
    t = []
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        wav_files = wav_files[:len(wav_files) - 10]
        if m_type == 'gmm-ubm-ubm':
            files = util.split_array_for_multiprocess(wav_files, 2)
            wav_files = files[0]
        else:
            files = util.split_array_for_multiprocess(wav_files, 2)
            wav_files = files[1]
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)

    return get_training_files(t, feature_type)


def get_svm_data_for_training(speaker_id, feature_type=config.FEATURE_TYPE):
    t = []
    y = []
    speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        wav_files = wav_files[:len(wav_files) - 10]
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)
            is_speaker = 1 if id == speaker_id else 0
            y.append(is_speaker)
    return get_training_files(t, feature_type), y


def get_training_files(t, feature_type=config.FEATURE_TYPE):
    training_files = []
    for element in t:
        parts = element.split('\\')
        file_path = parts[0] + '\\' + parts[1] + '\\' + feature_type + '\\' + parts[2].replace('.wav', '.json')
        path = dm.get_all_wav_path() + '\\' + file_path
        file_features = load_dataframe_from_path(path)
        features = file_features.features[0]
        training_files.append(features)
    # if feature_type == 'psf':
    #     training_files = util.get_correct_array_form(training_files)
    # return util.get_correct_array_form(training_files)
    return training_files


def load_test_files(speaker_ids):
    files = []
    for speaker_id in speaker_ids:
        if config.LOCAL:
            dir = dm.get_test_subfolders(speaker_id)
            for dir_path in dir:
                files_path = dm.get_test_path() + '\\' + speaker_id + '\\' + dir_path
                wav_files = dm.get_wav_files_in_folder(files_path)
                for x in range(len(wav_files)):
                    files.append(wav_files[x])
        else:
            wav_files = dm.get_wav_files(speaker_id)[-10:]
            for wav_file in wav_files:
                wav_file = dm.get_all_wav_path() + '\\' + speaker_id + '\\' + wav_file
                files.append(wav_file)
    return np.asarray(files)