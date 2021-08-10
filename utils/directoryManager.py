import logging
import os
from configparser import ConfigParser

from os.path import join as pjoin


"""
This file contains all methods used to create dirs, read content von dirs, direct file accessing methods should not be 
contained in this file
"""
def get_project_path():
    return os.getcwd()

# features
def get_feature_path(wav_path, version):
    sub_path = wav_path.split('\\')
    feature_path = ''
    for x in range(len(sub_path)):
        if x == (len(sub_path) - 1):
            feature_path = feature_path + '\\' + version + '\\' + sub_path[x]
        else:
            feature_path = feature_path + sub_path[x] + '\\'
    return feature_path


def get_feature_librosa_json_path(wav_path):
    json_path = get_feature_path(wav_path, 'librosa')
    return json_path.replace('.wav', '.json')


def get_feature_psf_json_path(wav_path):
    json_path = get_feature_path(wav_path, 'psf')
    return json_path.replace('.wav', '.json')


def create_feature_json_dir(file_path):
    sub_path = file_path.split('\\')
    new_dir_path = ''
    for x in range(len(sub_path)):
        if not x == (len(sub_path) - 1):
            new_dir_path = new_dir_path + sub_path[x] + '\\'
    make_dir(new_dir_path)


# folder structure
def get_parent_path(speaker_id):
    parent_path = make_dir(get_all_wav_path())
    return make_dir(os.path.join(parent_path, speaker_id))


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as error:
            logging.error("Creating directory %s has failed. Error %s" % (path, error))
    return path


def list_sub_folders(parent_path):
    return os.listdir(parent_path)


def get_sub_folder_path(parent_path, sub_folder):
    path = os.path.join(parent_path, sub_folder)
    if not os.path.exists(path):
        make_dir(path)
    return path


# results
def get_results_folder(model_type):
    data_path = make_dir(get_all_data_path() + '\\' + "result")
    return make_dir(data_path + '\\' + model_type)


# wav
def create_wav_file_name(speaker_id, number):
    number = f"{number:05}"
    if speaker_id == '':
        return str(number) + '.wav'
    else:
        return speaker_id + '\\' + str(number) + '.wav'


def get_file_name(speaker_id, number):
    parent_path = get_parent_path(speaker_id)
    wav_path = get_sub_folder_path(parent_path, 'wav')
    file_name = create_wav_file_name('', number)
    return pjoin(wav_path, file_name)


def get_wav_files(speaker_id):
    parent_path = get_parent_path(speaker_id)
    wav_folders = get_wav_folders(speaker_id)
    wav_files = []
    for directory in wav_folders:
        sub_dir_path = parent_path + '\\' + directory
        for base, dirs2, Files in os.walk(sub_dir_path):
            if not base.endswith('\librosa') and not base.endswith('\psf'):
                files = Files
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(directory + '\\' + file)
            files = []

    return wav_files


def get_wav_folders(speaker_id):
    parent_path = get_parent_path(speaker_id)
    return list_sub_folders(parent_path)


def get_wav_files_in_folder(path):
    files = []
    wav_files = []
    dir_path = path
    for base, dirs2, Files in os.walk(dir_path):
        if not base.endswith('\librosa') and not base.endswith('\psf'):
            files = Files
    for file in files:
        if file.endswith('.wav'):
            wav_files.append(dir_path + '\\' + file)
    return wav_files


def get_all_models_path():
    path = get_all_data_path() + '\\' + 'models'
    if not os.path.exists(path):
        make_dir(path)
    return path


def get_model_path(speaker_id, t):
    path = get_all_models_path()
    if t.__contains__('svm'):
        path = path + '\\' + "svm"
    if t.__contains__('gmm'):
        path = path + '\\' + "gmm"
        if t.__contains__('ubm'):
            path += '-ubm' + '\\'
            if not os.path.exists(path):
                make_dir(path)
            if t.__contains__('single'):
                path += 'gmm'
            if t.__contains__('universal'):
                path += 'ubm'

    if not os.path.exists(path):
        make_dir(path)

    models = []

    for base, dirs2, Files in os.walk(path):
        models = Files

    for model in models:
        if model.__contains__(speaker_id) and model.__contains__(t):
            return path + '\\' + model

    file_name = speaker_id + "_" + t + "_model.pickel"
    path = path + '\\' + file_name
    return path


def get_ids_of_paths(paths):
    ids = []
    for path in paths:
        id = get_id_of_path(path)
        if not id == 'no id in path':
            ids.append(id)
    return ids


def get_id_of_path(path):
    sub_paths = path.split('\\')
    for sub_path in sub_paths:
        if sub_path.__contains__('id'):
            return sub_path
    return "no id in path"


def check_if_file_exists_then_remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def get_all_wav_path():
    return get_all_data_path() + '\\' + "wav"


def get_all_wav_names():
    return os.listdir(get_all_wav_path())


def get_all_ids():
    return get_all_wav_names()


def get_all_data_path():
    return get_data_path()


def is_large_data_set():
    return True if get_all_data_path().__contains__('voxceleb') else False


def get_data_path():
    file = get_project_path() + '\\' + 'config.ini'
    config = ConfigParser()
    config.read(file)
    return config.get('system', 'DATASET_PATH')