import os

import scipy.io.wavfile as wav
from os.path import dirname, join as pjoin


def create_wav_file_name(speaker_id, number):
    number = f"{number:05}"
    return speaker_id + '-' + str(number) + '.wav'


def create_csv_file_name(speaker_id):
    return speaker_id + '.csv'


def get_parent_path(speaker_id):
    parent_path = make_dir(os.path.join(os.getcwd(), "data"))
    return make_dir(os.path.join(parent_path, speaker_id))


def sub_folder_switch(x):
    return {
        'csv': 1,
        'wav': 2
    }.get(x, 9)


def list_sub_folders(parent_path):
    return os.listdir(parent_path)


def get_sub_folder_path(parent_path, sub_folder):
    path = os.path.join(parent_path, sub_folder)
    if not os.path.exists(path):
        make_dir(path)
    return path
    print("Sub folder: " + sub_folder + " does not exist")


def create_sub_folder(parent_path, sub_folder_name):
    make_dir(os.path.join(parent_path, sub_folder_name))


def get_csv_path(speaker_id):
    parent_path = get_parent_path(speaker_id)
    csv_path = get_sub_folder_path(parent_path, 'csv')
    filename = create_csv_file_name(speaker_id)
    return csv_path + '\\' + filename


def get_wav_folder_path(speaker_id):
    parent_path = get_parent_path(speaker_id)
    return get_sub_folder_path(parent_path, 'wav')


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as error:
            print("Creating directory %s has failed. Error %s" % (path, error))
    return path


# def get_file_name(speaker_id, number):
#     parent_path = get_parent_path(speaker_id)
#     wav_path = get_subfolder_path(parent_path, 'wav')
#     file_name = create_wav_file_name(speaker_id, number)
#     return pjoin(wav_path, file_name)


def get_wav_files(speaker_id):
    parent_path = get_parent_path(speaker_id)
    directories = list_sub_folders(parent_path)
    files = []
    wav_files = []
    for directory in directories:
        if not directory == 'csv' and not directory == 'model':
            dir_path = parent_path + '\\' + directory
            for base, dirs2, Files in os.walk(dir_path):
                files = Files
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(directory + '\\' + file)
            files = []
    return wav_files


def get_model_path(speaker_id, type):
    parent_path = get_parent_path(speaker_id)
    model_folder_path = get_sub_folder_path(parent_path, 'model')
    file_name = speaker_id + "_"+ type + "_model.pickel"
    return model_folder_path + '\\' + file_name


def get_all_data_path():
    return os.path.join(os.getcwd(), "data")


def get_all_ids():
    ids = get_all_data_names()
    ids.remove('pairs.csv')
    return ids


def get_all_data_csv_file_path():
    path = os.path.join(os.getcwd(), "data") + '\\' + 'pairs.csv'
    return path


def get_all_data_names():
    return os.listdir(os.path.join(os.getcwd(), "data"))

