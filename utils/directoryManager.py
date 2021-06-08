import os

import scipy.io.wavfile as wav
from os.path import dirname, join as pjoin


def create_wav_file_name(name, number):
    return name + '-' + str(number) + '.wav'


def create_csv_file_name(name):
    return name + '.csv'


def get_parent_path(name):
    parent_path = make_dir(os.path.join(os.getcwd(), "data"))
    return make_dir(os.path.join(parent_path, name))


def sub_folder_switch(x):
    return {
        'csv': 1,
        'wav': 2
    }.get(x, 9)


def get_subfolder_path(parent_path, sub_folder):
    folder_layer = sub_folder_switch(sub_folder)
    if folder_layer == 9:
        return
    elif folder_layer == 1:
        return make_dir(os.path.join(parent_path, "csv"))
    elif folder_layer == 2:
        return make_dir(os.path.join(parent_path, "wav"))


def get_csv_path(name):
    parent_path = get_parent_path(name)
    csv_path = get_subfolder_path(parent_path, 'csv')
    filename = create_csv_file_name(name)
    return csv_path + '\\' + filename


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as error:
            print("Creating directory %s has failed. Error %s" % (path, error))
    return path


def get_file_name(name, number):
    parent_path = get_parent_path(name)
    wav_path = get_subfolder_path(parent_path, 'wav')
    file_name = create_wav_file_name(name, number)
    return pjoin(wav_path, file_name)


def get_file_amount(name):
    parent_path = get_parent_path(name)
    folder_path = get_subfolder_path(parent_path, 'wav')
    total = 0
    for base, dirs, files in os.walk(folder_path):
        print(len(files))
        total = len(files)
    return total


def get_all_data_path():
    return os.path.join(os.getcwd(), "data")


def get_all_data_names():
    return os.listdir(os.path.join(os.getcwd(), "data"))


def get_all_wav_names(name):
    parent_path = get_parent_path(name)
    wav_folder_path = get_subfolder_path(parent_path, 'wav')
    return os.listdir(wav_folder_path)
