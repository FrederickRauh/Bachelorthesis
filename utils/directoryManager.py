import os

import scipy.io.wavfile as wav
from os.path import dirname, join as pjoin


def create_file_name(name, number):
    return name + '-' + str(number) + '.wav'


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


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as error:
            print("Creating directory %s has failed. Error %s" % (path, error))
    return path


def getFileName(name, number):
    parent_path = get_parent_path(name)
    wav_path = get_subfolder_path(parent_path, 'wav')
    file_name = create_file_name(name, number)
    return pjoin(wav_path, file_name)