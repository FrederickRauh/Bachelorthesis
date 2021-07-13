import os

import numpy as np

import scipy.io.wavfile as wav

from datetime import datetime

from utils import directoryManager as dm


# util part
def remove_finished_ids(ids, finished_ids):
    for id in finished_ids:
        if ids.__contains__(id):
            ids.remove(id)
    return ids


def get_four_seconde_frame_of_wav_file(file_path):
    sr, signal = wav.read(file_path)
    # four seconds of data from .wav
    duration = len(signal) // float(sr)
    if duration >= 4:
        length_in_seconds = duration
        # middle = i  # (len(signal) // 2) - 1
        signal_per_second = sr
        middle = (len(signal) // 2)
        left_side = int(middle - (2 * sr))
        right_side = int(middle + (2 * sr))
        signal = signal[left_side:right_side]
    return sr, signal


def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


# Response from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split_array_for_multiprocess(array, num):
    avg = len(array) / float(num)
    output = []
    last = 0.0
    while last < len(array):
        output.append(array[int(last):int(last + avg)])
        last += avg
    return output


def adjust_file_amount_for_voxceleb(dir, speaker_id):
    parent_path = dm.get_voxceleb_path() + '\\' + speaker_id
    path = parent_path + '\\' + dir[0]
    i = 0
    for base, dirs2, Files in os.walk(path):
        for file in Files:
            if file.endswith(".wav"):
                i += 1
    if i < 10:
        next_dir = dm.get_voxceleb_subfolders(speaker_id)[len(dm.get_voxceleb_subfolders(speaker_id)) - 2]
        if next_dir.__contains__('model'):
            next_dir = dm.get_voxceleb_subfolders(speaker_id)[len(dm.get_voxceleb_subfolders(speaker_id)) - 3]
        dir.append(next_dir)
    return dir

def load_test_files(speaker_ids):
    files = []
    for speaker_id in speaker_ids:
        if dm.is_large_data_set():
            dir = [dm.get_voxceleb_subfolders(speaker_id)[len(dm.get_voxceleb_subfolders(speaker_id)) - 1]]
            dir = adjust_file_amount_for_voxceleb(dir, speaker_id)
        else:
            dir = dm.get_test_subfolders(speaker_id)
        for dir_path in dir:
            if dm.is_large_data_set():
                files_path = dm.get_voxceleb_path() + '\\' + speaker_id + '\\' + dir_path
            else:
                files_path = dm.get_test_path() + '\\' + speaker_id + '\\' + dir_path

            f = dm.get_wav_files_in_folder(files_path)
            for x in range(len(f)):
                files.append(f[x])

    return files


def get_duration(start_time):
    after_time = datetime.now()
    duration = after_time - start_time
    hours = duration.total_seconds() // 3600
    minutes = (duration.total_seconds() // 60) - (hours * 60)
    seconds = duration.total_seconds() - (hours * 3600) - (minutes * 60)
    return "--> duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds)
          # "----- Model: accuracy: %f; standard deviation of %f" % (score.mean(), score.std())
