import numpy as np

import scipy.io.wavfile as wav

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


def load_test_files(speaker_ids):
    files = []
    for speaker_id in speaker_ids:

        # dir = dm.get_test_subfolders(speaker_id)
        dir = dm.get_voxceleb_subfolders(speaker_id)

        for dir_path in dir:

            # files_path = dm.get_test_path() + '\\' + speaker_id + '\\' + dir_path
            files_path = dm.get_voxceleb_path() + '\\' + speaker_id + '\\' + dir_path

            f = dm.get_wav_files_in_folder(files_path)
            for x in range(len(f)):
                files.append(f[x])

    return files
