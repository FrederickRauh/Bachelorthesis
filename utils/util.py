import os

import numpy as np

import scipy.io.wavfile as wav

from datetime import datetime

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib

from utils import directoryManager as dm


# util part



def get_four_seconde_frame_of_audio(sr, signal, t):
    duration = len(signal) / float(sr)
    # four seconds of data from .wav if > 4sec
    if duration >= 4:
        length_in_seconds = duration
        # middle = i  # (len(signal) // 2) - 1
        signal_per_second = sr
        middle = (len(signal) // 2)
        left_side = int(middle - (2 * sr))
        right_side = int(middle + (2 * sr))
        signal = signal[left_side:right_side]
    # if < 4sec add padding of 0 to the back
    if duration < 4:
        missing_time = 4 - duration
        length_of_padding = missing_time * float(sr)
        for x in range(int(length_of_padding)):
            if t == 'psf':
                signal = np.append(signal, 0)
            else:
                signal = np.append(signal, 0)
    return sr, signal

# Turn 3Dim Array in 2D
def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


def get_features_for_prediciton(file_path, feature_type):
    if feature_type == 'psf':
        return [fpsf.extract_processed_features_from_file(file_path)]
        # return get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file_path)])
    else:
        return [flib.extract_processed_features_from_file(file_path)]
        # return get_correct_array_form([flib.extract_processed_mfcc_from_file(file_path)])


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
