import numpy as np

import scipy.io.wavfile as wav


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


# def get_correct_feature_array(files):
#     x = []
#     for file in files:
#         file_path = file
#         wav_path = file_path.replace('.csv', '.wav2')
#         features = extract_mfcc_from_file(wav_path)
#         # decide which feature array to use
#         features_small = features[1: 3, :]
#         feature_array = features_small
#         x.append(feature_array)
#     return x
    # return get_correct_array_form(x)