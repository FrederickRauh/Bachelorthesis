import numpy as np

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf
from utils import util

"""
AudioManager is used as util service for feature extraction and frontend classes. 
This file should only contain methods used on a signal.
"""


def get_four_seconds_frame_of_audio(sr, signal, t):
    duration = len(signal) / float(sr)
    # four seconds of data from .wav if > 4sec
    if duration >= 4:
        middle = (len(signal) // 2)
        left_side = int(middle - (2 * sr))
        right_side = int(middle + (2 * sr))
        signal = signal[left_side:right_side]
    # if < 4sec add padding of 0 to the back
    if duration < 4:
        missing_time = 4 - duration
        length_of_padding = missing_time * float(sr)
        for x in range(int(length_of_padding)):
            signal = np.append(signal, 0) if t == 'psf' else np.append(signal, 0)
    return sr, signal


def get_features_for_prediction(file_path, feature_type):
    if feature_type == 'psf':
        # return [fpsf.extract_processed_features_from_file(file_path)]
        return util.get_correct_array_form([fpsf.extract_processed_features_from_file(file_path)])
    else:
        return [flib.extract_processed_features_from_file(file_path)]
        # return util.get_correct_array_form([flib.extract_processed_features_from_file(file_path)])
