import math

import numpy as np

import scipy.io.wavfile as wav
from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf
from utils import util

"""
AudioManager is used as util service for feature extraction and frontend classes. 
This file should only contain methods used on a signal.
"""
def get_audio_length(file_path):
    sr, signal = wav.read(file_path)
    return len(signal) / float(sr)


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


def get_four_second_intervals_of_audio(sr, signal, t):
    duration = len(signal) / float(sr)
    signals = []
    if duration >= 4:
        frame_amount = int(math.floor(duration / 4))
        for x in range(frame_amount):
            frame_start = x * 63999
            frame_end = (x+1) * 63999
            frame = signal[frame_start:frame_end]
            sr, cut_signal = get_four_seconds_frame_of_audio(sr, frame, t)
            signals.append(cut_signal)
    else:
        # If smaller than 4 seconds
        if duration < 4:
            sr, cut_signal = get_four_seconds_frame_of_audio(sr, signal, t)
            signals.append(cut_signal)
    return sr, signals


def get_features_for_prediction(file_path, feature_type):
    if feature_type == 'psf':
        return fpsf.extract_processed_features_from_file(file_path)
        # return util.get_correct_array_form([fpsf.extract_processed_features_from_file(file_path)])
    else:
        return flib.extract_processed_features_from_file(file_path)
        # return util.get_correct_array_form([flib.extract_processed_features_from_file(file_path)])
