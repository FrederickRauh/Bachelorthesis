import librosa

import numpy as np
from scipy.stats import skew

from config import FEATURES
from utils import audioManager as am, dataframeManager as dam, directoryManager as dm, jsonManager as jm


def extract_mfcc_from_signal(signal):
    return librosa.feature.mfcc(signal, sr=FEATURES.SAMPLE_RATE, n_mfcc=FEATURES.N_MFCC)


def extract_processed_mfcc_from_file(file_path):
    signal, sr = librosa.load(file_path, sr=FEATURES.SAMPLE_RATE)
    sr, signal = am.get_four_seconds_frame_of_audio(sr, signal, 'librosa')
    mfcc = extract_mfcc_from_signal(signal)
    return mfcc


def extract_processed_features_from_file(file_path):
    signal, sr = librosa.load(file_path, sr=FEATURES.SAMPLE_RATE)
    sr, signal = am.get_four_seconds_frame_of_audio(sr, signal, 'librosa')
    mfcc = extract_mfcc_from_signal(signal)
    d_mfcc = librosa.feature.delta(mfcc)
    dd_mfcc = librosa.feature.delta(d_mfcc)

    # ft1 = np.concatenate((mfcc, d_mfcc, dd_mfcc), axis=0)
    ft1 = np.hstack((mfcc, d_mfcc, dd_mfcc))
    ft2 = librosa.feature.zero_crossing_rate(signal)[0]
    ft3 = librosa.feature.spectral_rolloff(signal)[0]
    ft4 = librosa.feature.spectral_centroid(signal)[0]
    ft5 = librosa.feature.spectral_contrast(signal)[0]
    ft6 = librosa.feature.spectral_bandwidth(signal)[0]
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis=1), np.max(ft1, axis=1),
                           np.median(ft1, axis=1), np.min(ft1, axis=1)))

    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
    return np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))


# def extract_processed_features_from_file(file_path):
#     signal, sr = librosa.load(file_path, sr=FEATURES.SAMPLE_RATE)
#     sr, signal = am.get_four_seconds_frame_of_audio(sr, signal, 'librosa')
#     mfcc = extract_mfcc_from_signal(signal)
#     d_mfcc = librosa.feature.delta(mfcc)
#     energy = librosa.feature.rms(S=signal)
#     d_energy = librosa.feature.delta(energy)


def get_right_format(ft):
    return np.hstack((np.mean(ft), np.std(ft), skew(ft), np.max(ft), np.median(ft), np.min(ft)))


# json version
def extract_mfcc_from_file_to_json(file_path):
    mfcc_feat = extract_processed_features_from_file(file_path)
    new_file_path = dm.get_feature_librosa_json_path(file_path)
    features = mfcc_feat
    jm.write_features_to_json_file(new_file_path, file_path, features)