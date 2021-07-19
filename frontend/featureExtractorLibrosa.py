import librosa
import csv

import numpy as np
import pandas as pd
from scipy.stats import skew

from utils.config import Features as config
from utils import directoryManager as dm, fileManager as fm, util


def extract_mfcc_from_file(file_path):
    # sr = 16000 to match psf..
    signal, sr = librosa.load(file_path, sr=16000)
    sr, signal = util.get_four_seconde_frame_of_audio(sr, signal, 'librosa')

    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=config.N_MFCC)
    # return mfcc
    d_mfcc = librosa.feature.delta(mfcc, order=1)
    dd_mfcc = librosa.feature.delta(d_mfcc, order=2)
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



    # return librosa.feature.mfcc(signal, n_mfcc=n_mfcc)
    # return librosa.feature.mfcc(y=signal, sr=sr, S=S, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter, n_mels=n_mels, n_fft=nfft, win_length=winlen, window=winfunc)
    # return librosa.feature.mfcc(signal, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)


def extract_processed_mfcc_from_file(file_path):
    mfcc = extract_mfcc_from_file(file_path)
    # d_mfcc = librosa.feature.delta(mfcc, order=1)
    # dd_mfcc = librosa.feature.delta(mfcc, order=2)
    # concat_mfcc_feat = np.hstack((mfcc, d_mfcc, dd_mfcc))
    # return concat_mfcc_feat
    return mfcc
    # return np.mean(extract_mfcc_from_file(file_path).T, axis=0)


def extract_mfcc_from_file_to_csv(file_path):
    mfcc_feat = extract_processed_mfcc_from_file(file_path)
    new_file_path = dm.get_feature_csv_path(file_path, 'librosa')
    features = mfcc_feat
    fm.write_features_to_librosa_csv_file(new_file_path, file_path, features)


def load_features_from_csv(file_path):
    csv_path = get_feature_csv_path(file_path)
    csv_data = pd.read_csv(csv_path)
    features = csv_data.features
    return features[0]


def get_feature_csv_path(wav_path):
    csv_path = dm.get_feature_librosa_csv_path(wav_path)
    return csv_path.replace('.wav', '.csv')


def get_features_out_of_csv(files):
    data_path = dm.get_all_wav_path()
    x = []
    for file in files:
        file_path = data_path + '\\' + file
        csv_path = dm.get_feature_librosa_csv_path(file_path)
        x.append(get_features_from_csv(csv_path))
    return x


def get_features_from_csv(file):
    csv_path = get_feature_csv_path(file)
    return pd.read_csv(csv_path).features


#json version
def extract_mfcc_from_file_to_json(file_path):
    mfcc_feat = extract_processed_mfcc_from_file(file_path)
    new_file_path = dm.get_feature_librosa_json_path(file_path)
    features = mfcc_feat
    fm.write_features_to_json_file(new_file_path, file_path, features)


def load_features_from_json(file_path):
    json_path = get_feature_json_path(file_path)
    json_data = pd.read_json(json_path)
    features = json_data.features
    return features[0]


def get_feature_json_path(wav_path):
    json_path = dm.get_feature_librosa_json_path(wav_path)
    return json_path.replace('.wav', '.json')
