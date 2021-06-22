import librosa

import csv

import numpy as np
import pandas as pd

from utils import directoryManager as dm
from utils import csvManager as cm


# def extract_mfcc_from_file(file_path):
#     signal, sr = librosa.load(file_path, sr=22050)
#     # hop_length = int(sr / 100)
#     # n_fft = int(sr / 40)
#     hop_length = 800
#     n_fft = 1600
#     features = librosa.feature.mfcc(signal, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
#     features_processed = np.mean(features.T, axis=0)
#     # features_small = features[:, 1:3]
#     feature_array = features
#     return feature_array


def extract_mfcc_from_file(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    hop_length = 800
    n_fft = 1600
    return librosa.feature.mfcc(signal, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
    # return np.mean(features.T, axis=0)


def extract_processed_mfcc_from_file(file_path):
    return np.mean(extract_mfcc_from_file(file_path).T, axis=0)


def extract_mfcc_from_file_to_csv(file_path):
    # mfcc_feat = extract_mfcc_librosa(file_path)
    mfcc_feat = extract_mfcc_from_file(file_path)
    new_file_path = dm.get_feature_csv_path(file_path, 'librosa')

    # features_small = mfcc_feat[1: 3, :]
    features = mfcc_feat
    # features = features_small
    # cm.write_features_to_file(file_path, features)
    cm.write_features_to_librosa_csv_file(new_file_path, file_path, features)


def load_features_from_csv(file_path):
    csv_path = get_feature_csv_path(file_path)
    csv_data = pd.read_csv(csv_path)
    features = csv_data.features
    return features[0]


def get_feature_csv_path(wav_path):
    sub_path = wav_path.split('\\')
    csv_path = ''
    for x in range(len(sub_path)):
        if x == (len(sub_path) - 1):
            csv_path = csv_path + '\\' + 'librosa' + '\\' + sub_path[x]
        else:
            csv_path = csv_path + sub_path[x] + '\\'
    return csv_path.replace('.wav', '.csv')


def get_features_out_of_csv(files):
    data_path = dm.get_all_data_path()
    x = []
    for file in files:
        # file_path = data_path + '\\' + file
        csv_path = get_feature_csv_path(data_path + '\\' + file)
        x.append(get_features_from_csv(csv_path))
    return x
    # return get_correct_array_form(x)


def get_features_from_csv(file):
    csv_path = get_feature_csv_path(file)
    return pd.read_csv(csv_path).features

    # with open(file, 'r') as f:
    #     reader = csv.reader(f, delimiter=',')
    #     data = []
    #     for row in reader:
    #         if len(row) > 0:
    #             new_row = []
    #             for entry in row:
    #                 new_row.append(float(entry))
    #             data.append(new_row)
    #     return data


def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


def get_correct_feature_array(files):
    x = []
    for file in files:
        file_path = file
        wav_path = file_path.replace('.csv', '.wav')
        features = extract_mfcc_from_file(wav_path)
        # decide which feature array to use
        features_small = features[1: 3, :]
        feature_array = features_small
        x.append(feature_array)
    return x
    # return get_correct_array_form(x)