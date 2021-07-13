import librosa
import csv

import numpy as np
import pandas as pd

from utils import directoryManager as dm
from utils import fileManager as fm


def extract_mfcc_from_file(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    hop_length = 800
    n_fft = 1600
    return librosa.feature.mfcc(signal, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)


def extract_processed_mfcc_from_file(file_path):
    return np.mean(extract_mfcc_from_file(file_path).T, axis=0)


def extract_mfcc_from_file_to_csv(file_path):
    mfcc_feat = extract_mfcc_from_file(file_path)
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
    mfcc_feat = extract_mfcc_from_file(file_path)
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
