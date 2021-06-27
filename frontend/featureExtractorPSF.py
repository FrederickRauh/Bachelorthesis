import python_speech_features as psf
import csv

import numpy as np
import pandas as pd

import scipy.io.wavfile as wav

from utils import directoryManager as dm
from utils import fileManager as fm


def extract_filterbank_energies_from_file(file_path):
    (rate, sig) = wav.read(file_path)
    fbank_feat = psf.logfbank(sig, rate)
    return fbank_feat


def extract_mfcc_from_file(file_path):
    # print('extracting mfcc from file : ', file_path)
    (rate, sig) = wav.read(file_path)
    return psf.mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None,
                    preemph=0.97, ceplifter=22, appendEnergy=True)


def extract_processed_mfcc_from_file(file_path):
    mfcc = extract_mfcc_from_file(file_path)
    return mfcc[1:3, :]


def extract_mfcc_from_file_to_csv(file_path):
    (rate, sig) = wav.read(file_path)
    mfcc_feat = psf.mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None,
                preemph=0.97, ceplifter=22, appendEnergy=True)
    new_file_path = dm.get_feature_psf_csv_path(file_path)
    features = mfcc_feat
    fm.write_features_to_psf_csv_file(new_file_path, file_path, features)


def load_features_from_csv(file_path):
    csv_path = get_feature_csv_path(file_path)
    csv_data = pd.read_csv(csv_path)
    features = np.array(csv_data.features)
    return features


def get_feature_csv_path(wav_path):
    csv_path = dm.get_feature_psf_csv_path(wav_path)
    return csv_path.replace('.wav', '.csv')


def get_features_out_of_csv(files):
    data_path = dm.get_all_data_path()
    x = []
    for file in files:
        file_path = data_path + '\\' + file
        csv_path = get_feature_csv_path(file_path)
        x.append(get_features_from_csv(csv_path))
    return x


def get_features_from_csv(file):
    csv_path = get_feature_csv_path(file)
    return pd.read_csv(csv_path).features


#json version
def extract_mfcc_from_file_to_json(file_path):
    mfcc_feat = extract_processed_mfcc_from_file(file_path)
    new_file_path = dm.get_feature_psf_json_path(file_path)
    features = mfcc_feat
    fm.write_features_to_librosa_json_file(new_file_path, file_path, features)


def load_features_from_json(file_path):
    json_path = dm.get_feature_psf_json_path(file_path)
    json_data = pd.read_json(json_path)
    features = json_data.features
    return features
