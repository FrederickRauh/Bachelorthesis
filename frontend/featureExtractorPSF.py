import csv
import scipy

import numpy as np
import pandas as pd
import speech_recognition as sr
import sounddevice as sd

import scipy.io.wavfile as wav

from python_speech_features import mfcc
from python_speech_features import logfbank

from utils import directoryManager as dm
from utils import csvManager as cm


def extract_filterbank_energies_from_file(file_path):
    # print('extracting filterbank energies from file : ', file_path)
    (rate, sig) = wav.read(file_path)
    fbank_feat = logfbank(sig, rate)
    return fbank_feat


def extract_mfcc_from_file(file_path):
    # print('extracting mfcc from file : ', file_path)
    (rate, sig) = wav.read(file_path)
    return mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None,
                    preemph=0.97, ceplifter=22, appendEnergy=True)


def extract_processed_mfcc_from_file(file_path):
    mfcc = extract_mfcc_from_file(file_path)
    return mfcc[1:3, :]


def extract_mfcc_from_file_to_csv(file_path):
    (rate, sig) = wav.read(file_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None,
                preemph=0.97, ceplifter=22, appendEnergy=True)

    new_file_path = dm.get_feature_csv_path(file_path, 'psf')
    # features_small = mfcc_feat[1: 3, :]
    features = mfcc_feat
    # features = features_small
    # cm.write_features_to_file(file_path, features)
    cm.write_features_to_psf_csv_file(new_file_path, file_path, features)


def load_features_from_csv(file_path):
    csv_path = get_feature_csv_path(file_path)
    csv_data = pd.read_csv(csv_path)
    features = csv_data.features
    # feature_array = []
    # for x in range(len(features)):
    #     feature_array.append(features[x])
    return features


def get_feature_csv_path(wav_path):
    sub_path = wav_path.split('\\')
    csv_path = ''
    for x in range(len(sub_path)):
        if x == (len(sub_path) - 1):
            csv_path = csv_path + '\\' + 'psf' + '\\' + sub_path[x]
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
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = []
        for row in reader:
            if len(row) > 0:
                new_row = []
                for entry in row:
                    new_row.append(float(entry))
                data.append(new_row)
        return data


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