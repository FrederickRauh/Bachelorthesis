import python_speech_features as psf
import csv

import numpy as np
import pandas as pd

import scipy.io.wavfile as wav
from scipy.signal.windows import hann
from scipy.stats import skew

from sklearn import preprocessing

from utils.config import FEATURES
from utils import audioManager as am, directoryManager as dm, fileManager as fm, util


def extract_filterbank_energies_from_file(file_path):
    (rate, sig) = wav.read(file_path)
    fbank_feat = psf.logfbank(sig, rate)
    return fbank_feat


def extract_signal_from_file(file_path):
    sr, signal = wav.read(file_path)
    sr, signal = am.get_four_seconds_frame_of_audio(sr, signal, 'psf')
    # if signal is stereo only take one channel
    if isinstance(signal[0], np.ndarray):
        signal = signal[:, 0]
    return signal


def extract_mfcc_from_signal(signal):
    return psf.mfcc(signal=signal,
                    samplerate=FEATURES.SAMPLE_RATE,
                    winlen=FEATURES.WINLEN,
                    numcep=FEATURES.N_MFCC,
                    nfilt=FEATURES.N_MELS,
                    nfft=FEATURES.NFFT,
                    appendEnergy=FEATURES.APPENDENERGY,
                    winfunc=FEATURES.WINFUNC)


def get_delta_delta_from_signal(signal):
    mfcc = extract_mfcc_from_signal(signal)
    # Deltas
    d_mfcc = psf.delta(mfcc, 2)
    # Deltas-Deltas
    dd_mfcc = psf.delta(d_mfcc, 2)
    return np.hstack((mfcc, d_mfcc, dd_mfcc))


def extract_filter_banks_and_energies_from_signal(signal):
    return psf.fbank(signal, samplerate=FEATURES.SAMPLE_RATE,
                     nfilt=FEATURES.N_MELS,
                     winlen=FEATURES.WINLEN,
                     winstep=FEATURES.WINSTEP,
                     winfunc=FEATURES.WINFUNC
                     )


def extract_processed_features_from_file(file_path):
    signal = extract_signal_from_file(file_path)
    ft1 = get_delta_delta_from_signal(signal)
    ft2, ft3 = extract_filter_banks_and_energies_from_signal(signal)
    return np.vstack((ft1, ft2))


def extract_mfcc_from_file_to_csv(file_path):
    mfcc_feat = extract_processed_features_from_file(file_path)
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
    data_path = dm.get_all_wav_path()
    x = []
    for file in files:
        file_path = data_path + '\\' + file
        csv_path = get_feature_csv_path(file_path)
        x.append(get_features_from_csv(csv_path))
    return x


def get_features_from_csv(file):
    csv_path = get_feature_csv_path(file)
    return pd.read_csv(csv_path).features


# json version
def extract_mfcc_from_file_to_json(file_path):
    mfcc_feat = extract_processed_features_from_file(file_path)
    new_file_path = dm.get_feature_psf_json_path(file_path)
    features = mfcc_feat
    fm.write_features_to_json_file(new_file_path, file_path, features)


def load_features_from_json(file_path):
    json_path = dm.get_feature_psf_json_path(file_path)
    json_data = pd.read_json(json_path)
    features = json_data.features
    return features
