import python_speech_features as psf

import numpy as np
import pandas as pd

import scipy.io.wavfile as wav
from sklearn import preprocessing

from utils import audioManager as am, directoryManager as dm, jsonManager as jm


def extract_filterbank_energies_from_file(file_path):
    (rate, sig) = wav.read(file_path)
    fbank_feat = psf.logfbank(sig, rate)
    return fbank_feat


def extract_signal_from_file(file_path):
    sr, signal = wav.read(file_path)
    sr, signal = am.get_four_seconds_frame_of_audio(sr, signal, 'psf')
    # if signal is stereo only take one channel
    return signal[:, 0] if isinstance(signal[0], np.ndarray) else signal


def extract_mfcc_from_signal(signal):
    return psf.mfcc(signal=signal,
                    samplerate=config.SAMPLE_RATE,
                    winlen=config.WINLEN,
                    numcep=config.N_MFCC,
                    nfilt=config.N_MELS,
                    nfft=config.NFFT,
                    appendEnergy=config.APPENDENERGY,
                    winfunc=config.WINFUNC)


def get_delta_delta_from_signal(mfcc):
    # Deltas
    d_mfcc = psf.delta(mfcc, 1)
    # Deltas-Deltas
    dd_mfcc = psf.delta(d_mfcc, 2)
    return d_mfcc, dd_mfcc



def extract_filter_banks_and_energies_from_signal(signal):
    return psf.fbank(signal, samplerate=config.SAMPLE_RATE,
                     nfilt=config.N_MELS,
                     winlen=config.WINLEN,
                     winstep=config.WINSTEP,
                     winfunc=config.WINFUNC
                     )


def extract_processed_features_from_file(file_path):
    signal = extract_signal_from_file(file_path)
    mfcc = extract_mfcc_from_signal(signal)
    mfcc = preprocessing.scale(mfcc)
    d_mfcc, dd_mfcc = get_delta_delta_from_signal(mfcc)
    # ft1 = get_delta_delta_from_signal(signal)
    # ft2, ft3 = extract_filter_banks_and_energies_from_signal(signal)
    processed_features = np.hstack((mfcc, d_mfcc))
    return processed_features


# json version
def extract_features_from_file_to_json(file_path):
    mfcc_feat = extract_processed_features_from_file(file_path)
    new_file_path = dm.get_feature_psf_json_path(file_path)
    features = mfcc_feat
    jm.write_features_to_json_file(new_file_path, file_path, features)