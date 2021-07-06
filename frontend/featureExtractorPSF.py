import python_speech_features as psf
import csv

import numpy as np
import pandas as pd

import scipy.io.wavfile as wav
from scipy.signal.windows import hann

from utils import util
from utils import directoryManager as dm
from utils import fileManager as fm


def extract_filterbank_energies_from_file(file_path):
    (rate, sig) = wav.read(file_path)
    fbank_feat = psf.logfbank(sig, rate)
    return fbank_feat

# Parameters:
# signal – the audio signal from which to compute features. Should be an N*1 array
# samplerate – the samplerate of the signal we are working with.
# winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
# winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
# numcep – the number of cepstrum to return, default 13
# nfilt – the number of filters in the filterbank, default 26.
# nfft – the FFT size. Default is 512.
# lowfreq – lowest band edge of mel filters. In Hz, default is 0.
# highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
# preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
# ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
# appendEnergy – if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
# winfunc – the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
# Returns:
# A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.

def extract_mfcc_from_file(file_path):
    # print('extracting mfcc from file : ', file_path)
    sr, signal = util.get_four_seconde_frame_of_wav_file(file_path)
    n_mfcc = 13
    n_mels = 40  # prev: 26
    n_fft = 2048  # prev: 0.025 and 2048(duo to error message)
    hop_length = 160  # prev: 0.01
    fmin = 0
    fmax = None
    preemph = 0.0  # prev: 0.07
    ceplifter = 0  # prev: 22
    appendEnergy = False # prev: True
    winlen = 0.064  # prev n_fft / sr
    winstep = 0.01  # prev: 0.036, hop_length / sr (default 0.01 (10ms))
    # sr = 16000 # to get a uniform samplerate
    # todo 16-64ms frames(winlen and winstep), maybe 4 seconds frames maybe add padding
    return psf.mfcc(signal=signal, samplerate=sr, winlen=winlen, winstep=winstep, numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
                    preemph=preemph, ceplifter=ceplifter, appendEnergy=appendEnergy, winfunc=hann)


def extract_processed_mfcc_from_file(file_path):
    mfcc = extract_mfcc_from_file(file_path)
    return mfcc[0:40, :]


def extract_mfcc_from_file_to_csv(file_path):
    mfcc_feat = extract_mfcc_from_file(file_path)
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
    fm.write_features_to_json_file(new_file_path, file_path, features)


def load_features_from_json(file_path):
    json_path = dm.get_feature_psf_json_path(file_path)
    json_data = pd.read_json(json_path)
    features = json_data.features
    return features
