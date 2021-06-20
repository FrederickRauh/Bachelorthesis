import librosa
import scipy
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
    mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat


def extract_mfcc_from_file_to_csv(file_path):
    # mfcc_feat = extract_with_python_speech(file_path)
    mfcc_feat = extract_with_librosa(file_path)
    features_small = mfcc_feat[1: 3, :]
    features = mfcc_feat
    # features = features_small
    # cm.write_features_to_file(file_path, features)
    cm.write_features_to_librosa_csv_file(file_path, features)


def extract_with_python_speech(file_path):
    (rate, sig) = wav.read(file_path)
    return mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None,preemph=0.97, ceplifter=22, appendEnergy=True)


def extract_with_librosa(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    print(sr)
    # hop_length = int(sr / 100)
    hop_length =  40
    n_fft = int(sr / 40)
    mfcc = librosa.feature.mfcc(signal, sr, n_mfcc=13, hop_length=hop_length, n_fft=n_fft)
    return mfcc