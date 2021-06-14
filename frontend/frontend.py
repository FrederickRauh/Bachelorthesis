import os

import scipy
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav

from python_speech_features import mfcc
from python_speech_features import logfbank

from utils import directoryManager as dm
from utils import csvManager as cm


# getSpeechInput()
# def getSpeechInput():
#     recognizer = sr.Recognizer()
#     try:
#         print("listening...")
#         with sr.Microphone() as source:
#             voice = recognizer.listen(source)
#             data = recognizer.recognize_google(voice)
#             print(data)
#     except:
#         pass


def get_voice_input_stream(timespan, samplerate, number, speaker_id):
    for x in range(timespan//number):
        get_voice_input(timespan, samplerate, x, speaker_id)


# getVoiceInput(30, 44100, 1)
def get_voice_input(timespan, samplerate, number, speaker_id):
    # samplerate = 44100
    # seconds = 5
    parent_path = dm.get_parent_path(speaker_id)
    wav_path = dm.get_subfolder_path(parent_path, 'wav')
    filename = dm.create_wav_file_name(speaker_id, number)
    recording = sd.rec(int(timespan * samplerate), samplerate=samplerate, channels=2)
    sd.wait()
    wav.write(wav_path + '\\' + filename, samplerate, recording)
    os.startfile(wav_path)


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
    (rate, sig) = wav.read(file_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0, highfreq=None,preemph=0.97, ceplifter=22, appendEnergy=True)
    features = mfcc_feat[1: 3, :]
    cm.write_features_to_file(file_path, features)


def process_features(speaker_id):
    files = dm.get_wav_files(speaker_id)
    if len(files) > 0:
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            features = extract_mfcc_from_file(file_path)
            cm.edit_csv(speaker_id, file, features)