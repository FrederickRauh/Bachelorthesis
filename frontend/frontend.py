import os

import scipy
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav

from python_speech_features import mfcc
from python_speech_features import logfbank

from utils import directoryManager as dm


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


def get_voice_input_stream(timespan, samplerate, number, name):
    for x in range(timespan//number):
        get_voice_input(timespan, samplerate, x, name)


# getVoiceInput(30, 44100, 1)
def get_voice_input(timespan, samplerate, number, name):
    # samplerate = 44100
    # seconds = 5
    parent_path = dm.get_parent_path(name)
    wav_path = dm.get_subfolder_path(parent_path, 'wav')
    filename = dm.create_file_name(name, number)
    recording = sd.rec(int(timespan * samplerate), samplerate=samplerate, channels=2)
    sd.wait()
    wav.write(wav_path + '\\' + filename, samplerate, recording)
    os.startfile(wav_path)


def extract_features(name, number):
    filepath = dm.getFileName(name, number)
    (rate, sig) = wav.read(filepath)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)

    return fbank_feat
