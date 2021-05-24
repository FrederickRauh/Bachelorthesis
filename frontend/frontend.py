import os

import scipy
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
from os.path import dirname, join as pjoin

from python_speech_features import mfcc
from python_speech_features import logfbank


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


# getVoiceInput(30, 44100, 1)
def getVoiceInput(timespan, samplerate, number, name):
    # samplerate = 44100
    # seconds = 5
    makeFolder(name)
    filename = create_file_name(name, number)
    print(filename)
    recording = sd.rec(int(timespan * samplerate), samplerate=samplerate, channels=2)
    sd.wait()
    path = makeFolder(name)
    wav.write(path + '\\' + filename, samplerate, recording)
    os.startfile(path)


def create_file_name(name, number):
    return name + '-' + str(number) + '.wav'


def makeFolder(name):
    parent_path = os.path.join(os.getcwd(), "data")
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    path = os.path.join(parent_path, name)
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError as error:
        print("Creating directory %s has failed. Error %s" % (path, error))
    return path


def getFileName(name, number):
    parent_path = os.path.join(os.getcwd(), "data")
    path = os.path.join(parent_path, name)
    file_name = create_file_name(name, number)
    return pjoin(path, file_name)


def extractFeatures(name, number):
    filepath = getFileName(name, number)
    (rate, sig) = wav.read(filepath)
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)

    return fbank_feat
