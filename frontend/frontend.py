import os
import scipy

import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav

from frontend import featureExtractorPSF as fe

from utils import directoryManager as dm, debug
from utils import fileManager as fm


# getSpeechInput()
# def getSpeechInput():
#     recognizer = sr.Recognizer()
#     try:
#         debug.print("listening...")
#         with sr.Microphone() as source:
#             voice = recognizer.listen(source)
#             data = recognizer.recognize_google(voice)
#             debug.print(data)
#     except:
#         pass


def get_voice_input_stream(timespan, samplerate, number, speaker_id, test):
    debug.log(("collecting voice samples...."))
    for x in range(number):
        get_voice_input(timespan, samplerate, x, speaker_id, test)


# getVoiceInput(30, 44100, 1)
def get_voice_input(timespan, samplerate, number, speaker_id, test):
    # samplerate = 44100
    # seconds = 5
    # parent_path = dm.get_parent_path(speaker_id)
    # wav_path = dm.get_sub_folder_path(parent_path, 'wav')
    wav_path = dm.get_file_name(speaker_id, number)
    if test:
        wav_path = wav_path.replace('data', 'test')
    filename = str(number)
    recording = sd.rec(int(timespan * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    wav.write(wav_path, samplerate, recording)
    os.startfile(wav_path)


def process_features_with_psf(speaker_id):
    files = dm.get_wav_files(speaker_id)
    if len(files) > 0:
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            features = fe.extract_mfcc_from_file_psf(file_path)
            fm.edit_csv(speaker_id, file, features)


def process_features_with_librosa(speaker_id):
    files = dm.get_wav_files(speaker_id)
    if len(files) > 0:
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            features = fe.extract_mfcc_from_file_librosa(file_path)
            fm.edit_csv(speaker_id, file, features)

