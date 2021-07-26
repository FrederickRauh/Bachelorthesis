import logging
import multiprocessing
import os
import scipy

from datetime import datetime

import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib

from utils import dataframeManager as dam, directoryManager as dm, util
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
from utils.config import IDS, SYSTEM, FEATURES


def get_voice_input_stream(timespan, samplerate, number, speaker_id, test):
    logging.debug("collecting voice samples....")
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
            features = fpsf.extract_mfcc_from_file_psf(file_path)
            fm.edit_csv(speaker_id, file, features)


def process_features_with_librosa(speaker_id):
    files = dm.get_wav_files(speaker_id)
    if len(files) > 0:
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            features = flib.extract_mfcc_from_file_librosa(file_path)
            fm.edit_csv(speaker_id, file, features)


def feature_extraction_for_n_speaker(speaker_ids, create_dataframe, feature_type, mfcc_count):

    if len(speaker_ids) > 9:
        PROCESSES = SYSTEM.PROCESSES
        split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, PROCESSES)
        pool = multiprocessing.Pool(processes=PROCESSES)
        data = []
        for x in range(PROCESSES):
            data.append((split_speaker_ids[x], feature_type, mfcc_count))
        pool.starmap(feature_extraction_for_files, data)
        pool.close()
        pool.join()
    else:
        feature_extraction_for_files(speaker_ids, feature_type, mfcc_count)
    if create_dataframe:
        if feature_type == 'librosa':
            dam.create_librosa_dataframe(speaker_ids)
        if feature_type == 'psf':
            dam.create_psf_dataframe(speaker_ids)



def feature_extraction_for_files(speaker_ids, feature_type, mfcc_count):
    FEATURES.overwrite_n_mfcc(FEATURES, mfcc_count)
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            if feature_type == 'librosa':
                flib.extract_mfcc_from_file_to_json(file_path)
            if feature_type == 'psf':
                fpsf.extract_mfcc_from_file_to_json(file_path)