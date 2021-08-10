import logging
import multiprocessing
import os

import sounddevice as sd
from configparser import ConfigParser

import scipy.io.wavfile as wav

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib
from utils import dataframeManager as dam, directoryManager as dm, util

file = rf'{dm.get_project_path()}/config.ini'
config = ConfigParser()
config.read(file)

def get_voice_input_stream(timespan, samplerate, number, speaker_id, test):
    logging.info("collecting voice samples....")
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


def feature_extraction_for_n_speaker(speaker_ids, create_dataframe):
    if len(speaker_ids) > 9:
    # if False:
        PROCESSES = config.getint('system', 'PROCESSES')
        split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, PROCESSES)
        pool = multiprocessing.Pool(processes=PROCESSES)
        data = []
        for x in range(PROCESSES):
            data.append((split_speaker_ids[x], config.get('features', 'FEATURE_TYPE')))
        pool.starmap(feature_extraction_for_files, data)
        pool.close()
        pool.join()
    else:
        feature_extraction_for_files(speaker_ids, config.get('features', 'FEATURE_TYPE'))
    if create_dataframe:
        if config.get('features', 'FEATURE_TYPE') == 'librosa':
            dam.create_librosa_dataframe(speaker_ids)
        if config.get('features', 'FEATURE_TYPE') == 'psf':
            dam.create_psf_dataframe(speaker_ids)


def feature_extraction_for_files(speaker_ids, feature_type):
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_path = rf'{dm.get_parent_path(speaker_id)}/{file}'
            if feature_type == 'librosa':
                flib.extract_features_from_file_to_json(file_path)
            if feature_type == 'psf':
                fpsf.extract_features_from_file_to_json(file_path)