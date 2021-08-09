import os
import json
import logging

import numpy as np
import pandas as pd

from utils import directoryManager as dm, util
from config import CONFIG as config

"""
DataframeManager is used when working with a dataframe/json file containing information for the training / predicting phase.
This file shall only contain methods that create, load, save dataframes to json as also output content of said files.
"""


def create_librosa_dataframe(speaker_ids):
    """
    create_librosa_dataframe (creates the dataframe which links all wav files to the corresponding json files containing
    their features)
    :arg speaker_ids
    """
    logging.info("creating librosa dataframe... ")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_name = speaker_id + '\\' + file
            all_features.append([speaker_id, file_name])
    features_dataframe = pd.DataFrame(all_features, columns=['speaker_id', 'file_name'])
    dataframe_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
    save_dataframe_to_json_file(features_dataframe, dataframe_path)
    return features_dataframe


def create_psf_dataframe(speaker_ids):
    """
    create_psf_dataframe (creates the dataframe which links all wav files to the corresponding json files containing
    their features)
    :arg speaker_ids
    """
    logging.info("creating psf dataframe... ")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        if dm.is_large_data_set():
            files = files[:len(files) - 10]
        for file in files:
            file_name = speaker_id + '\\' + file
            all_features.append([speaker_id, file_name])
    features_dataframe = pd.DataFrame(all_features, columns=['speaker_id', 'file_name'])
    dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
    save_dataframe_to_json_file(features_dataframe, dataframe_path)
    return features_dataframe


def save_dataframe_to_json_file(dataframe, path):
    dm.check_if_file_exists_then_remove(path)
    dataframe.to_json(path)


def load_dataframe_from_path(path):
    dataframe = pd.read_json(path)
    return dataframe
