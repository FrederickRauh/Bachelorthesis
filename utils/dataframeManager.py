import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm


def create_dataframe(speaker_ids):
    print("creating dataframe")
    all_features = []
    for speaker_id in speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file

            # flib.extract_mfcc_from_file_to_csv(file_path)
            # features = flib.extract_processed_mfcc_from_file(file_path)

            # features = flib.load_features_from_csv(file_path)
            features = fpsf.load_features_from_csv(file_path)
            print(features)

            file_name = speaker_id + '\\' + file
            all_features.append([features, speaker_id, file_name])
    features_dataframe = pd.DataFrame(all_features, columns=['feature', 'speaker_id', 'file_name'])
    save_dataframe_to_file(features_dataframe)
    return features_dataframe


def save_dataframe_to_file(dataframe):
    dataframe_path = dm.get_all_data_path() + '\\' + 'dataframe.csv'
    dm.check_if_file_exists_then_remove(dataframe_path)
    dataframe.to_csv(dataframe_path)


def load_dataframe():
    dataframe_path = dm.get_all_data_path() + '\\' + 'dataframe.csv'
    return pd.read_csv(dataframe_path)
