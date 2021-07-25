import pandas as pd

from backend.svm import svm_model as svm
from backend.gmm import gmm_model as gmm

from utils import dataframeManager as dam, directoryManager as dm


def get_dataset(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    return pd.read_csv(csv_path)


class Trainer(object):

    def __init__(self):
        pass

    def train_svm(self, dataframe, speaker_id, feature_type):
        svm.create_model(speaker_id, dataframe, feature_type)

    def train_multi_svm(self, speaker_ids, feature_type):
        dataframe_librosa_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
        dataframe_librosa = dam.load_dataframe_from_path(dataframe_librosa_path)
        for speaker_id in speaker_ids:
            self.train_svm(dataframe_librosa, speaker_id, feature_type)

    def train_gmm(self, dataframe, speaker_id, feature_type):
        gmm.create_model(speaker_id, dataframe, feature_type)

    def train_multi_gmm(self, speaker_ids, feature_type):
        dataframe_librosa_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
        dataframe_librosa = dam.load_dataframe_from_path(dataframe_librosa_path)
        for speaker_id in speaker_ids:
            self.train_gmm(dataframe_librosa, speaker_id, feature_type)