import pandas as pd

from backend.svm import svm_model as svm
from backend.gmm import gmm_model as gmm

from utils import directoryManager as dm


def get_dataset(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    return pd.read_csv(csv_path)


class Trainer(object):

    def __init__(self):
        pass

    def train_svm(self, dataframe, speaker_id):
        svm.create_model(speaker_id, dataframe)

    def train_multi_svm(self, speaker_ids, dataframe):
        for speaker_id in speaker_ids:
            self.train_svm(dataframe, speaker_id)

    def train_gmm(self, dataframe, speaker_id):
        gmm.create_model(speaker_id, dataframe)

    def train_multi_gmm(self, speaker_ids, dataframe):
        for speaker_id in speaker_ids:
            self.train_gmm(dataframe, speaker_id)