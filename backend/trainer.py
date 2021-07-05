import tensorflow as tf
import numpy as np
import pandas as pd

from datetime import datetime

from frontend import frontend as fr

from backend import svm_model as svm
from backend import gmm_model as gmm

from utils import dataframeManager as dam
from utils import directoryManager as dm
from utils import fileManager as fm
from utils import util
import matplotlib.pyplot as plt

import pickle


def get_dataset(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    return pd.read_csv(csv_path)


class Trainer(object):

    def __init__(self):
        print()

    def get_data_for_training(self, speaker_id, dataframe):
        all_data_csv = pd.read_csv(dm.get_all_data_csv_file_path())
        same = all_data_csv.same
        files_1 = all_data_csv.file_1
        files_2 = all_data_csv.file_2

        filter_arr = []
        for element in files_1:
            if speaker_id in element:
                filter_arr.append(True)
            else:
                filter_arr.append(False)

        y = same[filter_arr]
        files_2 = files_2[filter_arr]

        training_files = self.get_training_files(dataframe, files_2)

        return training_files, y

    def get_training_files(self, dataframe, files):
        training_files = []
        for element in files:
            training_features = dataframe.loc[dataframe['file_name'] == element].feature.array[0]['0']
            training_files.append(training_features)
        return util.get_correct_array_form(training_files)

    def train_svm(self, dataframe, speaker_id):
        print("started training svm_model for: ", speaker_id)
        training_files, y = self.get_data_for_training(speaker_id, dataframe)
        svm.create_model(speaker_id, training_files, y)

    def train_gmm(self, dataframe, speaker_id):
        print("started training gmm_model for:", speaker_id)
        training_files, y = self.get_data_for_training(speaker_id, dataframe)
        gmm.create_model(speaker_id, training_files, y)
