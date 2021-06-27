import tensorflow as tf
import numpy as np
import pandas as pd

from frontend import frontend as fr

from backend import svm_model as m

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

    def train_svm(self, dataframe, speaker_id):
        all_data_csv = pd.read_csv(dm.get_all_data_csv_file_path())

        same = all_data_csv.same
        files_1 = all_data_csv.file_1
        files_2 = all_data_csv.file_2
        filter_arr = []
        i = 0
        for element in files_1:
            if speaker_id in element:
                filter_arr.append(True)
            else:
                filter_arr.append(False)
        y = same[filter_arr]
        files_1 = files_1[filter_arr]
        files_2 = files_2[filter_arr]

        feature_data = dataframe
        features = feature_data.feature
        file_name = feature_data.file_name

        training_files = []
        for element in files_2:
            # i = [x for x in file_name if x == element]
            for x in range(len(file_name)):
                if element == file_name[x]:
                    # print("ELEMENT:", element, "features:", features[x])
                    training_features = features[x]['0']
                    training_files.append(training_features)

        # files = []
        # for file in files_2:
        #     file = file.replace('.wav', '.csv')
        #     files.append(file)
        training_files = util.get_correct_array_form(training_files)
        m.create_svm_model(speaker_id, training_files, y)

    def train_gmm(self):
        tf.reset_default_graph()
