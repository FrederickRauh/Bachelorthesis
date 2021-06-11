import tensorflow as tf
import numpy as np
import pandas as pd

from frontend import frontend as fr

from backend import model as m

from utils import directoryManager as dm
from utils import csvManager as cm
import matplotlib.pyplot as plt

import pickle

def get_dataset(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    return pd.read_csv(csv_path)


class Trainer(object):

    def __init__(self):
        print()

    def train_svm(self, speaker_id):
        all_data_csv = pd.read_csv(dm.get_all_data_csv_file_path())
        # csv_content = pd.read_csv(dm.get_csv_path(speaker_id))
        #
        # fbanks = csv_content.features
        # id = csv_content.id
        same = all_data_csv.same
        files_1 = all_data_csv.file_1
        files_2 = all_data_csv.file_2

        filter_arr = []
        i = 0;
        for element in files_1:
            if speaker_id in element:
                filter_arr.append(True)
            else:
                filter_arr.append(False)
        y = same[filter_arr]
        files_1 = files_1[filter_arr]
        files_2 = files_2[filter_arr]

        m.create_svm_model(speaker_id, files_2, y)

        # x = []
        # data_path = dm.get_all_data_path()
        # for file in files_2:
        #     features =fr.extract_mfcc_from_file(data_path + '\\' + file)
        #     features_small = features[1: 3,:]
        #     x.append(features_small)
        #
        # x = np.array(x)
        #
        # nsamples, nx, ny = x.shape
        # x = x.reshape((nsamples, nx * ny))

        # print("same: ", same)
        # print("file_2: ", x)




    def train(self):
        tf.reset_default_graph()
