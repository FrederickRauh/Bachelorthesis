import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from frontend import frontend as fr
from utils import directoryManager as dm
from utils import csvManager as cm
import matplotlib.pyplot as plt


def get_dataset(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    return pd.read_csv(csv_path)


class Trainer(object):

    def __init__(self):
        print()

    def train_svm(self, speaker_id):
        all_data_csv = pd.read_csv(dm.get_all_data_csv_file_path())
        csv_content = pd.read_csv(dm.get_csv_path(speaker_id))

        fbanks = csv_content.features
        id = csv_content.id
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
        x = []
        data_path = dm.get_all_data_path()
        for file in files_2:
            fbank =fr.extract_features_from_file(data_path + '\\' + file)
            fbank_small = fbank[1: 3,:]
            x.append(fbank_small)

        x = np.array(x)

        nsamples, nx, ny = x.shape
        d2_train_dataset = x.reshape((nsamples, nx * ny))

        # print("same: ", same)
        # print("file_2: ", x)

        clf_svm = svm.SVC(kernel="linear", C=2)
        clf_svm.fit(d2_train_dataset, y)

        y_pred_svm = clf_svm.predict(d2_train_dataset)
        accuracy_svm = metrics.accuracy_score(y, y_pred_svm)

        print(accuracy_svm)

    def train(self):
        tf.reset_default_graph()
