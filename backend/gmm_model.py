import pickle

import numpy as np
from datetime import datetime

import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as gmm
from sklearn import gaussian_process, metrics, model_selection
from sklearn.preprocessing import StandardScaler

from frontend import featureExtractorPSF as fe

from utils import directoryManager as dm


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def train_test_split(features, is_speaker, test_size=0.1):
    features_train, features_test, speaker_train, speaker_test = sklearn.model_selection.train_test_split(features, is_speaker,
                                                                                                    test_size=test_size)
    x_train = np.array(features_train)
    x_test = np.array(features_test)
    y_train = np.array(speaker_train)
    y_test = np.array(speaker_test)
    return x_train, x_test, y_train, y_test


def create_model(speaker_id, files, is_speaker):
    pass
    # X, y_true = files, is_speaker
    #
    # X = X[:, ::-1]
    # gmm_model = gmm(n_components=16)
    #
    # labels = gmm_model.fit(X).predict(X)
    #
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    #
    # plt.show()

    # start_time = datetime.now()
    # print("Training gmm_model :: There are:", len(files),
    #       "trainingfiles. Start at: ", start_time)
    # gmm_model = gmm(n_components=16, max_iter=200, covariance_type='diag', n_init=3).fit(files)
    # score = model_selection.cross_val_score(gmm_model, files, is_speaker, cv=5, scoring='accuracy')
    #
    # after_time = datetime.now()
    # duration = after_time - start_time
    # hours = duration.total_seconds() // 3600
    # minutes = duration.total_seconds() // 60
    # seconds = duration.total_seconds() - (duration.total_seconds() // 60)
    # print("duration: %0.0fh:%0.0fmin:%0.2fsec; accuracy: %f; standard deviation of %f" % (hours, minutes, seconds, score.mean(), score.std()))
    #
    # save_model(speaker_id, 'gmm', gmm_model)


def train_model():
    pass