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

from utils import directoryManager as dm, util


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def get_training_features(speaker_id, dataframe):
    t = []
    wav_files = dm.get_wav_files(speaker_id)
    for wav_file in wav_files:
        file = speaker_id + '\\' + wav_file
        t.append(file)

    training_files = []
    for element in t:
        training_features = dataframe.loc[dataframe['file_name'] == element].feature.array[0]['0']
        training_files.append(training_features)
    return util.get_correct_array_form(training_files)


def train_test_split(features, is_speaker, test_size=0.1):
    features_train, features_test, speaker_train, speaker_test = sklearn.model_selection.train_test_split(features, is_speaker,
                                                                                                    test_size=test_size)
    x_train = np.array(features_train)
    x_test = np.array(features_test)
    y_train = np.array(speaker_train)
    y_test = np.array(speaker_test)
    return x_train, x_test, y_train, y_test


def create_model(speaker_id, dataframe):
    training_features = get_training_features(speaker_id, dataframe)
    start_time = datetime.now()
    print("Training gmm_model for:", speaker_id, " :: There are:", len(training_features),
          "trainingfiles. Start at: ", start_time)
    gmm_model = gmm(n_components=16, max_iter=200, covariance_type='diag', n_init=3).fit(training_features)
    save_model(speaker_id, 'gmm', gmm_model)

    after_time = datetime.now()
    duration = after_time - start_time
    hours = duration.total_seconds() // 3600
    minutes = (duration.total_seconds() // 60) - (hours * 60)
    seconds = duration.total_seconds() - (hours * 3600) - (minutes * 60)
    print("--> duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds),
          # "----- Model: accuracy: %f; standard deviation of %f" % (score.mean(), score.std())
          )
