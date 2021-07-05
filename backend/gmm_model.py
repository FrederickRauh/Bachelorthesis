import pickle

import numpy as np
from datetime import datetime

import sklearn
from sklearn.mixture import GaussianMixture as gmm
from sklearn import gaussian_process, metrics

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
    best = 0
    model_to_save = 0
    training_cycles = 4
    for i in range(training_cycles):
        start_time = datetime.now()
        print("Training svm_model ::: run : ", i+1, " of ", training_cycles, "; There are:", len(files), "trainingfiles. Start at: ", start_time)
        x_train, x_test, y_train, y_test = train_test_split(files, is_speaker)
        train_data = [x_train, y_train]
        # why did i do  this?!
        gmm_model = gmm(n_components=19, covariance_type='diag').fit(train_data)

        y_pred_svm = gmm.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred_svm)

        after_time = datetime.now()
        duration = after_time - start_time

        print("model training took:", duration.total_seconds() // 60, "minutes;"
              , "current accuracy : ", accuracy)
        if accuracy > best:
            best = accuracy
            model_to_save = gmm_model

    print('model accuracy: ', best)
    save_model(speaker_id, 'svm', model_to_save)


def train_model():
    pass