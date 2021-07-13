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

from utils import dataframeManager as dam, directoryManager as dm, util


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def create_model(speaker_id, dataframe):
    training_features = dam.get_data_for_training_from_dataframe('gmm', speaker_id, dataframe)
    start_time = datetime.now()
    print("Training gmm_model for:", speaker_id, " :: There are:", len(training_features),
          "trainingfiles. Start at: ", start_time)

    gmm_model = gmm(n_components=16, max_iter=200, covariance_type='diag', n_init=3).fit(training_features)
    save_model(speaker_id, 'gmm', gmm_model)

    print(util.get_duration(start_time))
