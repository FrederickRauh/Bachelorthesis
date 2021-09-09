import pickle

from utils import directoryManager as dm
"""
This file is responsible for saving and loading models
"""


def save_model(speaker_id, t, model, sub_path=None):
    model_path = dm.get_model_path(speaker_id, t, sub_path=sub_path)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, t, sub_path=None):
    model_path = dm.get_model_path(speaker_id, t, sub_path=sub_path)
    return pickle.load(open(model_path, 'rb'))


def get_model_best_estimator_(speaker_id, t, sub_path=None):
    model = load_model(speaker_id, t, sub_path=sub_path)
    return model['gridsearchcv'].best_params_
