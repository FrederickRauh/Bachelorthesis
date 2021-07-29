import pickle

from utils import directoryManager as dm
"""
This file is responsible for saving and loading models
"""


def save_model(speaker_id, t, model):
    model_path = dm.get_model_path(speaker_id, t)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, t):
    model_path = dm.get_model_path(speaker_id, t)
    return pickle.load(open(model_path, 'rb'))
