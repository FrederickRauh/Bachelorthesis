import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

import pyannote.database as db

from utils import directoryManager as dm

import matplotlib.pyplot as plt


def get_dataset(name):
    csv_path = dm.get_csv_path(name)
    return pd.read_csv(csv_path)



class Trainer(object):

    def __init__(self):
        print('hello')

    def train_svm(self, name):
        # dataset = get_dataset(name)
        # x = dataset.features
        voxceleb = db.get_database('VoxCeleb')

        print(voxceleb)

    def train(self):
        tf.reset_default_graph()
