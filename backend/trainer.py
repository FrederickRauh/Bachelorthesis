import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pyannote.database as db

from utils import directoryManager as dm

import matplotlib.pyplot as plt


def get_dataset(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    return pd.read_csv(csv_path)



class Trainer(object):

    def __init__(self):
        print('hello')

    def train_svm(self, name):
        # dataset = get_dataset(name)
        # x = dataset.features
        voxceleb = db.get_database('VoxCeleb')
        for protocol_name in voxceleb.get_protocols('SpeakerVerification'):
            print(f'VoxCeleb.SpeakerVerification.{protocol_name}')

    def train(self):
        tf.reset_default_graph()
