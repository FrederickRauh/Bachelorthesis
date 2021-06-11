import pickle

import tensorflow
import numpy as np

import sklearn
from sklearn import svm
from sklearn import metrics

from frontend import frontend as fr
from backend import model as m

from utils import directoryManager as dm


class Predictor(object):

    def __init__(self):
        print()

    def predict_svm(speaker_id, features):
        svm_model = m.load_model(speaker_id)
        predicition_score = svm_model.predict(features)
        return predicition_score
