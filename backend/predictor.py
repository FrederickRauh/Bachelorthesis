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
        pass

    def predict_svm(self, speaker_id, file_path):
        prediction_array = []
        prediction_array.append(file_path)
        x = m.get_correct_feature_array(prediction_array)
        svm_model = m.load_model(speaker_id, 'svm')
        y = x
        score = svm_model.predict(y)
        return score
