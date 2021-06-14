import pickle
import csv
import json
import ast
import tensorflow
import numpy as np
import pandas as pd
from datetime import datetime

import sklearn
from sklearn import svm
from sklearn import metrics

from frontend import frontend as fr

from utils import csvManager as cm
from utils import directoryManager as dm


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


def get_correct_feature_array(files):
    # data_path = dm.get_all_data_path()
    x = []
    for file in files:
        # file_path = data_path + '\\' + file
        file_path = file
        wav_path = file_path.replace('.csv', '.wav')
        features = fr.extract_mfcc_from_file(wav_path)
        features_small = features[1: 3, :]
        x.append(features_small)
    return get_correct_array_form(x)


def get_features_out_of_csv(files):
    data_path = dm.get_all_data_path()
    x = []
    for file in files:
        file_path = data_path + '\\' + file
        x.append(get_features_from_csv(file_path))
    return get_correct_array_form(x)


def get_features_from_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = []
        for row in reader:
            if len(row) > 0:
                new_row = []
                for entry in row:
                    new_row.append(float(entry))
                data.append(new_row)
        return data



def create_svm_model(speaker_id, files, is_speaker):
    # data_path = dm.get_all_data_path()
    best = 0
    model_to_save = 0
    training_cycles = 2
    print('files : ', len(files))
    for i in range(training_cycles):
        dateTimeObj = datetime.now()
        print("Training svm_model ::: run : ", i+1, " of ", training_cycles, " at: ", dateTimeObj)
        files_train, files_test, speaker_train, speaker_test = sklearn.model_selection.train_test_split(files, is_speaker, test_size=0.2)

        x_train = get_features_out_of_csv(files_train)
        x_test = get_features_out_of_csv(files_test)
        y_train = np.array(speaker_train)
        y_test = np.array(speaker_test)

        svm_model = svm.SVC(kernel='rbf', gamma='scale')
        svm_model.fit(x_train, y_train)

        y_pred_svm = svm_model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred_svm)

        print("current accuracy : ", accuracy)
        if accuracy > best:
            best = accuracy
            model_to_save = svm_model

    print('model accuracy: ', best)
    save_model(speaker_id, 'svm', model_to_save)




