import pickle

import tensorflow
import numpy as np

import sklearn
from sklearn import svm
from sklearn import metrics

from frontend import frontend as fr

from utils import directoryManager as dm


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def create_svm_model(speaker_id, files, is_speaker):
    data_path = dm.get_all_data_path()
    best = 0
    model_to_save = 0
    for i in range(30):
        print("Training svm_model: run number : ", i)
        files_train, files_test, same_train, same_test = sklearn.model_selection.train_test_split(files, is_speaker, test_size=0.1)
        x = []
        for file in files_train:
            features = fr.extract_mfcc_from_file(data_path + '\\' + file)
            features_small = features[1: 3, :]
            x.append(features_small)

        x = np.array(x)
        y = np.array(same_train)

        nsamples, nx, ny = x.shape
        x = x.reshape((nsamples, nx * ny))

        svm_model = svm.SVC(kernel='rbf', gamma='scale')
        svm_model.fit(x, y)

        y_pred_svm = svm_model.predict(files_test)
        accuracy = metrics.accuracy_score(y, y_pred_svm)

        print(accuracy)
        if accuracy > best:
            best = accuracy
            model_to_save = svm_model

    print('model accuracy: ', best)
    save_model(speaker_id, 'svm', model_to_save)




