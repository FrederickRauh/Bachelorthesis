import pickle

import numpy as np
from datetime import datetime

import sklearn
from sklearn import svm
from sklearn import metrics

from frontend import featureExtractorPSF as fe

from utils import directoryManager as dm


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def create_svm_model(speaker_id, files, is_speaker):
    best = 0
    model_to_save = 0
    training_cycles = 4
    for i in range(training_cycles):
        dateTimeObj = datetime.now()
        print("Training svm_model ::: run : ", i+1, " of ", training_cycles, "; There are:", len(files), "trainingfiles. Start at: ", dateTimeObj)
        files_train, files_test, speaker_train, speaker_test = sklearn.model_selection.train_test_split(files, is_speaker, test_size=0.1)

        # print(get_features_out_of_csv(files_train)[0][0])

        # x_train = get_features_out_of_csv(files_train)
        # x_test = get_features_out_of_csv(files_test)
        x_train = np.array(files_train)
        x_test = np.array(files_test)
        y_train = np.array(speaker_train)
        y_test = np.array(speaker_test)

        # print(x_train)

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




