import pickle
import csv
import numpy as np
from datetime import datetime

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


def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


def turn_list_to_array(list_array):
    array = []
    for entries in list_array:
        under_array = []
        for entry in entries:
            under_array.append(np.asarray(entry))
        array.append(under_array)
    return array


def get_correct_feature_array(files):
    x = []
    for file in files:
        file_path = file
        wav_path = file_path.replace('.csv', '.wav')
        features = fr.extract_mfcc_from_file(wav_path)
        # decide which feature array to use
        features_small = features[1: 3, :]
        feature_array = features_small
        x.append(feature_array)
    return x
    # return get_correct_array_form(x)


def get_features_out_of_csv(files):
    data_path = dm.get_all_data_path()
    x = []
    for file in files:
        file_path = data_path + '\\' + file
        x.append(get_features_from_csv(file_path))
    return x
    # return get_correct_array_form(x)


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
    best = 0
    model_to_save = 0
    training_cycles = 2
    for i in range(training_cycles):
        dateTimeObj = datetime.now()
        print("Training svm_model ::: run : ", i+1, " of ", training_cycles, "; There are:", len(files), "trainingfiles. Start at: ", dateTimeObj)
        files_train, files_test, speaker_train, speaker_test = sklearn.model_selection.train_test_split(files, is_speaker, test_size=0.2)

        # print(get_features_out_of_csv(files_train)[0][0])

        x_train = get_features_out_of_csv(files_train)
        x_test = get_features_out_of_csv(files_test)
        y_train = np.array(speaker_train)
        y_test = np.array(speaker_test)

        print(x_train)

        svm_model = svm.SVC(kernel='rbf', gamma='scale')
        print("x_train : ", len(x_train))
        print("y_train : ", len(y_train))
        svm_model.fit(x_train, y_train)

        y_pred_svm = svm_model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred_svm)

        print("current accuracy : ", accuracy)
        if accuracy > best:
            best = accuracy
            model_to_save = svm_model

    print('model accuracy: ', best)
    save_model(speaker_id, 'svm', model_to_save)




