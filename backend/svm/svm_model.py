import pickle

from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from utils import dataframeManager as dam, directoryManager as dm, util


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def create_model(speaker_id, dataframe):
    training_features, is_speaker = dam.get_data_for_training_from_dataframe('svm', speaker_id, dataframe)
    start_time = datetime.now()
    print("Training svm_model for:", speaker_id, " :: There are:", len(training_features),
          "trainingfiles. Start at: ", start_time)
    # which kernel should be used and why? (Same for gamma)
    # write method to get best c(0.019 vs 2), kernel, etc.
    # choosen after reading paper: Evaluation of kernel methods for speaker verification and identification(2002)
    # and Kernel combination for SVM speaker verification(2008)
    # literature suggests to use rbf or poly for mfccs
    #
    # params = svc.plot_(files, is_speaker)
    # svm_model_custom = SVC(
    #     # kernel=params.get('kernel'),
    #     kernel='rbf',
    #     # gamma=params.get('gamma'), try 0.01 instead of auto,
    #     gamma=0.01,
    #     # C=params.get('C') try: 10 instead of 0.019,
    #     C=10,
    #     probability=True
    #     # degree=params.get('degree')
    # )
    # --------------------------------
    kernels = ['rbf', 'poly']
    C = np.arange(0.1, 5.1, 0.1)
    C = [round(x, 2) for x in C]
    # gamma = np.arange(0.1, 1.1, 0.1)
    gamma = ['auto']
    # degree = np.arange(2, 10, 1)
    # param_grid = dict(kernels=kernels, c=C, gamma=gamma)
    param_grid = [{
        'kernel': kernels,
        'C': C,
        'gamma': gamma
        # 'degree': degree
    }]
    # ---------------------

    scaler = StandardScaler()

    svm_pipe = make_pipeline(StandardScaler(), SVC( kernel='rbf', gamma=0.01, C=10, probability=True))

    cv = KFold(n_splits=4)

    svm_model = make_pipeline(StandardScaler(),
                    GridSearchCV(SVC(),
                                 param_grid=param_grid,
                                 cv=10,
                                 refit=True,
                                 n_jobs=-1, verbose=10))

    svm_model.fit(training_features, is_speaker)
    print(svm_model['gridsearchcv'].best_params_)
    # scaled_training_features = scaler.fit_transform(training_features)
    #
    # svm_model_custom.fit(scaled_training_features, is_speaker)
    # score = model_selection.cross_val_score(svm_model_custom, files, is_speaker, cv=10, scoring='accuracy')
    save_model(speaker_id, 'svm_custom', svm_model)

    print(util.get_duration(start_time))