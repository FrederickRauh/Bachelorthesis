import pickle

from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils import dataframeManager as dam, directoryManager as dm, util


def save_model(speaker_id, t, model):
    model_path = dm.get_model_path(speaker_id, t)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, t):
    model_path = dm.get_model_path(speaker_id, t)
    return pickle.load(open(model_path, 'rb'))


def create_model(speaker_id, dataframe, feature_type):
    training_features, is_speaker = dam.get_data_for_training_from_dataframe('svm', speaker_id, dataframe, feature_type)
    start_time = datetime.now()
    print("Training svm_model with", feature_type, "for:", speaker_id, ":: There are:", len(training_features),
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
    kernels = ['rbf']
    C = np.arange(0.1, 5.1, 0.1)
    C = [round(x, 2) for x in C]
    gamma = ['auto', 'scale']
    param_grid = [{
        'kernel': kernels,
        'C': C,
        'gamma': gamma
    }]
    # --------------------- GridSearch ------------------------
    cv = KFold(n_splits=4)
    # helpful with large datasets to keep an overview
    # n_jobs = -1 use all cpus, -2 use all but one
    verbose = 0
    n_jobs = -2
    if dm.is_large_data_set():
        verbose = 3
        n_jobs = -1

    # svm_pipe = make_pipeline(StandardScaler(), SVC( kernel='rbf', gamma=0.01, C=10, probability=True))

    svm_model = make_pipeline(
        StandardScaler(),
        GridSearchCV(SVC(),
                     param_grid=param_grid,
                     cv=10,
                     refit=True,
                     n_jobs=n_jobs,
                     verbose=verbose)
    )

    svm_model.fit(training_features, is_speaker)

    # helpful with large datasets to keep an overview
    if dm.is_large_data_set():
        print(svm_model['gridsearchcv'].best_params_)
    # scaled_training_features = scaler.fit_transform(training_features)
    #
    # svm_model_custom.fit(scaled_training_features, is_speaker)
    # score = model_selection.cross_val_score(svm_model_custom, files, is_speaker, cv=10, scoring='accuracy')
    save_model(speaker_id, 'svm_custom_' + feature_type, svm_model)

    print(util.get_duration(start_time))
