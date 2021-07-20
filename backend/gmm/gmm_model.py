import pickle

from datetime import datetime

from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import dataframeManager as dam, directoryManager as dm, util, debug


def save_model(speaker_id, t, model):
    model_path = dm.get_model_path(speaker_id, t)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, t):
    model_path = dm.get_model_path(speaker_id, t)
    return pickle.load(open(model_path, 'rb'))


def create_model(speaker_id, dataframe, feature_type):
    training_features = dam.get_data_for_training_from_dataframe('gmm', speaker_id, dataframe, feature_type)
    start_time = datetime.now()
    debug.log(("Training gmm_model with", feature_type, "features for:", speaker_id, ":: There are:", len(training_features),
          "trainingfiles. Start at: ", start_time))

    n_components = [13, 16]
    max_iter = [200]
    covariance_type = ['diag']
    n_init = [3]

    param_grid = [{
        'n_components':  n_components,
        'max_iter': max_iter,
        'covariance_type': covariance_type,
        'n_init': n_init,
    }]
    # --------------------- GridSearch ------------------------
    # helpful with large datasets to keep an overview
    # n_jobs = -1 use all cpus, -2 use all but one
    verbose = 0
    n_jobs = -2
    if dm.is_large_data_set():
        verbose = 0
        n_jobs = -1

    gmm_model = make_pipeline(
        StandardScaler(),
        GridSearchCV(GMM(),
                     param_grid=param_grid,
                     cv=10,
                     refit=True,
                     n_jobs=n_jobs,
                     verbose=verbose
                     )
    )
    gmm_model.fit(training_features)
    if dm.is_large_data_set():
        debug.log((gmm_model['gridsearchcv'].best_params_))
    # gmm_model = gmm(n_components=16, max_iter=200, covariance_type='diag', n_init=3).fit(training_features)
    save_model(speaker_id, 'gmm_' + feature_type, gmm_model)

    debug.log(util.get_duration(start_time))
