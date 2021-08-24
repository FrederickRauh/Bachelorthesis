import copy
import json
import logging
import math
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, jsonManager as jm, modelManager as m, plotter as p, util, \
    resultManager as rm, \
    trainingTestingManager as tt


class GMMUBM(object):
    config = None

    def __init__(self):
        file = rf'{dm.get_project_path()}/config.ini'
        config = ConfigParser()
        config.read(file)
        self.feature_type = config.get('features', 'feature_type')
        self.ubm_param_grid = [{
            'n_components': json.loads(config.get('gmm-ubm', 'ubm_n_components')),
            'max_iter': json.loads(config.get('gmm-ubm', 'ubm_max_iter')),
            'covariance_type': json.loads(config.get('gmm-ubm', 'ubm_covariance_type')),
            'n_init': json.loads(config.get('gmm-ubm', 'ubm_n_init'))
        }]
        self.gmm_param_grid = [{
            'n_components': json.loads(config.get('gmm-ubm', 'gmm_n_components')),
            'max_iter': json.loads(config.get('gmm-ubm', 'gmm_max_iter')),
            'covariance_type': json.loads(config.get('gmm-ubm', 'gmm_covariance_type')),
            'n_init': json.loads(config.get('gmm-ubm', 'gmm_n_init'))
        }]
        self.CV = config.getint('modelconfig', 'cv')
        self.REFIT = config.getboolean('modelconfig', 'refit')
        self.N_JOBS = config.getint('modelconfig', 'n_jobs')
        self.VERBOSE = config.getint('modelconfig', 'verbose')

        self.PROCESSES = config.getint("system", "processes")
        self.FEATURE_THRESHOLD = config.getfloat("gmm-ubm", "threshold")
        self.CREATE_SINGLE_RESULT = config.getboolean("result", "create_single_results")

    """
    # Training phase
    # UBM - Universal Background model
    """

    def create_ubm(self, speaker_ids):
        start_time = datetime.now()
        logging.info(f"Training ubm_model with {self.feature_type} features. Start at: {start_time}")
        all_training_features, _ = tt.get_data_for_training('gmm-ubm-ubm', speaker_ids=speaker_ids,
                                                            feature_type=self.feature_type)
        logging.info(f" ::: There are: {len(all_training_features)} trainingfiles. It took {util.get_duration(start_time)} to get files.")

        ubm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(GaussianMixture(),
                         param_grid=self.ubm_param_grid,
                         cv=self.CV,
                         refit=self.REFIT,
                         n_jobs=self.N_JOBS,
                         verbose=self.VERBOSE
                         )
        ).fit(all_training_features)
        logging.info(f"{ubm_model['gridsearchcv'].best_params_}")

        t = 'gmm_ubm_universal_background_model_' + self.feature_type
        m.save_model('', t, ubm_model)
        # p.draw_plt(files=all_training_features, model_path=t,
        #               name='', type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def create_speaker_model(self, speaker_id):
        start_time = datetime.now()
        logging.info(f"Training gmm_model with {self.feature_type} features for: {speaker_id}. Start at: {start_time}")
        training_features, _ = tt.get_data_for_training('gmm-ubm-gmm', [speaker_id], self.feature_type)
        logging.info(f" ::: There are: {len(training_features)} trainingfiles. It took {util.get_duration(start_time)} to get files.")

        ubm_model = m.load_model('', 'gmm_ubm_universal_background_model_' + self.feature_type)['gridsearchcv'].best_estimator_

        adaptive_values = {
                'covariances': ubm_model.covariances_,
                'means': ubm_model.means_,
                'weights': ubm_model.weights_,
                'converged': ubm_model.converged_,
                'threshold': ubm_model.tol
            }

        means = adaptive_values['means']
        # print(len(means))
        # means_list = []
        # for mean in adaptive_values['means']:
        #     means_list.append(np.ndarray.tolist(mean))
        #
        # self.gmm_param_grid[0].update({"means_init": means})

        gmm = BayesianGaussianMixture()
        gmm.means_init = means

        gmm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(gmm,
                         param_grid=self.gmm_param_grid,
                         cv=self.CV,
                         refit=self.REFIT,
                         n_jobs=self.N_JOBS,
                         verbose=self.VERBOSE,
                         return_train_score=True
                         )
        ).fit(training_features)
        logging.info(f"{gmm_model['gridsearchcv'].best_params_}")

        t = 'gmm_ubm_single_model_' + self.feature_type
        m.save_model(speaker_id, t, gmm_model)
        # p.draw_plt(files=training_features, model_path=t, name=speaker_id, type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        self.create_ubm(speaker_ids=speaker_ids)

        for speaker_id in speaker_ids:
            self.create_speaker_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """
    def predict_n_speakers(self, speaker_ids, test_files, extra_data_object):
        # _, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
        ubm_model = m.load_model('', 'gmm_ubm_universal_background_model_' + self.feature_type)
        gmm_models = [m.load_model(speaker_id, "gmm_ubm_single_model_" + self.feature_type) for speaker_id in speaker_ids]
        if self.PROCESSES > 1:
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
            split_gmm_models = util.split_array_for_multiprocess(gmm_models, self.PROCESSES)
            data = []
            for i in range(self.PROCESSES):
                data.append((split_speaker_ids[i], ubm_model, split_gmm_models[i], test_files))

            logging.info(f"starting multi process:{len(split_speaker_ids)}")
            pool = multiprocessing.Pool(processes=self.PROCESSES)
            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        else:
            logging.info(f"Starting single thread with: {len(speaker_ids)} ids")
            results = [self.predict_mult(speaker_ids, ubm_model, gmm_models, test_files)]

        overall_results = []
        for result in results:
            overall_results += result

        rm.create_overall_result_json(overall_results, 'gmmubm-' + self.feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, ubm_model, gmm_models, test_files):
        part_results = []
        for x in range(len(speaker_ids)):
            part_results.append([self.predict_speaker(speaker_ids[x], ubm_model, gmm_models[x], test_files)])
        return part_results

    def predict_file(self, ubm_model, gmm_model, file_path):
        features = am.get_features_for_prediction(file_path, self.feature_type)

        score_gmm = gmm_model.score_samples(features)
        score_ubm = ubm_model.score_samples(features)

        overall_score = (sum(score_gmm) / sum(score_ubm))


        if overall_score < self.FEATURE_THRESHOLD:
            return 1
        else:
            return 0

    def predict_speaker(self, speaker_id, ubm_model, gmm_model, test_files):
        start_time = datetime.now()
        speaker_object_result = {}

        score_of_files = []
        for file in test_files:
                score_of_files.append(self.predict_file(ubm_model, gmm_model, file))

        speaker_object_result.update(
                rm.sort_results_and_create_speaker_object(speaker_id, test_files, score_of_files))

        logging.info(f"{util.get_duration(start_time)}")

        if self.CREATE_SINGLE_RESULT:
            rm.create_single_result_json(speaker_id, 'gmmubm-' + self.feature_type, [[{speaker_id: speaker_object_result}]])

        return {speaker_id: speaker_object_result}