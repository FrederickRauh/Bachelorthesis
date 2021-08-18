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
        self.feature_type = config.get('features', 'FEATURE_TYPE')
        self.ubm_param_grid = [{
            'n_components': json.loads(config.get('gmm-ubm', 'UBM_N_COMPONENTS')),
            'max_iter': json.loads(config.get('gmm-ubm', 'UBM_MAX_ITER')),
            'covariance_type': json.loads(config.get('gmm-ubm', 'UBM_COVARIANCE_TYPE')),
            'n_init': json.loads(config.get('gmm-ubm', 'UBM_N_INIT'))
        }]
        self.gmm_param_grid = [{
            'n_components': json.loads(config.get('gmm-ubm', 'GMM_N_COMPONENTS')),
            'max_iter': json.loads(config.get('gmm-ubm', 'GMM_MAX_ITER')),
            'covariance_type': json.loads(config.get('gmm-ubm', 'GMM_COVARIANCE_TYPE')),
            'n_init': json.loads(config.get('gmm-ubm', 'GMM_N_INIT'))
        }]
        self.CV = config.getint('modelconfig', 'CV')
        self.REFIT = config.getboolean('modelconfig', 'REFIT')
        self.N_JOBS = config.getint('modelconfig', 'N_JOBS')
        self.VERBOSE = config.getint('modelconfig', 'VERBOSE')

        self.PROCESSES = config.getint("system", "PROCESSES")
        self.GMM_THRESHOLD = config.getfloat("gmm-ubm", "GMM_THRESHOLD")
        self.UBM_THRESHOLD = config.getfloat("gmm-ubm", "UBM_THRESHOLD")
        self.FEATURE_THRESHOLD = config.getfloat("system", "FEATURE_THRESHOLD")

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
        speaker_ids.sort()
        logging.info(f"starting training for: {speaker_ids}")
        self.create_ubm(speaker_ids=speaker_ids)
        # ubm_model = m.load_model('', 'gmm_ubm_universal_background_model_' + self.feature_type)
        # all_training_features, _ = tt.get_data_for_training('gmm-ubm-ubm', speaker_ids=speaker_ids,
        #                                                     feature_type=self.feature_type)
        #
        # t = 'gmm_ubm_universal_background_model_' + self.feature_type
        # p.draw_plt(files=all_training_features, model_path=t,
        #            name='', type=t)

        #

        # jm.write_features_to_json_file(rf"E:/voxceleb/vox1_server/data.json", "UBM", adaptive_values)
        for speaker_id in speaker_ids:
            self.create_speaker_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """
    def predict_n_speakers(self, speaker_ids):
        test_files, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
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

    def predict_file(self, speaker_id, ubm_model, gmm_model, file_path):
        features = am.get_features_for_prediction(file_path, self.feature_type)

        ubm_scores = ubm_model.predict_proba(features)
        gmm_scores = gmm_model.predict_proba(features)

        ubm_count = 0
        gmm_count = 0
        for score in ubm_scores:
            for x in range(len(score)):
                if score[x] >= self.UBM_THRESHOLD:
                    ubm_count += 1

        for score in gmm_scores:
            for x in range(len(score)):
                if score[x] >= self.GMM_THRESHOLD:
                    gmm_count += 1

        # print(speaker_id, file_path, gmm_count, ubm_count)
        features_count = len(features)
        ubm_count = math.log(ubm_count / features_count)
        gmm_count = math.log(gmm_count / features_count)

        overall_score = gmm_count / ubm_count

        if overall_score > 0.5:
            return 1
        else:
            return 0

    def predict_speaker(self, speaker_id, ubm_model, gmm_model, test_files):
        start_time = datetime.now()
        speaker_object_result = {}

        true_positive = []
        false_negative = []
        false_positive = []
        true_negative = []

        for file in test_files:
                id_of_file = dm.get_id_of_path(file)
                score = self.predict_file(speaker_id, ubm_model, gmm_model, file)

                if speaker_id == id_of_file:
                    if score == 1:
                        true_positive.append(file)
                    else:
                        false_negative.append(file)
                else:
                    if score == 1:
                        false_positive.append(file)
                    else:
                        true_negative.append(file)

        speaker_object_result.update(
            rm.create_speaker_object(true_positive, true_negative, false_positive, false_negative))

        logging.info(f"{util.get_duration(start_time)}")

        rm.create_single_result_json(speaker_id, 'gmmubm-' + self.feature_type, [[{speaker_id: speaker_object_result}]])

        return {speaker_id: speaker_object_result}