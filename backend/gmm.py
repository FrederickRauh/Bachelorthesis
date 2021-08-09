import logging
import multiprocessing
import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, modelManager as m, util, \
    resultManager as rm, trainingTestingManager as tt


class GMM(object):

    def __init__(self):
        file = dm.get_project_path() + '\\' + 'config.ini'
        config = ConfigParser()
        config.read(file)

        self.feature_type = config.get('features', 'FEATURE_TYPE')

        self.param_grid = [{
            'n_components': [16],
            'max_iter': [500],
            'covariance_type': ['diag'],
            'n_init': [3]
        }]

        self.CV = config.getint('modelconfig', 'CV')
        self.REFIT = config.getboolean('modelconfig', 'REFIT')
        self.N_JOBS = config.getint('modelconfig', 'N_JOBS')
        self.VERBOSE = config.getint('modelconfig', 'VERBOSE')

        self.PROCESSES = config.getint("system", "PROCESSES")

    """
    # Training phase
    """

    def create_model(self, speaker_id):
        training_features, _ = tt.get_data_for_training('gmm', [speaker_id], self.feature_type)
        start_time = datetime.now()
        logging.info(
            f"Training gmm_model with {self.feature_type} features for: {speaker_id} :: There are: {len(training_features)} trainingfiles. Start at: {start_time}")

        gmm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(GaussianMixture(),
                         param_grid=self.param_grid,
                         cv=self.CV,
                         refit=self.REFIT,
                         n_jobs=self.N_JOBS,
                         verbose=self.VERBOSE
                         )
        )
        gmm_model.fit(training_features)
        m.save_model(speaker_id, 'gmm_' + self.feature_type, gmm_model)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """

    def get_test_files_and_extra_data(self, speaker_ids):
        test_files = tt.load_test_files(speaker_ids)
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        return test_files, extra_data_object

    def predict_n_speakers(self, speaker_ids):
        test_files, extra_data_object = self.get_test_files_and_extra_data(speaker_ids=speaker_ids)

        split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
        logging.info(f"starting mult process:{len(split_speaker_ids)}")
        pool = multiprocessing.Pool(processes=self.PROCESSES)

        data = []
        for x in range(self.PROCESSES):
            data.append((split_speaker_ids[x], test_files))

        results = pool.starmap(self.predict_mult, data)
        pool.close()
        pool.join()

        # results = self.predict_mult(speaker_ids, test_files)

        overall_results = []
        for result in results:
            overall_results += result

        rm.create_result_json(overall_results, 'gmm-' + self.feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files):
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, dm.get_all_ids(), test_files, self.feature_type)])
        return part_results

    def predict_for_all_models(self, models, ids, file):
        x = am.get_features_for_prediction(file, self.feature_type)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm_model = models[i]
            scores = np.array(gmm_model.score(x))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        winner_id = ids[winner]
        return winner_id

    def predict_for_one_model(self, model, file):
        x = am.get_features_for_prediction(file, self.feature_type)
        score = model.predict_proba(x)
        return score.sum()

    def predict_speaker(self, speaker_id, speaker_ids, test_files):
        speaker_object_result = {}
        t = "gmm_" + self.feature_type

        true_positive = []
        false_negative = []
        false_positive = []
        true_negative = []

        models = [m.load_model(speaker_id, t) for speaker_id in speaker_ids]
        ids_of_models = [id for id in speaker_ids]

        for file in test_files:
            id_of_file = dm.get_id_of_path(file)
            winner = self.predict_for_all_models(models, ids_of_models, file, self.feature_type)

            if winner == speaker_id:
                if winner == id_of_file:  # match is the speaker
                    true_positive.append(file)
                else:  # imposter file
                    false_positive.append(file)
            else:
                if dm.get_id_of_path(file) == speaker_id:  # should have matched but failed.
                    false_negative.append(file)
                else:  # matches to a model, that is not owner of file
                    true_negative.append(file)

        speaker_object_result.update(
            rm.create_speaker_object(true_positive, true_negative, false_positive, false_negative))

        return {speaker_id: speaker_object_result}
