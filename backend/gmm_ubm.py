import logging
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from utils import audioManager as am, dataframeManager as dam, modelManager as m, util, \
    resultManager as rm
from config import MODELCONFIG, GMM as gmm_config, SYSTEM, FEATURES, IDS


class GMMUBM(object):

    def __init__(self):
        pass

    """
    # Training phase
    # UBM - Universal Background model
    """

    def create_ubm(self, speaker_id):
        all_training_features = dam.get_data_for_training('gmm-ubm', IDS.SPEAKER_IDS)
        logging.info(
            f"Training gmm_model with {SYSTEM.FEATURE_TYPE} features for: {IDS.SPEAKER_IDS} :: There are: "
            f"{len(all_training_features)} training files.")
        param_grid = [{
            'n_components': gmm_config.N_COMPONENTS,
            'max_iter': gmm_config.MAX_ITER,
            'covariance_type': gmm_config.COVARIANCE_TYPE,
            'n_init': gmm_config.N_INIT,
        }]

        ubm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(GaussianMixture(),
                         param_grid=param_grid,
                         cv=MODELCONFIG.CV,
                         refit=MODELCONFIG.REFIT,
                         n_jobs=MODELCONFIG.N_JOBS,
                         verbose=MODELCONFIG.VERBOSE
                         )
        )
        ubm_model.fit(all_training_features)
        m.save_model(speaker_id, 'gmm_ubm_universal_background_model_' + SYSTEM.FEATURE_TYPE, ubm_model)


    def create_single_model(self, speaker_id):
        start_time = datetime.now()
        self.create_ubm(speaker_id=speaker_id)
        training_features = dam.get_data_for_training('gmm', [speaker_id])
        logging.info(
            f"Training gmm_model with {SYSTEM.FEATURE_TYPE} features for: {speaker_id} :: There are: "
            f"{len(training_features)} training files. Start at: {start_time}")

        param_grid = [{
            'n_components': gmm_config.N_COMPONENTS,
            'max_iter': gmm_config.MAX_ITER,
            'covariance_type': gmm_config.COVARIANCE_TYPE,
            'n_init': gmm_config.N_INIT,
        }]

        gmm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(GaussianMixture(),
                         param_grid=param_grid,
                         cv=MODELCONFIG.CV,
                         refit=MODELCONFIG.REFIT,
                         n_jobs=MODELCONFIG.N_JOBS,
                         verbose=MODELCONFIG.VERBOSE
                         )
        )
        gmm_model.fit(training_features)
        m.save_model(speaker_id, 'gmm_ubm_single_' + SYSTEM.FEATURE_TYPE, gmm_model)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_single_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """
    def get_test_files_and_extra_data(self, speaker_ids):
        test_files = util.load_test_files(speaker_ids)
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        return test_files, extra_data_object

    def predict_n_speakers(self, speaker_ids, mfcc):
        test_files, extra_data_object = self.get_test_files_and_extra_data(speaker_ids=speaker_ids)

        split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, SYSTEM.PROCESSES)
        logging.info(f"starting mult process:{len(split_speaker_ids)}")
        pool = multiprocessing.Pool(processes=SYSTEM.PROCESSES)

        data = []
        for x in range(SYSTEM.PROCESSES):
            data.append((split_speaker_ids[x], test_files, mfcc))

        results = pool.starmap(self.predict_mult, data)
        pool.close()
        pool.join()

        overall_results = []
        for result in results:
            overall_results += result

        rm.create_result_json(overall_results, 'gmm-' + SYSTEM.FEATURE_TYPE, extra_data_object)


    def predict_mult(self, speaker_ids, test_files, mfcc):
        FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, IDS.SPEAKER_IDS, test_files)])
        return part_results

    def predict_speaker(self, speaker_id, speaker_ids):
        test_files, _ = self.get_test_files_and_extra_data(speaker_ids=speaker_ids)

        gmm_models = [m.load_model(speaker_id, 'gmm_ubm_single_' + SYSTEM.FEATURE_TYPE) for speaker_id in speaker_ids]
        ubm_models = [m.load_model(speaker_id, 'gmm_ubm_universal_background_model_' + SYSTEM.FEATURE_TYPE) for speaker_id in speaker_ids]
        ids_of_models = [id for id in speaker_ids]

        for file in test_files:
            print(file)
            x = am.get_features_for_prediction(file, feature_type=SYSTEM.FEATURE_TYPE)
            log_likelihood = np.zeros(len(ids_of_models))
            for i in range(len(ids_of_models)):
                gmm = gmm_models[i]
                ubm = ubm_models[i]
                scores = np.array(gmm.score(x) - ubm.score(x))

                log_likelihood[i] = scores.sum()
            winner = np.argmax(log_likelihood)
            winner_id = ids_of_models[winner]
            print(winner_id, '\n')