import logging
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from utils import audioManager as am, dataframeManager as dam, directoryManager as dm, modelManager as m, util, resultManager as rm
from config import MODELCONFIG, GMM as gmm_config, SYSTEM, FEATURES, IDS


class GMM(object):

    def __init__(self):
        pass

    """
    # Training phase
    """
    def create_model(self, speaker_id):
        training_features = dam.get_data_for_training('gmm', [speaker_id])
        start_time = datetime.now()
        logging.info(f"Training gmm_model with {SYSTEM.FEATURE_TYPE} features for: {speaker_id} :: There are: {len(training_features)} trainingfiles. Start at: {start_time}")

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
        m.save_model(speaker_id, 'gmm_' + SYSTEM.FEATURE_TYPE, gmm_model)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

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

        # results = self.predict_mult(speaker_ids, test_files, mfcc)

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

    def predict_for_all_modesl(self, models, ids, file):
        x = am.get_features_for_prediction(file, feature_type=SYSTEM.FEATURE_TYPE)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm_model = models[i]
            scores = np.array(gmm_model.score(x))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        winner_id = ids[winner]
        return winner_id

    def predict_for_one_model(self, model, file, feature_type):
        x = am.get_features_for_prediction(file, feature_type)
        score = model.predict_proba(x)
        return score.sum()

    def predict_speaker(self, speaker_id, speaker_ids, test_files):
        types = ['gmm']
        speaker_object_result = {}
        for t in types:
            t += "_" + SYSTEM.FEATURE_TYPE

            true_positive = []
            false_negative = []
            false_positive = []
            true_negative = []

            models = [m.load_model(speaker_id, t) for speaker_id in speaker_ids]
            ids_of_models = [id for id in speaker_ids]

            for file in test_files:
                id_of_file = dm.get_id_of_path(file)
                winner = self.predict_for_all_modesl(models, ids_of_models, file)

                if winner == speaker_id:
                    if winner == id_of_file:                    # match is the speaker
                        true_positive.append(file)
                    else:                                       # imposter file
                        false_positive.append(file)
                else:
                    if dm.get_id_of_path(file) == speaker_id:   # should have matched but failed.
                        false_negative.append(file)
                    else:                                       # matches to a model, that is not owner of file
                        true_negative.append(file)

            speaker_object_result.update(
                rm.create_speaker_object(true_positive, true_negative, false_positive, false_negative))
        return {speaker_id: speaker_object_result}
