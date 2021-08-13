import logging
import multiprocessing
from datetime import datetime

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, modelManager as m, plotter as p, util, \
    resultManager as rm, trainingTestingManager as tt


class GMM(object):

    def __init__(self):
        file = rf'{dm.get_project_path()}/config.ini'
        config = ConfigParser()
        config.read(file)

        self.feature_type = config.get('features', 'FEATURE_TYPE')

        self.param_grid = [{
            'n_components': [16],
            'max_iter': [10000],
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
        t = 'gmm_' + self.feature_type
        m.save_model(speaker_id, t, gmm_model)
        p.draw_plt(files=training_features, model_path=t, name=speaker_id, type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """
    def predict_n_speakers(self, speaker_ids):
        test_files, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
        if self.PROCESSES > 1:
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
            logging.info(f"starting mult process:{len(split_speaker_ids)}")
            pool = multiprocessing.Pool(processes=self.PROCESSES)

            data = []
            for x in range(self.PROCESSES):
                data.append((split_speaker_ids[x], test_files))

            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        else:
            results = [self.predict_mult(speaker_ids, test_files)]

        overall_results = []
        for result in results:
            overall_results += result

        rm.create_result_json(overall_results, 'gmm-' + self.feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files):
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, dm.get_all_ids(), test_files)])
        return part_results


    def predict_file(self, speaker_id, t, file):
        x = am.get_features_for_prediction(file, self.feature_type)
        model = m.load_model(speaker_id, t)
        scores = np.array(model.score(x))
        return scores.sum()

    def predict_speaker(self, speaker_id, speaker_ids, test_files):
        speaker_object_result = {}
        t = "gmm_" + self.feature_type

        true_positive = []
        false_negative = []
        false_positive = []
        true_negative = []

        # models = [m.load_model(speaker_id, t) for speaker_id in speaker_ids]
        # ids_of_models = [id for id in speaker_ids]

        for file in test_files:
            id_of_file = dm.get_id_of_path(file)
            score = self.predict_file(speaker_id, t, file)

            print(speaker_id, score)

            # if winner == speaker_id:
            #     if winner == id_of_file:  # match is the speaker
            #         true_positive.append(file)
            #     else:  # imposter file
            #         false_positive.append(file)
            # else:
            #     if dm.get_id_of_path(file) == speaker_id:  # should have matched but failed.
            #         false_negative.append(file)
            #     else:  # matches to a model, that is not owner of file
            #         true_negative.append(file)

        speaker_object_result.update(
            rm.create_speaker_object(true_positive, true_negative, false_positive, false_negative))

        return {speaker_id: speaker_object_result}
