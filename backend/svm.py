import json
import logging
import multiprocessing
from datetime import datetime

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, modelManager as m, plotter as p, util, \
    resultManager as rm, trainingTestingManager as tt


class SVM(object):

    def __init__(self):
        file = 'config.ini'
        config = ConfigParser()
        config.read(file)

        self.feature_type = config.get('features', 'FEATURE_TYPE')

        C = np.arange(config.getfloat('svm', 'C.lower'), config.getfloat('svm', 'C.upper'), 0.1)
        C = [round(x, 2) for x in C]

        self.param_grid = [{
            'kernel': json.loads(config.get('svm', 'KERNELS')),
            'C': C,
            'gamma': json.loads(config.get('svm', 'GAMMA'))
        }]

        self.CV = config.getint('modelconfig', 'CV')
        self.REFIT = config.getboolean('modelconfig', 'REFIT')
        self.N_JOBS = config.getint('modelconfig', 'N_JOBS')
        self.VERBOSE = config.getint('modelconfig', 'VERBOSE')

        self.PROCESSES = config.getint("system", "PROCESSES")

    """
    # Training part
    """

    def create_model(self, speaker_id, ):
        training_features, is_speaker = tt.get_data_for_training('svm', [speaker_id], self.feature_type)
        start_time = datetime.now()
        logging.info(f"Training svm_model with: {self.feature_type} features for: {speaker_id} :: "
                     f"There are: {len(training_features)} trainingfiles. Start at: {start_time}")

        print(len(training_features))

        svm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(SVC(),
                         param_grid=self.param_grid,
                         cv=self.CV,
                         refit=self.REFIT,
                         n_jobs=self.N_JOBS,
                         verbose=self.VERBOSE)
        ).fit(training_features, is_speaker)

        logging.info(f"{svm_model['gridsearchcv'].best_params_}")
        t = 'svm_' + self.feature_type
        m.save_model(speaker_id, t, svm_model)
        # p.draw_plt(files=training_features, model_path=t, name=speaker_id, type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction part
    """
    def predict_n_speakers(self, speaker_ids):
        test_files, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=dm.get_all_ids())

        if self.PROCESSES > 1:
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
            logging.info(f"starting multi process: {len(split_speaker_ids)}")
            pool = multiprocessing.Pool(processes=self.PROCESSES)

            data = []
            for x in range(self.PROCESSES):
                data.append((split_speaker_ids[x], test_files))

            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        else:
            results = [self.predict_mult(speaker_ids=speaker_ids, test_files=test_files)]

        overall_results = []
        for result in results:
            overall_results += result
        rm.create_overall_result_json(overall_results, 'svm-' + self.feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files):
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, test_files)])
        return part_results

    def predict_file(self, speaker_id, t, file_path):
        svm_model = m.load_model(speaker_id, t)
        features = am.get_features_for_prediction(file_path, self.feature_type)
        # score = svm_model.predict([features.flatten()])
        # return score
        # scores = []
        scores = svm_model.predict(features)
        score_sum = sum(scores)
        overall_score = score_sum / 399
        if overall_score > 0.20:
            return 1
        else:
            return 0

    def predict_speaker(self, speaker_id, test_files):
        speaker_object_result = {}
        t = 'svm_' + self.feature_type

        true_positive = []
        false_negative = []
        false_positive = []
        true_negative = []

        for file in test_files:
            score = self.predict_file(speaker_id, t, file)
            id_of_file = dm.get_id_of_path(file)

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

        rm.create_single_result_json(speaker_id, 'svm-' + self.feature_type, [[{speaker_id: speaker_object_result}]])

        return {speaker_id: speaker_object_result}
