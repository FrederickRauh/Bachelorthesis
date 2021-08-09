import logging
import multiprocessing
from datetime import datetime

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import audioManager as am, dataframeManager as dam, directoryManager as dm, modelManager as m, util, \
    resultManager as rm, trainingTestingManager as tt
from config import CONFIG as config


class SVM(object):
    score = 0

    def __init__(self):
        pass

    """
    # Training part
    """

    def create_model(self, speaker_id, feature_type=config.FEATURE_TYPE):
        training_features, is_speaker = tt.get_data_for_training('svm', [speaker_id], feature_type)
        start_time = datetime.now()
        logging.info(f"Training svm_model with: {feature_type} features for: {speaker_id} :: "
                     f"There are: {len(training_features)} trainingfiles. Start at: {start_time}")

        param_grid = [{
            'kernel': config.KERNELS,
            'C': config.C,
            'gamma': config.GAMMA
        }]

        svm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(SVC(),
                         param_grid=param_grid,
                         cv=config.CV,
                         refit=config.REFIT,
                         n_jobs=config.N_JOBS,
                         verbose=config.VERBOSE)
        )

        svm_model.fit(training_features, is_speaker)

        logging.info(f"{svm_model['gridsearchcv'].best_params_}")
        m.save_model(speaker_id, 'svm_' + feature_type, svm_model)

        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids, feature_type=config.FEATURE_TYPE):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id, feature_type=feature_type)

    """
    # Prediction part
    """

    def get_test_files_and_extra_data(self, speaker_ids):
        test_files = tt.load_test_files(speaker_ids)
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        return test_files, extra_data_object

    def predict_n_speakers(self, speaker_ids, feature_type=config.FEATURE_TYPE):
        test_files, extra_data_object = self.get_test_files_and_extra_data(speaker_ids=speaker_ids)

        split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, config.PROCESSES)
        logging.info(f"starting mult process: {len(split_speaker_ids)}")
        pool = multiprocessing.Pool(processes=config.PROCESSES)

        data = []
        for x in range(config.PROCESSES):
            data.append((split_speaker_ids[x], test_files, feature_type))

        results = pool.starmap(self.predict_mult, data)
        pool.close()
        pool.join()

        overall_results = []
        for result in results:
            overall_results += result
        rm.create_result_json(overall_results, 'svm-' + feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files, feature_type=config.FEATURE_TYPE):
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, test_files, feature_type)])
        return part_results

    def predict_file(self, speaker_id, t, file_path, feature_type=config.FEATURE_TYPE):
        x = am.get_features_for_prediction(file_path, feature_type)
        svm_model = m.load_model(speaker_id, t)
        self.score = svm_model.predict(x)

    def predict_speaker(self, speaker_id, test_files, feature_type=config.FEATURE_TYPE):
        speaker_object_result = {}
        t = 'svm_' + feature_type

        true_positive = []
        false_negative = []
        false_positive = []
        true_negative = []

        for file in test_files:
            self.predict_file(speaker_id, t, file, feature_type)
            id_of_file = dm.get_id_of_path(file)

            if speaker_id == id_of_file:
                if self.score == 1:
                    true_positive.append(file)
                else:
                    false_negative.append(file)
            else:
                if self.score == 1:
                    false_positive.append(file)
                else:
                    true_negative.append(file)

        speaker_object_result.update(
            rm.create_speaker_object(true_positive, true_negative, false_positive, false_negative))

        return {speaker_id: speaker_object_result}
