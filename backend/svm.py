import logging
import multiprocessing
import pickle
from datetime import datetime
from multiprocessing import Process

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import audioManager as am, dataframeManager as dam, directoryManager as dm, modelManager as m, util, \
    resultManager as rm
# from utils import debug
from utils.config import MODELCONFIG, SVM as svm_config, SYSTEM, FEATURES


class SVM(object):
    score = 0

    def __init__(self):
        # logging.basicConfig(level=logging.DEBUG)
        pass

    """
    # Training part
    """

    def create_model(self, speaker_id):
        training_features, is_speaker = dam.get_data_for_training('svm', speaker_id)
        start_time = datetime.now()
        logging.debug("Training svm_model with:", SYSTEM.FEATURE_TYPE, " features for:", speaker_id, ":: There are:",
                      len(training_features),
                      "trainingfiles. Start at: ", start_time)

        param_grid = [{
            'kernel': svm_config.KERNELS,
            'C': svm_config.C,
            'gamma': svm_config.GAMMA
        }]

        svm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(SVC(),
                         param_grid=param_grid,
                         cv=MODELCONFIG.CV,
                         refit=MODELCONFIG.REFIT,
                         n_jobs=MODELCONFIG.N_JOBS,
                         verbose=MODELCONFIG.VERBOSE)
        )

        svm_model.fit(training_features, is_speaker)


        logging.debug(svm_model['gridsearchcv'].best_params_)
        m.save_model(speaker_id, 'svm_' + SYSTEM.FEATURE_TYPE, svm_model)

        logging.debug(util.get_duration(start_time))

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction part
    """

    def predict_n_speakers(self, speaker_ids, mfcc):
        test_files = util.load_test_files(speaker_ids)
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])

        split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, SYSTEM.PROCESSES)
        logging.debug("starting mult process:", len(split_speaker_ids))
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
        rm.create_result_json(overall_results, 'svm-' + SYSTEM.FEATURE_TYPE, extra_data_object)

    def predict_mult(self, speaker_ids, test_files, mfcc):
        FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, test_files)])
        return part_results

    def predict_file(self, speaker_id, t, file_path, feature_type):
        x = am.get_features_for_prediction(file_path, feature_type)
        svm_model = m.load_model(speaker_id, t)
        self.score = svm_model.predict(x)

    def predict_speaker(self, speaker_id, test_files):
        types = ['svm']
        speaker_object_result = {}
        for t in types:
            t += '_' + SYSTEM.FEATURE_TYPE

            true_positive = []
            false_negative = []
            false_positive = []
            true_negative = []

            for file in test_files:
                self.predict_file(speaker_id, t, file, SYSTEM.FEATURE_TYPE)
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
