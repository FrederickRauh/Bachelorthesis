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
            'n_components': [2048],
            'max_iter': [200],
            'covariance_type': ['diag'],
            'n_init': [3]
        }]
        self.gmm_param_grid = [{
            'n_components': [16],
            'max_iter': [200],
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
    # UBM - Universal Background model
    """

    def create_ubm(self, speaker_ids):
        start_time = datetime.now()

        all_training_features, _ = tt.get_data_for_training('gmm-ubm-ubm', speaker_ids=speaker_ids,
                                                            feature_type=self.feature_type)

        logging.info(f"Training gmm_model with {self.feature_type} features for: 'UBM' :: There are: "
                     f"{len(all_training_features)} training files. Start at: {start_time}")

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

        t = 'gmm_ubm_universal_background_model_' + self.feature_type
        m.save_model('', t, ubm_model)
        p.draw_plt(files=all_training_features, model_path=t,
                      name='', type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def create_speaker_model(self, speaker_id, values):
        start_time = datetime.now()
        training_features, _ = tt.get_data_for_training('gmm-ubm-gmm', [speaker_id], self.feature_type)

        logging.info(
            f"Training gmm_model with {self.feature_type} features for: {speaker_id} :: There are: "
            f"{len(training_features)} training files. Start at: {start_time}")

        gmm = BayesianGaussianMixture()

        means = values['means']
        transformed_mean = means[0]

        # for i in range(len(means[0])):
        #     mean = []
        #     for j in range(len(means)):
        #         mean.append(means[j][i])
        #     transformed_mean.append(mean)
        # transformed_mean = np.asarray(transformed_mean)

        gmm.covariances_prior = values['covariances']
        gmm.mean_prior = transformed_mean
        gmm.weights_concentration_prior = values['weights']
        # gmm.converged_ = values['converged']
        # gmm.tol = values['threshold']

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

        t = 'gmm_ubm_single_model_' + self.feature_type
        m.save_model(speaker_id, t, gmm_model)
        p.draw_plt(files=training_features, model_path=t, name=speaker_id, type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):

        # self.create_ubm(speaker_ids=speaker_ids)

        ubm_model = m.load_model('', 'gmm_ubm_universal_background_model_' + self.feature_type)

        all_training_features, _ = tt.get_data_for_training('gmm-ubm-ubm', speaker_ids=speaker_ids,
                                                            feature_type=self.feature_type)


        t = 'gmm_ubm_universal_background_model_' + self.feature_type
        p.draw_plt(files=all_training_features, model_path=t,
                   name='', type=t)

        print("Drawn")

        adaptive_values = {
            'covariances': ubm_model['gridsearchcv'].best_estimator_.covariances_,
            'means': ubm_model['gridsearchcv'].best_estimator_.means_,
            'weights': ubm_model['gridsearchcv'].best_estimator_.weights_,
            'converged': ubm_model['gridsearchcv'].best_estimator_.converged_,
            'threshold': ubm_model['gridsearchcv'].best_estimator_.tol
        }

        jm.write_features_to_json_file(rf"E:/voxceleb/vox1_server/data.json", "UBM", adaptive_values)

        for speaker_id in speaker_ids:
            self.create_speaker_model(speaker_id=speaker_id, values=adaptive_values)

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
            results = self.predict_mult(speaker_ids, test_files)
        overall_results = []
        for result in results:
            overall_results += result

        rm.create_result_json(overall_results, 'gmm-' + self.feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files):
        part_results = []
        for speaker_id in speaker_ids:
            part_results.append([self.predict_speaker(speaker_id, dm.get_all_ids(), test_files)])
        return part_results

    def predict_speaker(self, speaker_id, speaker_ids):
        test_files, _ = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
        test_files = test_files[:10]
        for id in speaker_ids:
            ubm_model = m.load_model('', 'gmm_ubm_universal_background_model_' + self.feature_type)
            gmm_model = m.load_model(id, 'gmm_ubm_single_model_' + self.feature_type)
            print('________________________________________________')
            print('____________________' + speaker_id + '___________________')
            print('________________________________________________')
            # five = []
            # four = []
            # three = []
            # two = []
            # one = []
            # zero = []
            # minus = []

            for file in test_files:
                x = am.get_features_for_prediction(file, self.feature_type)

                ubm_scores = ubm_model.predict(x)
                gmm_scores = gmm_model.predict(x)
                # ubm_score = sum(ubm_scores[0])
                # gmm_score = sum(gmm_scores[0])

                gmm_score = gmm_model.score(x, gmm_scores)
                ubm_score = ubm_model.score(x, ubm_scores)
                total_score = 0
                # if gmm_score > 0:
                #     gmm_score = math.log10(gmm_score)
                # if gmm_score < 0:
                #     gmm_score = -1 * (math.log10(-1 * gmm_score))
                # if ubm_score > 0:
                #     ubm_score = math.log10(ubm_score)
                # if ubm_score < 0:
                #     ubm_score = -1 * (math.log10(-1 * ubm_score))

                # total_score = gmm_score - ubm_score
                total_score = 0
                # llr_score = math.log10(total_score)
                print("UBM:", ubm_score, "GMM:", gmm_score, "SCORE:", total_score, total_score < -213000)

            #     if llr_score > 5:
            #         five.append(file)
            #     else:
            #         if llr_score > 4:
            #             four.append(file)
            #         else:
            #             if llr_score > 3:
            #                 three.append(file)
            #             else:
            #                 if llr_score > 2:
            #                     two.append(file)
            #                 else:
            #                     if llr_score > 1:
            #                         one.append(file)
            #                     else:
            #                         if llr_score > 0:
            #                             zero.append(file)
            #                         else:
            #                             minus.append(file)
            # print("--------------------------------", id)
            # print(len(five))
            # print(len(four))
            # print(len(three))
            # print(len(two))
            # print(len(one))
            # print(len(zero))
            # print(len(minus))
