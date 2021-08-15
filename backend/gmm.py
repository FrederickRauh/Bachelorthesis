import json
import logging
from multiprocessing import Pool
from multiprocessing import log_to_stderr, get_logger
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

        log_to_stderr()
        logger = get_logger()
        logger.disabled = not config.getboolean('system', 'LOG')

        self.feature_type = config.get('features', 'FEATURE_TYPE')
        self.param_grid = [{
            'n_components': json.loads(config.get('gmm', 'G_N_COMPONENTS')),
            'max_iter': json.loads(config.get('gmm', 'G_MAX_ITER')),
            'covariance_type': json.loads(config.get('gmm', 'G_COVARIANCE_TYPE')),
            'n_init': json.loads(config.get('gmm', 'G_N_INIT'))
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
        start_time = datetime.now()
        logging.info(
            f"Training gmm_model with {self.feature_type} features for: {speaker_id}. Start at: {start_time}")
        training_features, _ = tt.get_data_for_training('gmm', [speaker_id], self.feature_type)
        logging.info(f" ::: There are: {len(training_features)} trainingfiles. It took {util.get_duration(start_time)} to get files.")

        gmm_model = make_pipeline(
            StandardScaler(),
            GridSearchCV(GaussianMixture(),
                         param_grid=self.param_grid,
                         cv=self.CV,
                         refit=self.REFIT,
                         n_jobs=self.N_JOBS,
                         verbose=self.VERBOSE
                         )
        ).fit(training_features)
        t = 'gmm_' + self.feature_type
        m.save_model(speaker_id, t, gmm_model)
        # p.draw_plt(files=training_features, model_path=t, name=speaker_id, type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids):
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """
    def predict_n_speakers(self, speaker_ids):
        """
        Used to predict for n speakers. Test files are loaded from speaker folders (/dataset/wav/{id}/) last 10 files are taken
        :param speaker_ids:
        :return: outputs overall results into one big result.json, containing overview over all models and their performance.
        """
        test_files, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=dm.get_all_ids())

        # model loading and feature extraction done outside of potential threads to minimise file access, leading to speed improvement.
        models = [m.load_model(speaker_id, "gmm_" + self.feature_type) for speaker_id in speaker_ids]
        test_features = []
        for test_file in test_files:
            vector = am.get_features_for_prediction(test_file, self.feature_type)
            test_features.append(vector)

        # if defined in config.ini (SYSTEM, PROCESSES) > 1 multiple processes are started
        if self.PROCESSES > 1:
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
            logging.info(f"starting mult process:{len(split_speaker_ids)}")


            data = []
            for x in range(self.PROCESSES):
                data.append((split_speaker_ids[x], models, test_features, test_files))

            pool = Pool(processes=self.PROCESSES)

            for speaker_ids in split_speaker_ids:
                logging.info(f"Starting thread pool with: {len(speaker_ids)} ids")

            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        else:
            logging.info(f"Starting single thread with: {len(speaker_ids)} ids")
            results = [self.predict_mult(speaker_ids, models, test_features, test_files)]

        overall_results = []
        for result in results:
            overall_results += result

        rm.create_overall_result_json(overall_results, 'gmm-' + self.feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, models, test_features, test_files):
        part_results = []
        for speaker_id in speaker_ids:
            logging.info(f"collecting for {speaker_id}")
            part_results.append([self.predict_speaker(speaker_id, dm.get_all_ids(), models, test_features, test_files)])
            logging.info(f"results for {speaker_id} collected")
        return part_results


    def predict_file(self, speaker_id, t, file_path):
        model = m.load_model(speaker_id, t)
        features = am.get_features_for_prediction(file_path, self.feature_type)

        # scores = np.array(model.score(features))
        scores = model.predict_proba(features)
        count = 0
        for score in scores:
            for x in range(len(score)):
                if score[x] >= 0.9:
                    count += 1

        count /= 399

        if count > 0.7:
            return 1
        else:
            return 0

        # scores = np.array(model.score(x) - 1)
        # return scores.sum()

    def predict_speaker(self, speaker_id, speaker_ids, models, test_features, test_files):
        """
        used to predict all given test_features, from test_files for one specific defined id
        :param speaker_id:
        :param speaker_ids:
        :param models:
        :param test_features:
        :param test_files:
        :return: Speakerobject in dict object {id: speaker_object}
        """
        start_time = datetime.now()
        speaker_object_result = {}
        t = "gmm_" + self.feature_type

        true_positive = []
        false_negative = []
        false_positive = []
        true_negative = []

        # models = [m.load_model(speaker_id, t) for speaker_id in speaker_ids]
        # ids_of_models = [id for id in speaker_ids]

        # log_likelihood = np.zeros(len(models))
        #
        # for i in range(len(models)):
        #     scores = self.predict_file(models[i], test_features[x])
        #     log_likelihood[i] = scores

        # winner = ids_of_models[np.argmax(log_likelihood)]

        # if winner == speaker_id:
        #     if winner == id_of_file:  # match is the speaker
        #         true_positive.append(test_files[x])
        #     else:  # imposter file
        #         false_positive.append(test_files[x])
        # else:
        #     if dm.get_id_of_path(test_files[x]) == speaker_id:  # should have matched but failed.
        #         false_negative.append(test_files[x])
        #     else:  # matches to a model, that is not owner of file
        #         true_negative.append(test_files[x])

        for file in test_files:
            id_of_file = dm.get_id_of_path(file)
            score = self.predict_file(speaker_id, t, file)

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

        rm.create_single_result_json(speaker_id, 'gmm-' + self.feature_type, [[{speaker_id: speaker_object_result}]])

        return {speaker_id: speaker_object_result}
