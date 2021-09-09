import json
import logging
import math
from multiprocessing import Pool
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from configparser import ConfigParser

from utils import audioManager as am, directoryManager as dm, modelManager as m, util, \
    resultManager as rm, trainingTestingManager as tt


class GMM(object):

    def __init__(self):
        file = rf'{dm.get_project_path()}/config.ini'
        config = ConfigParser()
        config.read(file)

        self.feature_type = config.get('features', 'feature_type')
        self.param_grid = [{
            'n_components': json.loads(config.get('gmm', 'g_n_components')),
            'max_iter': json.loads(config.get('gmm', 'g_max_iter')),
            'covariance_type': json.loads(config.get('gmm', 'g_covariance_type')),
            'n_init': json.loads(config.get('gmm', 'g_n_init'))
        }]

        self.CV = config.getint('modelconfig', 'cv')
        self.REFIT = config.getboolean('modelconfig', 'refit')
        self.N_JOBS = config.getint('modelconfig', 'n_jobs')
        self.VERBOSE = config.getint('modelconfig', 'verbose')

        self.TRAINING_FILES = config.getfloat("training_testing", "training_files")
        self.PROCESSES = config.getint("system", "processes")
        self.THRESHOLD = config.getfloat("gmm", "g_threshold")
        self.FEATURE_THRESHOLD = config.getfloat("gmm", "threshold")
        self.CREATE_SINGLE_RESULT = config.getboolean("result", "create_single_results")

    """
    # Training phase
    """

    def create_model(self, speaker_id):
        start_time = datetime.now()
        logging.info(
            f"Training gmm_model with {self.feature_type} features for: {speaker_id}, with {self.TRAINING_FILES}. Start at: {start_time}")
        training_features, _ = tt.get_data_for_training('gmm', [speaker_id], self.feature_type, training_files=self.TRAINING_FILES)

        logging.info(
            f" ::: There are: {len(training_features)} trainingvectors. It took {util.get_duration(start_time)} to get files.")
        start_time = datetime.now()
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
        logging.info(f"{gmm_model['gridsearchcv'].best_params_}")

        t = 'gmm_' + self.feature_type
        sub_path = str(int(100 * self.TRAINING_FILES))
        m.save_model(speaker_id, t, gmm_model, sub_path=sub_path)
        # p.draw_plt(files=training_features, model_path=t, name=speaker_id, type=t)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids,  training_files=None):
        if training_files:
            self.TRAINING_FILES = training_files
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction phase
    """

    def predict_n_speakers(self, speaker_ids, test_files, extra_data_object, extra_info=None):
        """
        Used to predict for n speakers. Test files are loaded from speaker folders (/dataset/wav/{id}/) last 10 files are taken
        :param speaker_ids:
        :param test_files
        :param extra_data_object
        :return: outputs overall results into one big result.json, containing overview over all models and their performance.
        """
        if extra_info:
            extra_info = str(int(extra_info * 100))

        # model loading and feature extraction done outside of potential threads to minimise file access, leading to speed improvement.
        models = [m.load_model(speaker_id, "gmm_" + self.feature_type, sub_path=extra_info) for speaker_id in speaker_ids]

        # if defined in config.ini (SYSTEM, PROCESSES) > 1 multiple processes are started
        if self.PROCESSES > 1:
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
            split_models = util.split_array_for_multiprocess(models, self.PROCESSES)
            data = []
            for i in range(self.PROCESSES):
                data.append((split_speaker_ids[i], split_models[i], test_files))

            logging.info(f"starting multi process:{len(split_speaker_ids)}")
            pool = Pool(processes=self.PROCESSES)
            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        else:
            logging.info(f"Starting single thread with: {len(speaker_ids)} ids")
            results = [self.predict_mult(speaker_ids, models, test_files)]

        overall_results = []
        for result in results:
            overall_results += result

        rm.create_overall_result_json(overall_results, 'gmm-' + self.feature_type, extra_data_object, extra_name=extra_info)


    def predict_mult(self, speaker_ids, models, test_files):
        part_results = []
        for i in range(len(speaker_ids)):
            part_results.append([self.predict_speaker(speaker_ids[i], models[i], test_files)])
        return part_results


    def predict_file(self, model, file_path):
        features = am.get_features_for_prediction(file_path, self.feature_type)
        feature_length = len(features[0])
        amount_of_features = len(features)
        scores = []
        for feature in features:

            score_gmm = model.score_samples(feature)

            length = len(score_gmm)
            for x in range(length):
                scores.append(score_gmm[x])
        score = (sum(scores) / feature_length / amount_of_features) + 52

        if score >= self.FEATURE_THRESHOLD:
            return 1
        else:
            return 0


    def predict_speaker(self, speaker_id, model, test_files):
        """
        used to predict all given test_features, from test_files for one specific defined id
        :param speaker_id:
        :param model:
        :param test_files:
        :return: Speakerobject in dict object {id: speaker_object}
        """
        start_time = datetime.now()
        speaker_object_result = {}

        score_of_files = []
        for file in test_files:
            score_of_files.append(self.predict_file(model, file))

        speaker_object_result.update(
            rm.sort_results_and_create_speaker_object(speaker_id, test_files, score_of_files))

        logging.info(f"{util.get_duration(start_time)}")

        if self.CREATE_SINGLE_RESULT:
            rm.create_single_result_json(speaker_id, 'gmm-' + self.feature_type, [[{speaker_id: speaker_object_result}]])

        return {speaker_id: speaker_object_result}
