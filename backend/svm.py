import json
import logging
import multiprocessing
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from configparser import ConfigParser

from tqdm import tqdm

from utils import audioManager as am, modelManager as m, util, \
    resultManager as rm, trainingTestingManager as tt


class SVM(object):

    def __init__(self):
        file = 'config.ini'
        config = ConfigParser()
        config.read(file)

        self.feature_type = config.get('features', 'feature_type')

        self.param_grid = [{
            'kernel': json.loads(config.get('svm', 'kernels')),
            'C': json.loads(config.get('svm', 'c')),
            'gamma': json.loads(config.get('svm', 'gamma')),
            'class_weight': json.loads(config.get('svm', 'class_weight'))
        }]

        self.CV = config.getint('modelconfig', 'cv')
        self.REFIT = config.getboolean('modelconfig', 'refit')
        self.N_JOBS = config.getint('modelconfig', 'n_jobs')
        self.VERBOSE = config.getint('modelconfig', 'verbose')

        self.TRAINING_FILES = config.getfloat("training_testing", "training_files")
        self.PROCESSES = config.getint("system", "processes")
        self.FEATURE_THRESHOLD = config.getfloat("svm", "svm_threshold")
        self.CREATE_SINGLE_RESULT = config.getboolean("result", "create_single_results")

    """
    # Training part
    """

    def create_model(self, speaker_id):
        start_time = datetime.now()
        logging.info(f"Training svm_model with {self.feature_type} features for: {speaker_id}, "
                     f"with {self.TRAINING_FILES}. Start at: {start_time}")

        training_features, is_speaker = tt.get_data_for_training('svm', [speaker_id], self.feature_type,
                                                                 training_files=self.TRAINING_FILES)

        logging.info(f" ::: There are: {len(training_features)} trainingvectors. It took {util.get_duration(start_time)} "
                     f"to get files.")

        start_time = datetime.now()
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
        sub_path = str(int(100 * self.TRAINING_FILES))
        m.save_model(speaker_id, t, svm_model, sub_path=sub_path)
        logging.info(f"{util.get_duration(start_time)}")

    def train(self, speaker_ids, training_files=None):
        if training_files:
            self.TRAINING_FILES = training_files
        for speaker_id in speaker_ids:
            self.create_model(speaker_id=speaker_id)

    """
    # Prediction part
    """

    def predict_n_speakers(self, speaker_ids, test_files, extra_data_object, extra_info=None):
        """

        :param speaker_ids:
        :param test_files:
        :param extra_data_object:
        :param extra_info:
        :return:
        """
        if extra_info:
            extra_info = str(int(extra_info * 100))

        models = [m.load_model(speaker_id, "svm_" + self.feature_type, sub_path=extra_info) for speaker_id in speaker_ids]
        if self.PROCESSES > 1:
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, self.PROCESSES)
            split_models = util.split_array_for_multiprocess(models, self.PROCESSES)
            data = []
            for i in range(self.PROCESSES):
                data.append((split_speaker_ids[i], split_models[i], test_files))

            pool = multiprocessing.Pool(processes=(self.PROCESSES + 1))
            logging.info(f"starting multi process: {self.PROCESSES}")
            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        else:
            logging.info(f"Starting single thread with: {len(speaker_ids)} ids")
            results = [self.predict_mult(speaker_ids=speaker_ids, models=models, test_files=test_files)]

        overall_results = []
        for result in results:
            overall_results += result
        rm.create_overall_result_json(overall_results, 'svm-' + self.feature_type, extra_data_object, extra_name=extra_info)

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
            scores.append(sum(model.predict(feature)))

        overall_score = sum(scores) / feature_length / amount_of_features

        if overall_score > self.FEATURE_THRESHOLD:
            return 1
        else:
            return 0

    def predict_speaker(self, speaker_id, model, test_files):
        start_time = datetime.now()
        speaker_object_result = {}

        logging.info(f"starting prediction for speaker: {start_time}")
        score_of_files = []

        if self.PROCESSES <= 1:
            for file in tqdm(test_files):
                score_of_files.append(self.predict_file(model, file))
        else:
            for file in test_files:
                score_of_files.append(self.predict_file(model, file))

        logging.info(f"all scores collected: {util.get_duration(start_time)}")
        speaker_object_result.update(
            rm.sort_results_and_create_speaker_object(speaker_id, test_files, score_of_files))

        logging.info(f"{util.get_duration(start_time)}")

        if self.CREATE_SINGLE_RESULT:
            rm.create_single_result_json(speaker_id, 'svm-' + self.feature_type,
                                         [[{speaker_id: speaker_object_result}]])

        return {speaker_id: speaker_object_result}
