import concurrent.futures
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from datetime import datetime

from backend.gmm import gmm_model as m

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib

from utils import audioManager as am, directoryManager as dm, resultManager as rm, util, debug
from utils.config import IDS, FEATURES


class Predictor(object):
    results = []

    def __init__(self):
        pass

    def predict_for_all_modesl(self, models, ids, file, feature_type):
        x = am.get_features_for_prediction(file, feature_type)
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

    def predict_speaker_gmm(self, speaker_id, speaker_ids, test_files, feature_type):
        types = ['gmm']

        speaker_object_result = {}
        for t in types:
            t = t + "_" + feature_type

            true_positive = []
            accepted_ids = []
            false_negative = []
            missed_ids = []
            false_positiv = []
            imposter_ids = []
            true_negativ = []
            denied_ids = []

            models = [m.load_model(speaker_id, t) for speaker_id in speaker_ids]
            ids_of_models = [id for id in speaker_ids]

            for file in test_files:
                id_of_file = dm.get_id_of_path(file)
                winner = self.predict_for_all_modesl(models, ids_of_models, file, feature_type)

                # match is the speaker
                if winner == speaker_id:
                    if winner == id_of_file:
                        true_positive.append(file)
                        accepted_ids.append(dm.get_id_of_path(file))
                    # imposter file
                    else:
                        false_positiv.append(file)
                        imposter_ids.append(dm.get_id_of_path(file))
                else:
                    # should have matched but failed.
                    if dm.get_id_of_path(file) == speaker_id:
                        false_negative.append(file)
                        missed_ids.append(dm.get_id_of_path(file))
                    else:
                        # matches to a model, that is not owner of file
                        true_negativ.append(file)
                        denied_ids.append(dm.get_id_of_path(file))

            # speaker_object_result.update({t:
            speaker_object_result.update({"Accepted": {"amount": len(true_positive),
                                                       "ids": accepted_ids,
                                                       "files": true_positive},
                                          "Denied": {"amount": len(true_negativ),
                                                     "ids": denied_ids,
                                                     "files": true_negativ},
                                          "Imposter": {"amount": len(false_positiv),
                                                       "ids": imposter_ids,
                                                       "files": false_positiv},
                                          "Missed": {"amount": len(false_negative),
                                                     "ids": missed_ids,
                                                     "files": false_negative},
                                          "extra": {"total_id_files": len(true_positive) + len(false_negative),
                                                    "total_imposter_files": len(true_negativ) + len(false_positiv),
                                                    "total_files": len(test_files),
                                                    "model_details": m.load_model(speaker_id, t)[
                                                        'gridsearchcv'].best_params_}
                                          })
        return {speaker_id: speaker_object_result}

    def predict_multiple_speakers_gmm(self, speaker_ids, feature_type, mfcc):
        test_files = util.load_test_files(speaker_ids)
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        if True:
            PROCESSES = 16
            split_speaker_ids = util.split_array_for_multiprocess(speaker_ids, PROCESSES)
            debug.log(("starting mult process:", len(split_speaker_ids)))
            pool = multiprocessing.Pool(processes=PROCESSES)
            data = []
            for x in range(PROCESSES):
                data.append((split_speaker_ids[x], test_files, feature_type, mfcc))
            results = pool.starmap(self.predict_mult, data)
            pool.close()
            pool.join()
        overall_results = []
        for result in results:
            overall_results += result

        rm.create_result_json(overall_results, 'gmm-' + feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files, feature_type, mfcc):
        FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
        part_results = []
        for speaker_id in speaker_ids:
            start_time = datetime.now()
            debug.log(
                ("GMM ::  predicting for:", speaker_id, "files:", len(test_files), " feature_type: ", feature_type))
            part_results.append([self.predict_speaker_gmm(speaker_id, IDS.speaker_ids, test_files, feature_type)])
            debug.log((util.get_duration(start_time)))
        return part_results
