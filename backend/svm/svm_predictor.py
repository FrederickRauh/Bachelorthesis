import multiprocessing

import pandas as pd

from datetime import datetime

from backend.svm import svm_model as m

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib

from utils import audioManager as am, directoryManager as dm, resultManager as rm, util, debug
from utils.config import FEATURES, SYSTEM


class Predictor(object):

    def __init__(self):
        pass

    def predict_svm(self, speaker_id, t, file_path, feature_type):
        x = am.get_features_for_prediction(file_path, feature_type)
        svm_model = m.load_model(speaker_id, t)
        score = svm_model.predict(x)
        return score

    def predict_speaker_svm(self, speaker_id, test_files, feature_type):
        types = ['svm_custom']
        speaker_object_result = {}
        for t in types:
            t = t + "_" + feature_type

            true_positive = []
            accepted_ids = []
            false_negative = []
            missed_ids = []
            false_positive = []
            imposter_ids = []
            true_negative = []
            denied_ids = []

            for file in test_files:
                score = self.predict_svm(speaker_id, t, file, feature_type)
                id_of_file = dm.get_id_of_path(file)

                if speaker_id == id_of_file:
                    if score == 1:
                        true_positive.append(file)
                        accepted_ids.append(id_of_file)
                    else:
                        false_negative.append(file)
                        missed_ids.append(id_of_file)
                else:
                    if score == 1:
                        false_positive.append(file)
                        imposter_ids.append(id_of_file)
                    else:
                        true_negative.append(file)
                        denied_ids.append(id_of_file)

            # speaker_object_result.update({t:
            speaker_object_result.update({"Accepted": {"amount": len(true_positive),
                                                       "ids": accepted_ids,
                                                       "files": true_positive},
                                          "Denied": {"amount": len(true_negative),
                                                     "ids": denied_ids,
                                                     "files": true_negative},
                                          "Imposter": {"amount": len(false_positive),
                                                       "ids": imposter_ids,
                                                       "files": false_positive},
                                          "Missed": {"amount": len(false_negative),
                                                     "ids": missed_ids,
                                                     "files": false_negative},
                                          "extra": {"total_id_files": len(true_positive) + len(false_negative),
                                                    "total_imposter_files": len(true_negative) + len(false_positive),
                                                    "total_files": len(test_files),
                                                    "model_details": m.load_model(speaker_id, t)[
                                                        'gridsearchcv'].best_params_}
                                          })
        return {speaker_id: speaker_object_result}

    def predict_n_speakers_svm(self, speaker_ids, feature_type, mfcc):
        test_files = util.load_test_files(speaker_ids)
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        if len(speaker_ids) > 9:
            PROCESSES = SYSTEM.PROCESSES
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
        rm.create_result_json(overall_results, 'svm-' + feature_type, extra_data_object)

    def predict_mult(self, speaker_ids, test_files, feature_type, mfcc):
        FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
        part_results = []
        for speaker_id in speaker_ids:
            start_time = datetime.now()
            debug.log(
                ("SVM ::  predicting for:", speaker_id, "files:", len(test_files), " feature_type: ", feature_type))
            part_results.append([self.predict_speaker_svm(speaker_id, test_files, feature_type)])

            debug.log((util.get_duration(start_time)))
        return part_results
