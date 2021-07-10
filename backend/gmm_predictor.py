import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from backend import gmm_model as m

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm
from utils import resultManager as rm
from utils import util


class Predictor(object):

    def __init__(self):
        pass

    def predict_gmm(self, models, file):
        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file)])
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm_model = models[i]
            scores = np.array(gmm_model.score(x))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        return winner

    def predict_speaker_gmm(self, speaker_id, speaker_ids, test_files):
        types = ['gmm']

        speaker_object_result = {}
        for t in types:

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
                winner = self.predict_gmm(models, file)
                id_of_file = dm.get_id_of_path(file)

                if id_of_file == speaker_id:
                    if ids_of_models[winner] == speaker_id:
                        true_positive.append(file)
                        accepted_ids.append(id_of_file)
                    else:
                        false_negative.append(file)
                        missed_ids.append(id_of_file)
                else:
                    if not ids_of_models[winner] == 0:
                        false_positiv.append(file)
                        imposter_ids.append(id_of_file)
                    else:
                        true_negativ.append(file)
                        denied_ids.append(id_of_file)

            speaker_object_result.update({t:
                                              {"Accepted": {"amount": len(true_positive),
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
                                                          "files": false_negative}
                                               }})

        return {speaker_id: speaker_object_result}

    def predict_multiple_speakers_gmm(self, speaker_ids):
        test_files = util.load_test_files(speaker_ids)

        results = []
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        for speaker_id in speaker_ids:
            print("GMM ::  predicting for:", speaker_id, "files:", len(test_files))
            results.append([self.predict_speaker_gmm(speaker_id, speaker_ids, test_files)])

        rm.create_result_json(results, 'gmm', extra_data_object)
