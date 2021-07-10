import pandas as pd

from backend import svm_model as m

from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm
from utils import resultManager as rm
from utils import util


class Predictor(object):

    def __init__(self):
        pass

    def predict_svm(self, speaker_id, t, file_path):
        # x = m.get_correct_feature_array([file_path])
        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file_path)])
        svm_model = m.load_model(speaker_id, t)
        score = svm_model.predict(x)
        return score

    def predict_speaker_svm(self, speaker_id, test_files):
        # types = ['svm_rbf', 'svm_linear', 'svm_poly', 'svm_custom']
        # types = ['svm_rbf', 'svm_linear', 'svm_poly']
        types = ['svm_custom']
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
            for file in test_files:
                score = self.predict_svm(speaker_id, t, file)
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

    def predict_multiple_speakers_svm(self, speaker_ids):
        test_files = util.load_test_files(speaker_ids)

        results = []
        extra_data = [[test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        for speaker_id in speaker_ids:
            print("SVM ::  predicting for:", speaker_id, "files:", len(test_files))
            results.append([self.predict_speaker_svm(speaker_id, test_files)])

        rm.create_result_json(results, 'svm', extra_data_object)
