import pandas as pd

from backend import svm_model as m

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm
from utils import resultManager as rm
from utils import util


class Predictor(object):

    def __init__(self):
        pass

    def predict_svm(self, speaker_id, type, file_path):
        # x = m.get_correct_feature_array([file_path])
        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file_path)])
        svm_model = m.load_model(speaker_id, type)
        score = svm_model.predict(x)
        return score

    def predict_speaker_svm(self, speaker_id, model, test_files):
        print("predicting for:", speaker_id, len(test_files))
        true_positive = []
        accepted_ids = []
        false_negative = []
        missed_ids = []
        false_positiv = []
        imposter_ids = []
        true_negativ = []
        denied_ids = []
        type = ['svm_rbf', 'svm_linear', 'svm_poly']
        speaker_object_result = {}
        if model.__contains__('svm'):
            for t in type:
                for file in test_files:
                    score = self.predict_svm(speaker_id, t, file)
                    id_of_test_path = dm.get_id_of_path(file)

                    if speaker_id == id_of_test_path:
                        if score == 1:
                            true_positive.append(file)
                            accepted_ids.append(id_of_test_path)
                        else:
                            false_negative.append(file)
                            missed_ids.append(id_of_test_path)
                    else:
                        if score == 1:
                            false_positiv.append(file)
                            imposter_ids.append(id_of_test_path)
                        else:
                            true_negativ.append(file)
                            denied_ids.append(id_of_test_path)
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
        overall_test_files = []
        for speaker_id in speaker_ids:
            # dir = dm.get_test_subfolders(speaker_id)
            dir = [dm.get_voxceleb_subfolders(speaker_id)[len(dm.get_voxceleb_subfolders(speaker_id)) - 1]]
            for d in dir:
                dir_path = d
                # files_path = dm.get_test_path() + '\\' + speaker_id + '\\' + dir_path
                files_path = dm.get_voxceleb_path() + '\\' + speaker_id + '\\' + dir_path
                test_files = dm.get_wav_files_in_folder(files_path)
                for x in range(len(test_files)):
                    overall_test_files.append(test_files[x])

        results = []
        extra_data = [[overall_test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        for speaker_id in speaker_ids:
            results.append([self.predict_speaker_svm(speaker_id, 'svm', overall_test_files)])

        rm.create_result_json(results, extra_data_object)
