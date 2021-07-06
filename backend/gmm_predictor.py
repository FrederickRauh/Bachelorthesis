import pandas as pd

from backend import svm_model as m

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm
from utils import util


class Predictor(object):

    def __init__(self):
        pass

    def predict_gmm(self, speaker_id, file_path):
        # x = m.get_correct_feature_array([file_path])

        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file_path)])
        gmm_model = m.load_model(speaker_id, 'gmm')
        score = gmm_model.predict(x)
        return score

    def predict_speaker_gmm(self, speaker_id, model, test_files):
        true_positive = []
        accepted_ids = []
        false_negative = []
        missed_ids = []
        false_positiv = []
        imposter_ids = []
        true_negativ = []
        denied_ids = []

        if model == 'gmm':
            for file in test_files:
                score = self.predict_gmm(speaker_id, file)
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

        return {speaker_id: {"Accepted": {"amount": len(true_positive), "ids": accepted_ids, "files": true_positive},
                             "Miss": {"amount": len(false_negative), "ids": missed_ids, "files": false_negative},
                             "Imposter": {"amount": len(false_positiv), "ids": imposter_ids, "files": false_positiv},
                             "Denied": {"amount": len(true_negativ), "ids": denied_ids, "files": false_negative}
                             }}

    def predict_multiple_speakers_gmm(self, speaker_ids):
        overall_test_files = []
        for speaker_id in speaker_ids:
            dir = dm.get_test_subfolders(speaker_id)
            # dir = dm.get_voxceleb_subfolders(speaker_id)
            for d in dir:
                dir_path = d
                files_path = dm.get_test_path() + '\\' + speaker_id + '\\' + dir_path
                # files_path = dm.get_voxceleb_path() + '\\' + speaker_id + '\\' + dir_path
                test_files = dm.get_wav_files_in_folder(files_path)
                for x in range(len(test_files)):
                    overall_test_files.append(test_files[x])

        results = []
        extra_data = [[overall_test_files]]
        extra_data_object = pd.DataFrame(extra_data, columns=['overall_test_files'])
        for speaker_id in speaker_ids:
            results.append([self.predict_speaker_gmm(speaker_id, 'gmm', overall_test_files)])

        new_results = []
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for result in results:
            true_positive += result[0].get('Accepted').get('amount')
            new_results.append(result[0])

        confusion_mat = {'t_p': true_positive}

        self.create_result_json(new_results, confusion_mat, extra_data_object)

    def create_result_json(self, results, extra_data_object):
        extra_data = {"test_files_amount": len(extra_data_object.overall_test_files[0]),
                      "test_files": extra_data_object.overall_test_files[0]}
        result_json = [(results, extra_data)]
        result_file = pd.DataFrame(result_json, columns=['speaker', 'extra_data'])
        path = dm.get_all_data_path() + '\\' + "result.json"
        dm.check_if_file_exists_then_remove(path)
        result_file.to_json(path)
