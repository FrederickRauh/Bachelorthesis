import pandas as pd

from backend import svm_model as m

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm
from utils import util


class Predictor(object):

    def __init__(self):
        pass

    def predict_svm(self, speaker_id, file_path):
        # x = m.get_correct_feature_array([file_path])

        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file_path)])
        svm_model = m.load_model(speaker_id, 'svm')
        score = svm_model.predict(x)
        return score

    def predict_speaker(self, speaker_id, model, test_files):
        accepted = []
        accepted_ids = []
        missed = []
        missed_ids = []
        denied = []
        denied_ids = []
        imposter = []
        imposter_ids = []
        if model == 'svm':
            for file in test_files:
                score = self.predict_svm(speaker_id, file)
                id_of_test_path = dm.get_id_of_path(file)

                if speaker_id == id_of_test_path:
                    if score == 1:
                        accepted.append(file)
                        accepted_ids.append(id_of_test_path)
                    else:
                        missed.append(file)
                        missed_ids.append(id_of_test_path)
                else:
                    if score == 1:
                        imposter.append(file)
                        imposter_ids.append(id_of_test_path)
                    else:
                        denied.append(file)
                        denied_ids.append(id_of_test_path)
        return {speaker_id: {"Accepted":
                    {"amount": len(accepted), "ids": accepted_ids, "files": accepted},
                "Miss": {"amount": len(missed), "ids": missed_ids, "files": missed},
                "Denied": {"amount": len(denied), "ids": denied_ids, "files": denied},
                "Imposter": {"amount": len(imposter), "ids": imposter_ids, "files": imposter}
                }}

    def predict_multiple_speakers(self, speaker_ids):
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
            results.append([self.predict_speaker(speaker_id, 'svm', overall_test_files)])

        new_results = []
        for result in results:
            new_results.append(result[0])

        self.create_result_json(new_results, extra_data_object)

    def create_result_json(self, results, extra_data_object):
        extra_data = {"test_files_amount": len(extra_data_object.overall_test_files[0]), "test_files": extra_data_object.overall_test_files[0]}
        result_json = [(results, extra_data)]
        result_file = pd.DataFrame(result_json, columns=['speaker', 'setup'])
        path = dm.get_all_data_path() + '\\' + "result.json"
        dm.check_if_file_exists_then_remove(path)
        result_file.to_json(path)