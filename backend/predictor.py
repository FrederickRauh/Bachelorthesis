from backend import svm_model as m

from frontend import featureExtractorPSF as fe
from utils import directoryManager as dm


class Predictor(object):

    def __init__(self):
        pass

    def predict_svm(self, speaker_id, file_path):
        # x = m.get_correct_feature_array([file_path])

        x = fe.extract_processed_features_librosa(file_path)
        svm_model = m.load_model(speaker_id, 'svm')
        y = [x]
        score = svm_model.predict(y)
        return score

    def predict_speaker(self, speaker_id, model, test_files):
        accepted = 0
        accepted_files = []
        miss = 0
        denied = 0
        imposter = 0
        successful_imposter = []
        if model == 'svm':
            for file in test_files:
                score = self.predict_svm(speaker_id, file)
                id_of_test_path = dm.get_id_of_path(file)

                if speaker_id == id_of_test_path:
                    if score == 1:
                        accepted_files.append(file)
                        accepted += 1
                    else:
                        miss += 1
                else:
                    if score == 1:
                        successful_imposter.append(id_of_test_path)
                        imposter += 1
                    else:
                        denied += 1
            print("WANTS TO BE:", speaker_id, "Missed:", miss, "; Denied:", denied, "\nHits: ", accepted, accepted_files, "\nImposter:", imposter, successful_imposter)
