import numpy as np
import pandas as pd

from datetime import datetime

from backend.gmm import gmm_model as m

from frontend import featureExtractorPSF as fpsf

from utils import directoryManager as dm, resultManager as rm, util


class Predictor(object):

    def __init__(self):
        pass

    def predict_for_all_modesl(self, models, ids, file):
        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file)])
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm_model = models[i]
            scores = np.array(gmm_model.score(x))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        winner_id = ids[winner]
        return winner_id

    def predict_for_one_model(self, model, file):
        x = util.get_correct_array_form([fpsf.extract_processed_mfcc_from_file(file)])
        score = model.predict_proba(x)
        return score.sum()

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

            # for file in test_files:
            #     score = self.predict_for_one_model(m.load_model(speaker_id, t), file)
            #     print("predicted for file:", file, "score:", score, )


            for file in test_files:
                id_of_file = dm.get_id_of_path(file)
                winner = self.predict_for_all_modesl(models, ids_of_models, file)

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
            start_time = datetime.now()

            print("GMM ::  predicting for:", speaker_id, "files:", len(test_files))
            results.append([self.predict_speaker_gmm(speaker_id, speaker_ids, test_files)])

            after_time = datetime.now()
            duration = after_time - start_time
            hours = duration.total_seconds() // 3600
            minutes = (duration.total_seconds() // 60) - (hours * 60)
            seconds = duration.total_seconds() - (hours * 3600) - (minutes * 60)
            print("--> duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds),
                  # "----- Model: accuracy: %f; standard deviation of %f" % (score.mean(), score.std())
                  )

        rm.create_result_json(results, 'gmm', extra_data_object)
