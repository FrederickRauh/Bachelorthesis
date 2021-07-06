import pandas as pd

from collections import defaultdict

from utils import directoryManager as dm


def create_speaker_object_with_confusion_mat(results):
    speaker_object = {}
    confusion_mat = {}
    sub_keys = ['svm_rbf', 'svm_linear', 'svm_poly']
    for sub_key in sub_keys:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for result in results:
            key = list(result[0].keys()).__getitem__(0)
            tp += int(result[0][key][sub_key]["Accepted"]['amount'])
            tn += int(result[0][key][sub_key]["Denied"]['amount'])
            fp += int(result[0][key][sub_key]["Imposter"]['amount'])
            fn += int(result[0][key][sub_key]["Missed"]['amount'])
            speaker_object.update({key: result[0][key]})
            # speaker_object[key][sub_key] = result[0][key][sub_key]
        accuracy = ((tp + tn) / (tp + tn + fp + fn))
        recall = (tp / (tp + fn))
        precision = 1
        if not tp == 0:
            precision = (tp / (tp + fp))
        f1_score = ((2 * recall * precision) / (recall + precision))
        confusion_mat.update({sub_key:
                                  {'accuracy': accuracy,
                                   'recall': recall,
                                   'precision': precision,
                                   'f1_score': f1_score,
                                   't_p': tp, 't_n': tn, 'f_p': fp, 'f_n': fn
                                   }})

    return speaker_object, confusion_mat


def create_result_json(results, extra_data_object):
    speaker_object, confusion_mat = create_speaker_object_with_confusion_mat(results)

    extra_data = {"test_files_amount": len(extra_data_object.overall_test_files[0]),
                  "test_files": extra_data_object.overall_test_files[0]}
    result_json = [(confusion_mat, [speaker_object], extra_data)]
    result_file = pd.DataFrame(result_json, columns=['confusion_mat', 'speaker_object', 'extra_data'])

    path = dm.get_all_data_path() + '\\' + "result.json"
    dm.check_if_file_exists_then_remove(path)
    result_file.to_json(path)
