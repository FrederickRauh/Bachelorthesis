import pandas as pd

from utils import dataframeManager as dam, directoryManager as dm

"""
ResultManager contains the methods to calculate the confusion mat after prediction and 
outputting it in the result folder.
"""


def load_result(file_path):
    result_json = dam.load_datafram_from_path(file_path)
    return result_json.confusion_mat[0]


def create_speaker_object(true_positive, true_negative, false_positive, false_negative):
    accepted_ids = dm.get_ids_of_paths(true_positive)
    denied_ids = dm.get_ids_of_paths(true_negative)
    imposter_ids = dm.get_ids_of_paths(false_positive)
    missed_ids = dm.get_ids_of_paths(false_negative)
    file_amount = len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative)
    return {"Accepted": {"amount": len(true_positive),
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
                      "total_files": file_amount}
            }


def create_speaker_object_with_confusion_mat(results):
    speaker_object = {}
    confusion_mat = {}
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for result in results:
        key = list(result[0].keys()).__getitem__(0)
        tp += int(result[0][key]["Accepted"]['amount'])
        tn += int(result[0][key]["Denied"]['amount'])
        fp += int(result[0][key]["Imposter"]['amount'])
        fn += int(result[0][key]["Missed"]['amount'])
        speaker_object.update({key: result[0][key]})


    false_accept_rate = -1
    if not (fp + tn) == 0: false_accept_rate = fp / (fp + tn)
    false_reject_rate = -1
    if not (fn + tp) == 0: false_reject_rate = fn / (fn + tp)
    equal_error_rate = (false_accept_rate + false_reject_rate) / 2
    accuracy = 100 - equal_error_rate
    recall = -1
    if not (tp + fn) == 0: recall = tp / (tp + fn)
    precision = -1
    if not (tp + fp) == 0: precision = tp / (tp + fp)
    f1_score = -1
    if not (recall + precision) == 0: f1_score = ((2 * recall * precision) / (recall + precision))

    confusion_mat.update({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'false_accept_rate': false_accept_rate,
        'false_reject_rate': false_reject_rate,
        'equal_error_rate': equal_error_rate,
        'f1_score': f1_score,
        't_p': tp, 't_n': tn, 'f_p': fp, 'f_n': fn
    })

    return speaker_object, confusion_mat


def create_result_json(results, t, extra_data_object):
    speaker_object, confusion_mat = create_speaker_object_with_confusion_mat(results)

    extra_data = {"test_files_amount": len(extra_data_object.overall_test_files[0]),
                  "test_files": extra_data_object.overall_test_files[0]}
    result_json = [(confusion_mat, [speaker_object], extra_data)]
    result_file = pd.DataFrame(result_json, columns=['confusion_mat', 'speaker_object', 'extra_data'])
    t = t.split('-')
    directory_path = dm.get_results_folder(t[0])
    version_path = dm.make_dir(directory_path + '\\' + 'version' + str(config.VERSION))
    path = version_path + '\\' + t[1] + '-' + str(config.G_N_COMPONENTS) + '-' + 'all-features' + ".json"
    dm.check_if_file_exists_then_remove(path)
    result_file.to_json(path)
