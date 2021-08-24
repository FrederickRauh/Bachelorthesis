from configparser import ConfigParser

import pandas as pd

from utils import dataframeManager as dam, directoryManager as dm

"""
ResultManager contains the methods to calculate the confusion mat after prediction and 
outputting it in the result folder.
"""
file = rf'{dm.get_project_path()}/config.ini'
config = ConfigParser()
config.read(file)


def load_result(file_path):
    result_json = dam.load_datafram_from_path(file_path)
    return result_json.confusion_mat[0]


def sort_results_and_create_speaker_object(speaker_id, file_list, score_list):
    true_positive = []
    false_negative = []
    false_positive = []
    true_negative = []
    for x in range(len(file_list)):
        score = score_list[x]
        file = file_list[x]

        id_of_file = dm.get_id_of_path(file)
        if speaker_id == id_of_file:
            if score == 1:
                if not file.__contains__('--attack--'):
                    true_positive.append(file)
                else:
                    false_positive.append(file)
            else:
                if not file.__contains__('--attack--'):
                    false_negative.append(file)
                else:
                    true_negative.append(file)
        else:
            if score == 1:
                false_positive.append(file)
            else:
                true_negative.append(file)

    return create_speaker_object(true_positive, true_negative, false_positive, false_negative)


def create_speaker_object(true_positive, true_negative, false_positive, false_negative):
    accepted_ids = dm.get_ids_of_paths(true_positive)
    denied_ids = dm.get_ids_of_paths(true_negative)
    imposter_ids = dm.get_ids_of_paths(false_positive)
    missed_ids = dm.get_ids_of_paths(false_negative)
    file_amount = len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative)
    speaker_object = {"Accepted": {"amount": len(true_positive),
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
                                 "files": false_negative}}
    if config.getboolean("result", "add_extra"):
        speaker_object = {"Accepted": {"amount": len(true_positive),
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
                                    "total_files": file_amount}}
    return speaker_object


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
    precision = -1
    if not (tp + fp) == 0: precision = tp / (tp + fp)
    recall = -1
    if not (tp + fn) == 0: recall = tp / (tp + fn)

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


def create_overall_result_json(results, t, extra_data_object):
    sorted_results = []
    ids = []
    for result in results:
        ids.append(list(result[0].keys()).__getitem__(0))
    ids.sort()
    for id in ids:
        for result in results:
            if list(result[0].keys()).__getitem__(0) == id:
                sorted_results.append(result)

    speaker_object, confusion_mat = create_speaker_object_with_confusion_mat(sorted_results)

    extra_data = {"test_files_amount": len(extra_data_object.overall_test_files[0]),
                  "test_files": extra_data_object.overall_test_files[0],
                  "feature_version": config.get('features', 'feature_type')
                  }
    result_json = [(confusion_mat, [speaker_object], extra_data)]
    result_file = pd.DataFrame(result_json, columns=['confusion_mat', 'speaker_object', 'extra_data'])
    t = t.split('-')
    directory_path = dm.get_results_folder(t[0])
    system_version = config.get('result', 'version')
    version_path = dm.make_dir(rf'{directory_path}/version-{system_version}')
    path = rf'{version_path}/result.json'
    dm.check_if_file_exists_then_remove(path)
    result_file.to_json(path)


def create_single_result_json(speaker_id, t, results):
    speaker_object, confusion_mat = create_speaker_object_with_confusion_mat(results)
    result_json = [(confusion_mat, [speaker_object])]
    result_file = pd.DataFrame(result_json, columns=['confusion_mat', 'speaker_object'])
    t = t.split('-')
    directory_path = dm.get_results_folder(t[0])
    system_version = config.get('result', 'version')
    version_path = dm.make_dir(rf'{directory_path}/version-{system_version}')
    path = rf'{version_path}/{speaker_id}-result.json'
    dm.check_if_file_exists_then_remove(path)
    result_file.to_json(path)