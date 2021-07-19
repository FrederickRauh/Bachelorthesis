import pandas as pd

from utils import directoryManager as dm


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
    false_accept_rate = 'nan'
    false_reject_rate = 'nan'
    equal_error_rate = 'nan'
    accuracy = 'nan'
    recall = 'nan'
    precision = 'nan'
    f1_score = 'nan'

    if not (fp + tn) == 0:
        false_accept_rate = fp / (fp + tn)

    if not (fn + tp) == 0:
        false_reject_rate = fn / (fn + tp)

    equal_error_rate = (false_accept_rate + false_reject_rate) / 2

    accuracy = 100 - equal_error_rate

    if not (tp + fn) == 0:
        recall = tp / (tp + fn)

    if not (tp + fp) == 0:
        precision = tp / (tp + fp)

    if not (recall + precision) == 0:
        f1_score = ((2 * recall * precision) / (recall + precision))

    confusion_mat.update(
        {
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


def create_speaker_object_with_confusion_mat_with_subkeys(results):
    speaker_object = {}
    confusion_mat = {}
    # sub_keys = ['svm_rbf', 'svm_linear', 'svm_poly', 'svm_custom']
    # # sub_keys = ['svm_rbf', 'svm_linear', 'svm_poly']
    keys = list(results[0][0].keys())
    sub_keys = list(results[0][0][keys[0]].keys())
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
        false_acception_rate = 'nan'
        false_rejection_rate = 'nan'
        equal_error_rate = 'nan'
        accuracy = 'nan'
        recall = 'nan'
        precision = 'nan'
        f1_score = 'nan'

        if not (fp + tn) == 0:
            false_acception_rate = fp / (fp + tn)

        if not (fn + tp) == 0:
            false_rejection_rate = fn / (fn + tp)

        equal_error_rate = (false_acception_rate + false_rejection_rate) / 2

        accuracy = 100 - equal_error_rate

        if not (tp + fn) == 0:
            recall = tp / (tp + fn)

        if not (tp + fp) == 0:
            precision = tp / (tp + fp)

        if not (recall + precision) == 0:
            f1_score = ((2 * recall * precision) / (recall + precision))

        confusion_mat.update({sub_key:
            {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'false_accept_rate': false_acception_rate,
                'false_reject_rate': false_rejection_rate,
                'equal_error_rate': equal_error_rate,
                'f1_score': f1_score,
                't_p': tp, 't_n': tn, 'f_p': fp, 'f_n': fn
            }})

    return speaker_object, confusion_mat


def create_result_json(results, t, extra_data_object):
    speaker_object, confusion_mat = create_speaker_object_with_confusion_mat(results)

    extra_data = {"test_files_amount": len(extra_data_object.overall_test_files[0]),
                  "test_files": extra_data_object.overall_test_files[0]}
    result_json = [(confusion_mat, [speaker_object], extra_data)]
    result_file = pd.DataFrame(result_json, columns=['confusion_mat', 'speaker_object', 'extra_data'])

    path = dm.get_all_data_path() + '\\' + "result-" + t + ".json"
    dm.check_if_file_exists_then_remove(path)
    result_file.to_json(path)
