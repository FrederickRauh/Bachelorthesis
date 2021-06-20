import csv
import os

import pandas as pd

from utils import directoryManager as dm


# def set_max_size():
#     maxInt = sys.maxsize
#     while True:
#         try:
#             csv.field_size_limit(maxInt)
#             break
#         except OverflowError:
#             maxInt = int(maxInt/10)

def write_overall_csv(rows):
    csv_path = dm.get_all_data_path() + '\\' + "pairs.csv"

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["same", "file_1", "file_2"])
        writer.writerows(rows)


def create_csv(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "name", "gender", "file_name", "features"])


def find_csv(speaker_id):
    csv_path = dm.get_csv_path(speaker_id)
    if not os.path.isfile(csv_path):
        create_csv(speaker_id)
    return csv_path


def create_feature_csv(csv_path, rows):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "features"])
        writer.writerows(rows)


def find_feature_csv(csv_path):
    if not os.path.isfile(csv_path):
        create_feature_csv(csv_path)
    return csv_path


def edit_csv(speaker_id, file_name, features):
    name = 'frederick'
    gender = 'm'

    feature_array = get_feature_array_for_csv(features)
    csv_path = find_csv(speaker_id)
    csv_file = pd.read_csv(csv_path)
    wav_files = csv_file["file_name"]

    can_add = True
    for x in wav_files:
        if x == file_name:
            can_add = False
    if can_add:
        with open(csv_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([speaker_id, name, gender, file_name, feature_array])


def write_features_to_file(file_path, features):
    csv_path = file_path.replace('.wav', '.csv')
    with open(csv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(features)


def write_features_to_librosa_csv_file(file_path, features):
    csv_path = file_path.replace('.wav', '.csv')
    with open(csv_path, 'w') as file:
        writer = csv.writer(file)
        for feature in features:
            writer.writerow(feature)


def get_feature_array_for_csv(features):
    feature_array = []
    for x in range(len(features)):
        feature_array.append(features[x])
    return feature_array


def read_csv(csv_file_path):
    rows = []
    with open(csv_file_path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            rows.append(row)
    return rows


def create_overall_csv():
    ids = dm.get_all_data_names()
    if ids.__contains__('pairs.csv'):
        ids.remove('pairs.csv')

    folder_struc = '\\' + "wav" + '\\'

    rows = []

    for x in range(len(ids)):
        name_x = ids[x]
        wav_files_x = dm.get_wav_files(name_x)
        for y in range(len(ids)):
            name_y = ids[y]
            wav_files_y = dm.get_wav_files(name_y)
            identical = 0
            if name_x == name_y:
                identical = 1

            for x_wav in range(len(wav_files_x)):
                for y_wav in range(len(wav_files_y)):
                    if not wav_files_x[x_wav] == wav_files_y[y_wav]:
                        rows.append([identical, name_x + '\\' + wav_files_x[x_wav], name_y + '\\' + wav_files_y[y_wav]])
    write_overall_csv(rows)