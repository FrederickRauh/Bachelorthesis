
import csv
import json
import logging
import os

import pandas as pd

from utils import directoryManager as dm


# Overall CSV (pairs.csv)
def create_overall_csv():
    logging.debug("creating pairs.csv")
    ids = dm.get_all_ids()
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


def write_overall_csv(rows):
    csv_path = dm.get_all_data_path() + '\\' + "pairs.csv"
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["same", "file_1", "file_2"])
        writer.writerows(rows)


# CSV for each user(user_path + '\\' + csv)
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


# CSV file that contains the features of a single wav file
def create_feature_csv(csv_path, rows):
    dm.create_feature_csv_dir(csv_path)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "features"])
        writer.writerows(rows)


def find_feature_csv(csv_path):
    if not os.path.isfile(csv_path):
        create_feature_csv(csv_path, [])
    return csv_path


def edit_csv(speaker_id, file_name, features):
    name = speaker_id
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


def write_features_to_librosa_csv_file(csv_path, wav_path, features):
    csv_path = csv_path.replace('.wav', '.csv')
    find_feature_csv(csv_path)
    entry = []
    for feature in features:
        entry.append([wav_path, feature])
    csv_file = pd.DataFrame(entry, columns=['wav_path', 'features'])
    dm.check_if_file_exists_then_remove(csv_path)
    csv_file.to_csv(csv_path)
    # with open(csv_path, 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(features)
    #     # for feature in features:
    #     #     writer.writerow(feature)


def write_features_to_psf_csv_file(csv_path, wav_path, features):
    csv_path = csv_path.replace('.wav', '.csv')
    find_feature_csv(csv_path)
    entry = []
    for feature in features:
        entry.append([wav_path, feature])
    csv_file = pd.DataFrame(entry, columns=['wav_path', 'features'])
    dm.check_if_file_exists_then_remove(csv_path)
    csv_file.to_csv(csv_path)
    # with open(csv_path, 'w') as file:
    #     writer = csv.writer(file)
    #     for feature in features:
    #         writer.writerow(feature)


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


# JSON PART
def create_feature_json(json_path, rows):
    dm.create_feature_json_dir(json_path)
    with open(json_path, 'w', newline='') as file:
        writer = json.writer(file)
        writer.writerow(["file_name", "features"])
        writer.writerows(rows)


def find_feature_json(json_path):
    if not os.path.isfile(json_path):
        create_feature_csv(json_path, [])
    return json_path


def write_features_to_json_file(json_path, wav_path, features):
    json_path = json_path.replace('.wav', '.json')
    find_feature_json(json_path)
    entry = []
    entry.append([wav_path, features, len(features)])
    json_file = pd.DataFrame(entry, columns=['wav_path', 'features', 'feature count'])
    dm.check_if_file_exists_then_remove(json_path)
    json_file.to_json(json_path)


def load_features_from_json_of_wav_file(wav_path, version):
    json_path = dm.get_all_wav_path() + '\\' + dm.get_feature_path(wav_path, version)
    json_path = json_path.replace('.wav', '.json')
    dataframe = pd.read_json(json_path)
    return dataframe.features[0]