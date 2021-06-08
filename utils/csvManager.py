import sys
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


def edit_csv(speaker_id, file_name, fbanks):

    name = 'frederick'
    gender = 'm'

    fbank_array = []
    for x in range(len(fbanks)):
        fbank_array.append(fbanks[x])

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
            writer.writerow([speaker_id, name, gender, file_name, fbank_array])


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
        wav_files_x = dm.get_all_wav_names(name_x)
        for y in range(len(ids)):
            name_y = ids[y]
            wav_files_y = dm.get_all_wav_names(name_y)
            identical = 0
            if name_x == name_y:
                identical = 1

            for x_wav in range(len(wav_files_x)):
                for y_wav in range(len(wav_files_y)):
                    if not wav_files_x[x_wav] == wav_files_y[y_wav]:
                        rows.append([identical, name_x + folder_struc + wav_files_x[x_wav], name_y+ folder_struc + wav_files_y[y_wav]])
    write_overall_csv(rows)


def write_overall_csv(rows):
    csv_path = dm.get_all_data_path() + '\\' + "pairs.csv"

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["same", "file_1", "file_2"])
        writer.writerows(rows)