import os
import csv

from utils import directoryManager as dm


def create_overall_csv():
    ids = dm.get_all_data_names()
    ids.remove('pairs.csv')

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
                        rows.append([identical, wav_files_x[x_wav], wav_files_y[y_wav]])
    write_csv(rows)


def write_csv(rows):
    csv_path = dm.get_all_data_path() + '\\' + "pairs.csv"

    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["same", "file_1", "file_2"])
        writer.writerows(rows)

            # print('x:' + name_x + ', y:' + name_y + ', identical:' + identical)