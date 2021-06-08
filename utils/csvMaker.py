import csv
import os

from utils import directoryManager as dm


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


def edit_csv(speaker_id, length, fbanks):
    name = ''
    gender = ''
    file_name = ''
    csv_path = find_csv(speaker_id)
    fbank_array = []
    for x in range(len(fbanks)):
        fbank_array.append(fbanks[x])
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([speaker_id, name, gender, file_name, fbank_array])
