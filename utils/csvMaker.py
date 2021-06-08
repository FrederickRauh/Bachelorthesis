import csv
import os

from utils import directoryManager as dm


def create_csv(name):
    csv_path = dm.get_csv_path(name)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "name", "gender", "length", "features"])


def find_csv(name):
    csv_path = dm.get_csv_path(name)
    if not os.path.isfile(csv_path):
        create_csv(name)
    return csv_path


def edit_csv(name, length, fbanks):
    csv_path = find_csv(name)
    fbank_array = []
    for x in range(len(fbanks)):
        fbank_array.append(fbanks[x])
    with open(csv_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([name, length, fbank_array])
