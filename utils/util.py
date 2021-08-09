import os

import numpy as np

from datetime import datetime

from utils import directoryManager as dm


# Turn 3Dim Array in 2D
def get_correct_array_form(array):
    x = np.array(array)
    nsamples, nx, ny = x.shape
    return x.reshape((nsamples, nx * ny))


# Response from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split_array_for_multiprocess(array, num):
    avg = len(array) / float(num)
    output = []
    last = 0.0
    while last < len(array):
        output.append(array[int(last):int(last + avg)])
        last += avg
    return output


def adjust_file_amount_for_voxceleb(dir, speaker_id):
    parent_path = dm.get_voxceleb_path() + '\\' + speaker_id
    path = parent_path + '\\' + dir[0]
    i = 0
    for base, dirs2, Files in os.walk(path):
        for file in Files:
            if file.endswith(".wav"):
                i += 1
    if i < 10:
        next_dir = dm.get_voxceleb_subfolders(speaker_id)[len(dm.get_voxceleb_subfolders(speaker_id)) - 2]
        if next_dir.__contains__('model'):
            next_dir = dm.get_voxceleb_subfolders(speaker_id)[len(dm.get_voxceleb_subfolders(speaker_id)) - 3]
        dir.append(next_dir)
    return dir


def get_duration(start_time):
    after_time = datetime.now()
    duration = after_time - start_time
    hours = duration.total_seconds() // 3600
    minutes = (duration.total_seconds() // 60) - (hours * 60)
    seconds = duration.total_seconds() - (hours * 3600) - (minutes * 60)
    return "--> duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds)
