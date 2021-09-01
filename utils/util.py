import math
import os
import random
import string

import numpy as np

from datetime import datetime

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import directoryManager as dm, modelManager as m

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
    last_index = len(output)
    while (last_index) > num:
        output[last_index - 2] = output[last_index - 2] + output[last_index - 1]
        output.pop()
        last_index = len(output)
    return output


def get_percent_index(length, percent, train):
    if train:
        return int(math.floor(length * percent))
    else:
        return (length - int(math.ceil(length * percent))) * (-1)


def get_duration(start_time):
    after_time = datetime.now()
    duration = after_time - start_time
    hours = duration.total_seconds() // 3600
    minutes = (duration.total_seconds() // 60) - (hours * 60)
    seconds = duration.total_seconds() - (hours * 3600) - (minutes * 60)
    return "--> duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds)


def get_random_name():
    return ''.join(random.choices(string.ascii_letters + string.digits + '-' + '_', k=16))



