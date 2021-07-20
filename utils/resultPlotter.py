import os
from random import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from utils import dataframeManager as dam, directoryManager as dm


def get_confusion_mats(mfcc_count, dataset, model_type, feature_type):
    confusion_mats = []
    dataset_path = os.getcwd().replace('utils', '') + 'results' + '\\' + dataset + '\\' + model_type
    subfolders = dm.list_sub_folders(dataset_path)
    for subfolder in subfolders:
        file_path = dataset_path + '\\' + subfolder + '\\' + feature_type + '-' + str(mfcc_count) + '.json'
        if os.path.exists(file_path):
            dataframe = dam.load_dataframe_from_path(file_path)
            confusion_mat = dataframe.confusion_mat[0]
            confusion_mats.append(confusion_mat)

    return confusion_mats


def compute_mean(confusion_mats):
    ACCURACY = []
    PRECISION = []
    RECALL = []
    FAR = []
    FRR = []
    ERR = []
    F1_SCORE = []

    for confusion_mat in confusion_mats:
        ACCURACY.append(confusion_mat['accuracy'])
        PRECISION.append(confusion_mat['precision'])
        RECALL.append(confusion_mat['recall'])
        FAR.append(confusion_mat['false_accept_rate'])
        FRR.append(confusion_mat['false_reject_rate'])
        ERR.append(confusion_mat['equal_error_rate'])
        F1_SCORE.append(confusion_mat['f1_score'])

    ACCURACY = np.array(ACCURACY)
    PRECISION = np.array(PRECISION)
    RECALL = np.array(RECALL)
    FAR = np.array(FAR)
    FRR = np.array(FRR)
    ERR = np.array(ERR)
    F1_SCORE = np.array(F1_SCORE)
    return ACCURACY.mean(), PRECISION.mean(), RECALL.mean(), FAR.mean(), FRR.mean(), ERR.mean(), F1_SCORE.mean()


datasets = ['ba']
model_types = ['svm', 'gmm']

for dataset in datasets:
    for model_tpe in model_types:
        overall = []
        mfccs = np.arange(13, 41, 1)
        ACC_plot = []
        PRECISION_plot = []
        RECALL_plot = []
        FAR_plot = []
        FRR_plot = []
        ERR_plot = []
        F1_SCORE_plot = []
        CORRESPONDING_MFCC_plot = []
        for mfcc in mfccs:
            confusion_mats = get_confusion_mats(mfcc, dataset, model_tpe, 'librosa')
            acc, precision, recall, far, frr, err, f1_score = compute_mean(confusion_mats)
            ACC_plot += [acc]
            PRECISION_plot += [precision * 100]
            RECALL_plot += [recall * 100]
            FAR_plot += [far * 100]
            FRR_plot += [frr * 100]
            ERR_plot += [err * 100]
            F1_SCORE_plot += [f1_score * 100]
            CORRESPONDING_MFCC_plot += [mfcc]


        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        ax1.set_title('Accuracy: ' + model_tpe + ' ' + dataset)
        ax1.set_xlabel('MFCC count')
        ax1.set_ylabel('%')
        ax1.plot(CORRESPONDING_MFCC_plot, ACC_plot, color='b', label='accuracy')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.set_title('Precision, Recall, F1_score: ' + model_tpe + ' ' + dataset)
        ax2.set_xlabel('MFCC count')
        ax2.set_ylabel('%')
        ax2.plot(CORRESPONDING_MFCC_plot, PRECISION_plot, color='k', label='precision')
        ax2.plot(CORRESPONDING_MFCC_plot, RECALL_plot, color='g', label='recall')
        ax2.plot(CORRESPONDING_MFCC_plot, F1_SCORE_plot, color='m', label='f1_score')

        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        ax3.set_title('FAR, FRR, ERR: ' + model_tpe + ' ' + dataset)
        ax3.set_xlabel('MFCC count')
        ax3.set_ylabel('%')
        ax3.plot(CORRESPONDING_MFCC_plot, FAR_plot, color='r', label='far')
        ax3.plot(CORRESPONDING_MFCC_plot, FRR_plot, color='g', label='frr')
        ax3.plot(CORRESPONDING_MFCC_plot, ERR_plot, color='c', label='err')


        # ax2 = ax1.twinx()
        #
        fig1.legend(loc="upper left", prop={'size': 7})
        fig2.legend(loc="upper left", prop={'size': 7})
        fig3.legend(loc="upper left", prop={'size': 7})

        # fig1.gca().legend()
        fig1.show()
        fig2.show()
        fig3.show()
