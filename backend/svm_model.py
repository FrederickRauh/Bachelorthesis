import pickle

import numpy as np
from datetime import datetime

import sklearn
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

from frontend import featureExtractorPSF as fe

from utils import directoryManager as dm


# def custom_kernel(X, Y):


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def create_model(speaker_id, files, is_speaker):
    start_time = datetime.now()
    print("Training svm_model :: There are:", len(files),
          "trainingfiles. Start at: ", start_time)

    # which kernel should be used and why? (Same for gamma)
    # write method to get best c(0.019 vs 2), kernel, etc.
    # choosen after reading paper: Evaluation of kernel methods for speaker verification and identification(2002)
    # and Kernel combination for SVM speaker verification(2008)
    # svm_model = svm.SVC(kernel='poly', gamma='scale', C=2, degree=3)

    kernels = ['rbf', 'linear', 'poly']
    # C = np.arange(0.019, 5.001, 0.001)
    C = [1, 2, 5]
    gamma = ['scale', 'auto']
    total_score_rbf_auto = []
    total_score_rbf_scale = []

    total_score_linear_auto = []
    total_score_linear_scale = []

    total_score_poly_auto = []
    total_score_poly_scale = []



    for kernel in kernels:
        for g in gamma:
            for c in C:
                svm_model = svm.SVC(kernel=kernel, gamma=g, C=c)
                score = model_selection.cross_val_score(svm_model, files, is_speaker, cv=5, scoring='accuracy')
                print("Kernel:", kernel, "gamma:", g, "C=%f; accuracy: %f; standard deviation of %f" % (c, score.mean(), score.std()))
                if kernel=='rbf':
                    if g == 'auto':
                        total_score_rbf_auto.append(score.mean())
                    if g == 'scale':
                        total_score_rbf_scale.append(score.mean())
                if kernel=='linear':
                    if g == 'auto':
                        total_score_linear_auto.append(score.mean())
                    if g == 'scale':
                        total_score_linear_scale.append(score.mean())
                if kernel=='poly':
                    if g == 'auto':
                        total_score_poly_auto.append(score.mean())
                    if g == 'scale':
                        total_score_poly_scale.append(score.mean())

    x = C
    plt.figure()
    plt.plot(x, total_score_rbf_auto, 'tab:red', x, total_score_rbf_scale, 'tab:purple',
             x, total_score_linear_auto, 'tab:cyan', x, total_score_linear_scale, 'tab:blue',
             x, total_score_poly_auto, 'tab:green', x, total_score_poly_scale, 'tab:orange')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.show()

    svm_model_rbf = svm.SVC(kernel='rbf', gamma='scale', C=2)
    svm_model_rbf.fit(files, is_speaker)
    score = model_selection.cross_val_score(svm_model_rbf, files, is_speaker, cv=5, scoring='accuracy')
    print("RBF: accuracy: %f; standard deviation of %f" % (score.mean(), score.std()))
    save_model(speaker_id, 'svm_rbf', svm_model_rbf)
    #
    # svm_model_linear = svm.SVC(kernel='linear', gamma='scale', C=2)
    # svm_model_linear.fit(files, is_speaker)
    # score = model_selection.cross_val_score(svm_model_linear, files, is_speaker, cv=5, scoring='accuracy')
    # print("LINEAR: accuracy: %f; standard deviation of %f" % (score.mean(), score.std()))
    # save_model(speaker_id, 'svm_linear', svm_model_linear)
    #
    # svm_model_poly = svm.SVC(kernel='poly', gamma='scale', C=2, degree=3)
    # svm_model_poly.fit(files, is_speaker)
    # score = model_selection.cross_val_score(svm_model_poly, files, is_speaker, cv=5, scoring='accuracy')
    # print("POLY: accuracy: %f; standard deviation of %f" % (score.mean(), score.std()))
    # save_model(speaker_id, 'svm_poly', svm_model_poly)

    after_time = datetime.now()
    duration = after_time - start_time
    hours = duration.total_seconds() // 3600
    minutes = duration.total_seconds() // 60
    seconds = duration.total_seconds() - (duration.total_seconds() // 60)
    print("duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds))

    # save_model(speaker_id, 'svm', svm_model)


