import json
import random

from configparser import ConfigParser

import numpy as np
from matplotlib import pyplot as plt

import utils.modelManager
from utils import dataframeManager as dam, directoryManager as dm, modelManager as m, trainingTestingManager


def draw_plt(files, model_path, name, type):
    model = m.load_model(name, model_path)
    random.shuffle(files)
    files = files[:int(len(files) / 20)]
    labels = model.predict(files)

    if name == '':
        name = 'UBM'

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    # ax.set_xlim3d([200, 700])
    # ax.set_ylim3d([-2, 2])
    # ax.set_zlim3d([-0.1, 0.1])

    # print(files[:, 0])
    feature_length = len(files[0])
    file_length = len(files)
    x = files[:, 0:1]
    y = files[:, 1:2]
    z = files[:, 5:6]
    x_plot = []
    y_plot = []
    z_plot = []
    exp = 1
    for i in range(len(files)):
        x_plot.append(x[i] * exp)
        y_plot.append(y[i] * exp)
        z_plot.append(z[i] * exp)

    # x = files[:, :int((feature_length / 2))]
    # y = files[:, int((feature_length / 2)):(int(feature_length) - 1)]
    # z = files[:, ]
    # x_plot = []
    # y_plot = []
    # z_plot = []
    # for i in range(file_length):
    #     x_plot.append(x[i].mean())
    #     y_plot.append(y[i].mean())
    #     z_plot.append(z[i].mean())
    ax.scatter(x_plot, y_plot, c=labels)
    plt.title(name)
    path = dm.get_model_plt_path(name, type)
    print(f"Drawing: {name}, {len(files)}, PATH: {path}")
    plt.savefig(path)
    plt.close(fig)


def draw_features(features, name):
    fig = plt.figure()
    ax = fig.add_subplot()
    for feature in features:
        ax.plot(feature)
    plt.title(name)
    plt.show()


def draw_models(speaker_id, feature_type):
    training_features, _ = trainingTestingManager.get_data_for_training('gmm', [speaker_id], feature_type)
    x = int(len(training_features) / 50)
    draw_plt(training_features, "gmm_" + feature_type, speaker_id, 'psf')
    # training_features[:int(len(training_features) / 50)]


def random_func():
    speaker_ids = json.loads(config.get('system', 'ids'))
    t = 'gmm_' + 'psf'
    for speaker_id in speaker_ids:
        model = utils.modelManager.load_model(speaker_id, t)

        components = model['gridsearchcv'].best_estimator_.get_params()['n_components']
        means = model['gridsearchcv'].best_estimator_.means_.flatten()
        covariance = np.sqrt(model['gridsearchcv'].best_estimator_.covariances_.flatten())
        weights = model['gridsearchcv'].best_estimator_.weights_

        Gaussian_nr = 0

        data, _ = utils.trainingTestingManager.get_data_for_training('gmm', [speaker_id],
                                                                     config.get("features", "feature_type"))
        x = np.linspace(-100, 100, 2000)
        for mu, sd, p in zip(means, covariance, weights):
            print('Gaussian {:}: μ = {:.2}, σ = {:.2}, weight = {:.2}'.format(Gaussian_nr, mu, sd, p))
            Gaussian_nr += 1


if __name__ == '__main__':
    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)
    speaker_ids = json.loads(config.get('system', 'ids'))
    if speaker_ids == []:
        speaker_ids = dm.get_all_ids()
    print(f"ID Amount: {len(speaker_ids)}")
    for id in speaker_ids:
        draw_models(id, config.get('features', 'feature_type'))
