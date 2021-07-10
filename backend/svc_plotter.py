import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import model_selection


def create_svc_plot(files, is_speaker):
    kernels = ['rbf', 'linear', 'poly']
    C = np.arange(0.001, 5.001, 1)
    C = np.arange(0.1, 5.1, 0.1)
    gamma = np.arange(0.1, 1.1, 0.1)
    gamma2 = ['auto', 'scale']

    result_dict = {}
    for kernel in kernels:
        for g in gamma:
            key = "kernel=" + kernel, "gamma=" + str(g)
            results = []
            for c in C:
                print(c)
                svm_model = svm.SVC(kernel=kernel, gamma=g, C=c)
                score = model_selection.cross_val_score(svm_model, files, is_speaker, cv=5, scoring='accuracy')
                results.append(score.mean())
            result_dict.update({key: results})
        for g in gamma2:
            key = "kernel=" + kernel, "gamma=" + str(g)
            results = []
            for c in C:
                print(c)
                svm_model = svm.SVC(kernel=kernel, gamma=g, C=c)
                score = model_selection.cross_val_score(svm_model, files, is_speaker, cv=5, scoring='accuracy')
                results.append(score.mean())
            result_dict.update({key: results})

    keys = list(result_dict.keys())

    for key in keys:
        if key[0].__contains__('rbf'):
            plt.figure(0)
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.title('rbf')
            plt.plot(C, result_dict.get(key), label=key)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        if key[0].__contains__('linear'):
            plt.figure(1)
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.title('linear')
            plt.plot(C, result_dict.get(key), label=key)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # axs[1].legend()
        if key[0].__contains__('poly'):
            plt.figure(2)
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.title('poly')
            plt.plot(C, result_dict.get(key), label=key)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # axs[2].legend()
    plt.figure(0)
    plt.show()
    plt.figure(1)
    plt.show()
    plt.figure(2)
    plt.show()


def plot_(files, is_speaker):
    kernels = ['rbf', 'poly']
    C = np.arange(0.019, 5.01, 0.01)
    # gamma = np.arange(0.1, 1.1, 0.1)
    # degree = np.arange(2, 3, 1)
    # param_grid = dict(kernels=kernels, c=C, gamma=gamma)
    param_grid = [{
        'kernel': kernels,
        'C': C,
        # 'gamma': gamma,
        # 'degree': degree
    }]
    cv = 10
    print("tries:", len(kernels) *
          len(C) *
          # len(gamma) *
          # len(degree) *
          cv)
    #  n_jobs = -1 use all available processors
    grid = model_selection.GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(files, is_speaker)
    print("best score:", grid.best_score_, "best params:", grid.best_params_, "best index:", grid.best_index_, "best estimator:", grid.best_estimator_)
    return grid.best_params_

    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #
    # fig.legend(lines, labels, loc='upper center')
    # plt.show()
