import pickle

from datetime import datetime

from sklearn import svm

from utils import directoryManager as dm, util


def save_model(speaker_id, type, model):
    model_path = dm.get_model_path(speaker_id, type)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(speaker_id, type):
    model_path = dm.get_model_path(speaker_id, type)
    return pickle.load(open(model_path, 'rb'))


def get_data_for_training(speaker_id, dataframe):
    t = []
    y = []
    speaker_ids = dm.get_all_ids()
    for id in speaker_ids:
        wav_files = dm.get_wav_files(id)
        for wav_file in wav_files:
            file = id + '\\' + wav_file
            t.append(file)
            is_speaker = 0
            if id == speaker_id:
                is_speaker = 1
            y.append(is_speaker)

    training_files = get_training_files(dataframe, t)

    return training_files, y


def get_training_files(dataframe, files):
    training_files = []
    for element in files:
        training_features = dataframe.loc[dataframe['file_name'] == element].feature.array[0]['0']
        training_files.append(training_features)
    return util.get_correct_array_form(training_files)


def create_model(speaker_id, dataframe):
    training_features, is_speaker = get_data_for_training(speaker_id, dataframe)
    start_time = datetime.now()
    print("Training svm_model for:", speaker_id, " :: There are:", len(training_features),
          "trainingfiles. Start at: ", start_time)
    # which kernel should be used and why? (Same for gamma)
    # write method to get best c(0.019 vs 2), kernel, etc.
    # choosen after reading paper: Evaluation of kernel methods for speaker verification and identification(2002)
    # and Kernel combination for SVM speaker verification(2008)
    # literature suggests to use rbf or poly for mfccs
    #
    # params = svc.plot_(files, is_speaker)
    svm_model_custom = svm.SVC(
        # kernel=params.get('kernel'),
        kernel='rbf',
        # gamma=params.get('gamma'),
        gamma='auto',
        # C=params.get('C'),
        C=0.019
        # degree=params.get('degree')
    )
    svm_model_custom.fit(training_features, is_speaker)
    # score = model_selection.cross_val_score(svm_model_custom, files, is_speaker, cv=10, scoring='accuracy')
    save_model(speaker_id, 'svm_custom', svm_model_custom)

    after_time = datetime.now()
    duration = after_time - start_time
    hours = duration.total_seconds() // 3600
    minutes = (duration.total_seconds() // 60) - (hours * 60)
    seconds = duration.total_seconds() - (hours * 3600) - (minutes * 60)
    print("--> duration: %0.0fh:%0.0fmin:%0.2fsec" % (hours, minutes, seconds),
          # "----- Model: accuracy: %f; standard deviation of %f" % (score.mean(), score.std())
          )
    # save_model(speaker_id, 'svm', svm_model)
