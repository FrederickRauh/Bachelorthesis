import os

import scipy.io.wavfile as wav
from os.path import dirname, join as pjoin


def create_wav_file_name(speaker_id, number):
    number = f"{number:05}"
    if speaker_id == '':
        return str(number) + '.wav'
    else:
        return speaker_id + '\\' + str(number) + '.wav'


def create_csv_file_name(speaker_id):
    return speaker_id + '.csv'


def get_parent_path(speaker_id):
    parent_path = make_dir(get_all_wav_path())
    return make_dir(os.path.join(parent_path, speaker_id))


def sub_folder_switch(x):
    return {
        'csv': 1,
        'wav': 2
    }.get(x, 9)


def list_sub_folders(parent_path):
    return os.listdir(parent_path)


def get_sub_folder_path(parent_path, sub_folder):
    path = os.path.join(parent_path, sub_folder)
    if not os.path.exists(path):
        make_dir(path)
    return path


def create_sub_folder(parent_path, sub_folder_name):
    make_dir(os.path.join(parent_path, sub_folder_name))


def get_csv_path(speaker_id):
    parent_path = get_parent_path(speaker_id)
    csv_path = get_sub_folder_path(parent_path, 'csv')
    filename = create_csv_file_name(speaker_id)
    return csv_path + '\\' + filename


def create_feature_csv_dir(file_path):
    sub_path = file_path.split('\\')
    new_dir_path = ''
    for x in range(len(sub_path)):
        if not x == (len(sub_path) - 1):
            new_dir_path = new_dir_path + sub_path[x] + '\\'
    make_dir(new_dir_path)


def get_feature_path(wav_path, version):
    sub_path = wav_path.split('\\')
    feature_path = ''
    for x in range(len(sub_path)):
        if x == (len(sub_path) - 1):
            feature_path = feature_path + '\\' + version + '\\' + sub_path[x]
        else:
            feature_path = feature_path + sub_path[x] + '\\'
    return feature_path


def get_feature_librosa_csv_path(wav_path):
    json_path = get_feature_path(wav_path, 'librosa')
    return json_path.replace('.wav', '.csv')


def get_feature_librosa_json_path(wav_path):
    json_path = get_feature_path(wav_path, 'librosa')
    return json_path.replace('.wav', '.json')


def get_feature_psf_csv_path(wav_path):
    json_path = get_feature_path(wav_path, 'psf')
    return json_path.replace('.wav', '.csv')


def get_feature_psf_json_path(wav_path):
    json_path = get_feature_path(wav_path, 'psf')
    return json_path.replace('.wav', '.json')


def create_feature_json_dir(file_path):
    sub_path = file_path.split('\\')
    new_dir_path = ''
    for x in range(len(sub_path)):
        if not x == (len(sub_path) - 1):
            new_dir_path = new_dir_path + sub_path[x] + '\\'
    make_dir(new_dir_path)


def get_wav_folder_path(speaker_id):
    parent_path = get_parent_path(speaker_id)
    return get_sub_folder_path(parent_path, 'wav')


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as error:
            print("Creating directory %s has failed. Error %s" % (path, error))
    return path


def get_file_name(speaker_id, number):
    parent_path = get_parent_path(speaker_id)
    wav_path = get_sub_folder_path(parent_path, 'wav')
    file_name = create_wav_file_name('', number)
    return pjoin(wav_path, file_name)


def get_wav_files_in_folder(path):
    files = []
    wav_files = []
    dir_path = path
    for base, dirs2, Files in os.walk(dir_path):
        if not base.endswith('\librosa') and not base.endswith('\psf'):
            files = Files
    for file in files:
        if file.endswith('.wav'):
            wav_files.append(dir_path + '\\' + file)
        # files = []
    return wav_files


def remove_wav_files_if_voxceleb(directories, parent_path):
    if directories.__contains__("model"):
        directories.remove('model')
    last_dir = directories.pop()
    if last_dir.__contains__("psf"):
        last_dir.remove('psf')
    if last_dir.__contains__('librosa'):
        last_dir.remove('librosa')
    last_dir_path = parent_path + '\\' + last_dir
    i = 0
    for base, dirs2, Files in os.walk(last_dir_path):
        for file in Files:
            if file.endswith(".wav"):
                i += 1
    if i < 10:
        directories.pop()
    return directories


def get_wav_files(speaker_id):
    parent_path = get_parent_path(speaker_id)
    directories = list_sub_folders(parent_path)
    # -------------------------------------------------
    # done in order to keep unseen data for testing afterwards
    if is_large_data_set():
        directories = remove_wav_files_if_voxceleb(directories, parent_path)
    files = []
    wav_files = []
    for directory in directories:
        if not directory.__contains__('csv') and not directory == 'model':
            dir_path = parent_path + '\\' + directory
            for base, dirs2, Files in os.walk(dir_path):
                if not base.endswith('\librosa') and not base.endswith('\psf'):
                    files = Files
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(directory + '\\' + file)
            files = []
    return wav_files


def get_all_models_path():
    path = get_all_data_path() + '\\' + 'models'
    if not os.path.exists(path):
        make_dir(path)
    return path


def get_model_path(speaker_id, t):
    path = get_all_models_path()
    if t.__contains__('svm'):
        path = path + '\\' + "svm"
    if t.__contains__('gmm'):
        path = path + '\\' + "gmm"

    if not os.path.exists(path):
        make_dir(path)

    for base, dirs2, Files in os.walk(path):
        models = Files
    for model in models:
        if model.__contains__(speaker_id) and model.__contains__(t):
            return path + '\\' + model

    file_name = speaker_id + "_" + t + "_model.pickel"
    path = path + '\\' + file_name
    return path
    
    # model_folder_path = get_sub_folder_path(parent_path, 'model')
    #
    # return model_folder_path + '\\' + file_name


def get_id_of_path(path):
    sub_paths = path.split('\\')
    for sub_path in sub_paths:
        if sub_path.__contains__('id'):
            return sub_path
    return "no id in path"


def check_if_file_exists_then_remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def get_all_wav_path():
    return get_all_data_path() + '\\' + "wav"



def get_all_ids():
    ids = get_all_wav_names()
    # if ids.__contains__('pairs.csv'):
    #     ids.remove('pairs.csv')
    # if ids.__contains__('dataframe.csv'):
    #     ids.remove('dataframe.csv')
    # if ids.__contains__('librosa-dataframe.json'):
    #     ids.remove('librosa-dataframe.json')
    # if ids.__contains__('psf-dataframe.json'):
    #     ids.remove('psf-dataframe.json')
    # if ids.__contains__('result-svm.json'):
    #     ids.remove('result-svm.json')
    # if ids.__contains__('result-gmm.json'):
    #     ids.remove('result-gmm.json')
    return ids


def get_all_data_csv_file_path():
    path = get_all_data_path() + '\\' + 'pairs.csv'
    return path


def get_all_wav_names():
    return os.listdir(get_all_wav_path())


# used to switch between
def get_all_data_path():
    # return os.path.join(get_project_path(), "data")
    return get_my_path()


def is_large_data_set():
    if get_all_data_path().__contains__('voxceleb'):
        return True
    return False


def get_project_path():
    return os.getcwd()


def get_my_path():
    return 'E:' + '\\' + 'voxceleb' + '\\' + 'vox1_bigba_wav'


def get_test_path():
    return os.path.join(get_project_path(), "test")


def get_test_subfolders(speaker_id):
    path = get_test_path() + '\\' + speaker_id
    return os.listdir(path)


def get_voxceleb_path():
    return 'E:' + '\\' + 'voxceleb' + '\\' + 'vox1_dev_wav' + '\\' + 'wav'


def get_voxceleb_subfolders(speaker_id):
    path = get_voxceleb_path() + '\\' + speaker_id
    return os.listdir(path)
