from backend import trainer
from backend.trainer import Trainer
from backend.predictor import Predictor
from backend import model as m
from frontend import frontend as fr
from utils import directoryManager as dm
from utils import csvManager as cm

print("starting...")
speaker_id = 'id00001'

# timespan(sec), samplerate, number, speaker_id
# frontend.get_voice_input_stream(4, 44100, 1, speaker_id)
#
#
cm.create_overall_csv()
speaker_ids = dm.get_all_ids()
print("IDs : ", speaker_ids)
trainer = Trainer()

# for speaker_id in speaker_ids:
#     files = dm.get_wav_files(speaker_id)
    # for file in files:
    #     file_path = dm.get_parent_path(speaker_id) + '\\' + file
    #     fr.extract_mfcc_from_file_to_csv(file_path)
    # trainer.train_svm(speaker_id)

# speaker_id = 'id00001'
# file = speaker_id + '\\' + "wav" + '\\' + 'id00001-0.csv'
# files = [file]
#
# print(m.get_correct_feature_array(files))
# data = m.get_features_out_of_csv(files)
# print(data)

# frontend.process_features(speaker_id)



predictor = Predictor()
for speaker_id in speaker_ids:
    dir = dm.get_voxceleb_subfolders(speaker_id)
    dir_path = dir[len(dir) - 1]
    files_path = dm.get_voxceleb_path() + '\\' + speaker_id + '\\' + dir_path
    # print(files_path)
    test_files = dm.get_wav_files_in_folder(files_path)
    for file in test_files:
        score = predictor.predict_svm(speaker_id, file)
        if score == 1:
            print('ITS HIM ::::: ', speaker_id, ', ', files_path, ',', score)
        else:
            print(score)


# files_path = 'id10002' + '\\' + 'Y2Gr1I2DO7M'
# path = test_path + '\\' + files_path
# test_files = dm.get_wav_files_in_folder(path)



