import pandas as pd

from backend import trainer
from backend.trainer import Trainer
from backend.predictor import Predictor
from backend import svm_model as m

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf
from frontend import frontend as fr

from utils import dataframeManager as dam
from utils import directoryManager as dm
from utils import fileManager as cm

print("starting...")

# speaker_id = 'id00001'
# # # # timespan(sec), samplerate, amount, speaker_id, (test file?)
# fr.get_voice_input_stream(4, 44100, 100, speaker_id, False)


speaker_ids = dm.get_all_ids()

# # # preparation phase
# print("prep phase...") # create the overall csv, extract mfcc from files and create dataframes(json)
# cm.create_overall_csv()
# for speaker_id in speaker_ids:
#     files = dm.get_wav_files(speaker_id)
#     for file in files:
#         file_path = dm.get_parent_path(speaker_id) + '\\' + file
#         flib.extract_mfcc_from_file_to_json(file_path)
#         fpsf.extract_mfcc_from_file_to_json(file_path)
# dam.create_librosa_dataframe(speaker_ids)
# dam.create_psf_dataframe(speaker_ids)
#
# # # Training phase
print("training phase...")
# trainer = Trainer()
# dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
# dataframe = dam.load_dataframe_from_path(dataframe_path)
# for speaker_id in speaker_ids:
#     trainer.train_svm(dataframe, speaker_id)



# # #Prediction phase
print("prediction phase...")
predictor = Predictor()
overall_test_files = []
for speaker_id in speaker_ids:
    # dir = dm.get_test_subfolders(speaker_id)
    dir = dm.get_voxceleb_subfolders(speaker_id)
    dir_path = dir[len(dir) - 1]
    files_path = dm.get_test_path() + '\\' + speaker_id + '\\' + dir_path
    files_path = dm.get_voxceleb_path() + '\\' + speaker_id + '\\' + dir_path
    test_files = dm.get_wav_files_in_folder(files_path)
    for x in range(len(test_files)):
        overall_test_files.append(test_files[x])

print("overall_test_files length: ", len(overall_test_files))

for speaker_id in speaker_ids:
    speaker_report = []
    predictor.predict_speaker(speaker_id, 'svm', overall_test_files)




