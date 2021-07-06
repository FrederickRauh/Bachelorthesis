import json

import numpy as np
import pandas as pd

from backend import trainer
from backend.trainer import Trainer
from backend.svm_predictor import Predictor as svm_pred
from backend.gmm_predictor import Predictor as gmm_pred
from backend import svm_model as m

from frontend import featureExtractorLibrosa as flib
from frontend import featureExtractorPSF as fpsf
from frontend import frontend as fr

from utils import dataframeManager as dam
from utils import directoryManager as dm
from utils import fileManager as cm
from utils import util

#
# finished_ids = ['id10001', 'id10002', 'id10003', 'id10004', 'id10005',
#                 'id10006', 'id10007', 'id10008', 'id10009', 'id10010',
#                 'id10011', 'id10012', 'id10013', 'id10014', 'id10015',
#                 'id10016', 'id10017']
finished_ids = []
# id00001 Start at:  2021-06-29 00:26:27.136246, id10017 Start at:  2021-07-01 04:09:48.288758
print("starting...")
speaker_ids = dm.get_all_ids()
speaker_ids = util.remove_finished_ids(speaker_ids, finished_ids)
# speaker_id = 'id00001'
# # # # timespan(sec), samplerate, amount, speaker_id, (test file?)
# fr.get_voice_input_stream(4, 16000, 100, speaker_id, False)


# # preparation phase
print("prep phase...")  # create the overall csv, extract mfcc from files and create dataframes(json)
# # cm.create_overall_csv()
# for speaker_id in speaker_ids:
#     files = dm.get_wav_files(speaker_id)
#     for file in files:
#         file_path = dm.get_parent_path(speaker_id) + '\\' + file
#         # flib.extract_mfcc_from_file_to_json(file_path)
#         fpsf.extract_mfcc_from_file_to_json(file_path)
# # dam.create_librosa_dataframe(speaker_ids)
# dam.create_psf_dataframe(speaker_ids)
#
# # Training phase
print("training phase...")
trainer = Trainer()
dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
dataframe = dam.load_dataframe_from_path(dataframe_path)
for speaker_id in speaker_ids:
    trainer.train_svm(dataframe, speaker_id)

# #Prediction phase
print("prediction phase...")
svm_pred = svm_pred()
svm_pred.predict_multiple_speakers_svm(speaker_ids)
# gmm_pred = gmm_pred()
# gmm_pred.predict_multiple_speakers_gmm(speaker_ids)
