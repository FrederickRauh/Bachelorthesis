from backend.gmm.gmm_predictor import Predictor as gmm_pred
from backend.svm.svm_predictor import Predictor as svm_pred
from backend.trainer import Trainer

from utils import dataframeManager as dam
from utils import directoryManager as dm

print("starting...")
speaker_ids = dm.get_all_ids()
finished_ids = ['id10001', 'id10002', 'id10003', 'id10004', 'id10005',
                'id10006', 'id10007', 'id10008', 'id10009', 'id10010',
                'id10011', 'id10012', 'id10013', 'id10014', 'id10015',
                'id10016', 'id10017', 'id10018', 'id10019', 'id10020'
                ]
#                 ,
#                 'id10021', 'id10022', 'id10023', 'id10024', 'id10025',
#                 'id10026', 'id10027', 'id10028', 'id10029', 'id10030',
#                 'id10031', 'id10032', 'id10033', 'id10034', 'id10035']
#                 ,
#                 'id10036', 'id10037', 'id10038', 'id10039', 'id10040',
#                 'id10041', 'id10042', 'id10043', 'id10044']
# speaker_ids = [speaker_ids[0]]
# speaker_ids = util.remove_finished_ids(speaker_ids, finished_ids)
# speaker_ids = finished_ids


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


# # Training phase
print("training phase...")
# # # # # # # # # # Multi Process # # # # # # # # # #
# trainer = Mult_Trainer()
# dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
# dataframe = dam.load_dataframe_from_path(dataframe_path)
# trainer.train_multi(speaker_ids, dataframe)

# # # # # # # # # # # Single Process # # # # # # # # # #
trainer = Trainer()
dataframe_path = dm.get_all_data_path() + '\\' + 'psf-dataframe.json'
dataframe = dam.load_dataframe_from_path(dataframe_path)
trainer.train_multi_svm(speaker_ids, dataframe)
# trainer.train_multi_gmm(speaker_ids, dataframe)


# #Prediction phase
print("prediction phase...")
svm_pred = svm_pred()
svm_pred.predict_multiple_speakers_svm(speaker_ids)
# gmm_pred = gmm_pred()
# gmm_pred.predict_multiple_speakers_gmm(speaker_ids)


