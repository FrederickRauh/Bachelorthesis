import numpy as np

from backend.gmm.gmm_predictor import Predictor as gmm_pred
from backend.svm.svm_predictor import Predictor as svm_pred
from backend.trainer import Trainer

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib

from utils.config import Ids as config_id, Features
from utils import dataframeManager as dam, directoryManager as dm, util

print("starting...")
#############Config##############
config_id = config_id()
mfccs = np.arange(13, 41, 1)
trainer = Trainer()
svm_pred = svm_pred()
gmm_pred = gmm_pred()

# mfccs = [20]
for mfcc in mfccs:
    Features.over_write_n_mfcc(Features, mfcc)
    print("MFCC_COUNT", Features.N_MFCC)
    # # preparation phase
    for speaker_id in config_id.speaker_ids:
        files = dm.get_wav_files(speaker_id)
        for file in files:
            file_path = dm.get_parent_path(speaker_id) + '\\' + file
            flib.extract_mfcc_from_file_to_json(file_path)
    dam.create_librosa_dataframe(config_id.speaker_ids)

    # # Training phase
    print("training...")
    # # # # # # # # # # # Single Process # # # # # # # # # #
    dataframe_librosa_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
    dataframe_librosa = dam.load_dataframe_from_path(dataframe_librosa_path)
    trainer.train_multi_svm(config_id.speaker_ids, dataframe_librosa, feature_type='librosa')
    trainer.train_multi_gmm(config_id.speaker_ids, dataframe_librosa, feature_type='librosa')

    # #Prediction phase
    print("predicting...")
    svm_pred.predict_multiple_speakers_svm(config_id.speaker_ids, feature_type='librosa')
    gmm_pred.predict_multiple_speakers_gmm(config_id.speaker_ids, feature_type='librosa')
