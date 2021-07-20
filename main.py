import numpy as np

from datetime import datetime

from backend.gmm.gmm_predictor import Predictor as gmm_pred
from backend.svm.svm_predictor import Predictor as svm_pred
from backend.trainer import Trainer

from frontend import featureExtractorPSF as fpsf, featureExtractorLibrosa as flib

from utils.config import IDS, FEATURES, CONFIG, MODELCONFIG
from utils import dataframeManager as dam, debug, directoryManager as dm, util


#############Config##############
debug.log(("Working in:", dm.get_all_data_path()))
version = np.arange(0, 5, 1)
mfccs = np.arange(13, 41, 1)
trainer = Trainer()
svm_pred = svm_pred()
gmm_pred = gmm_pred()

MODELCONFIG.overwrite_n_jobs(MODELCONFIG, -2)

debug.log(("starting..."))
for v in version:
    CONFIG.overwrite_version(CONFIG, v)
    for mfcc in mfccs:
        start_time = datetime.now()
        FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
        debug.log(("MFCC_COUNT:", FEATURES.N_MFCC, " :", start_time))

        # # preparation phase
        debug.log(("feature extraction..."))
        for speaker_id in IDS.speaker_ids:
            files = dm.get_wav_files(speaker_id)
            for file in files:
                file_path = dm.get_parent_path(speaker_id) + '\\' + file
                flib.extract_mfcc_from_file_to_json(file_path)
        # dam.create_librosa_dataframe(config_id.speaker_ids)

        # # Training phase
        debug.log(("training..."))
        # # # # # # # # # # # Single Process # # # # # # # # # #
        dataframe_librosa_path = dm.get_all_data_path() + '\\' + 'librosa-dataframe.json'
        dataframe_librosa = dam.load_dataframe_from_path(dataframe_librosa_path)
        trainer.train_multi_svm(IDS.speaker_ids, dataframe_librosa, feature_type='librosa')
        trainer.train_multi_gmm(IDS.speaker_ids, dataframe_librosa, feature_type='librosa')

        # #Prediction phase
        debug.log(("predicting..."))
        svm_pred.predict_multiple_speakers_svm(IDS.speaker_ids, feature_type='librosa')
        gmm_pred.predict_multiple_speakers_gmm(IDS.speaker_ids, feature_type='librosa')

        debug.log(("MFCC_COUNT:", FEATURES.N_MFCC, ":", util.get_duration(start_time)))
