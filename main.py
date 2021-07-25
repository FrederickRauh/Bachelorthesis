import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm.gmm_predictor import Predictor as gmm_pred
from backend.svm.svm_predictor import Predictor as svm_pred
from backend.trainer import Trainer

from utils.config import IDS, FEATURES, CONFIG, MODELCONFIG
from utils import debug, directoryManager as dm, util


def preparation_phase(mfcc):
    start_time = datetime.now()
    debug.log(("feature extraction...."))
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=IDS.SPEAKER_IDS, create_dataframe=False, feature_type='librosa', mfcc_count=mfcc)
    debug.log((util.get_duration(start_time)))


def training_phase(version):
    start_time = datetime.now()
    debug.log(("training..."))
    if version == 'gmm':
        trainer.train_multi_gmm(IDS.SPEAKER_IDS, feature_type='librosa')
    if version == 'svm':
        trainer.train_multi_svm(IDS.SPEAKER_IDS, feature_type='librosa')
    debug.log((util.get_duration(start_time)))


def prediction_phase(version, mfcc):
    start_time = datetime.now()
    debug.log(("predicting..."))
    if version == 'gmm':
        gmm_pred.predict_n_speakers_gmm(IDS.SPEAKER_IDS, feature_type='librosa', mfcc=mfcc)
    if version == 'svm':
        svm_pred.predict_n_speakers_svm(IDS.SPEAKER_IDS, feature_type='librosa', mfcc=mfcc)
    debug.log((util.get_duration(start_time)))


if __name__ == '__main__':
    #############Config##############
    debug.log(("Working in:", dm.get_all_data_path()))
    version = np.arange(0, 6, 1)
    mfccs = np.arange(12, 41, 1)
    trainer = Trainer()
    svm_pred = svm_pred()
    gmm_pred = gmm_pred()

    # MODELCONFIG.overwrite_n_jobs(MODELCONFIG, -2)

    for v in version:
        debug.log(("starting version:", v, " ..."))
        CONFIG.overwrite_version(CONFIG, v)

        for mfcc in mfccs:
            FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
            start_time_mfcc = datetime.now()
            debug.log(("MFCC_COUNT:", FEATURES.N_MFCC, " Version GMM :", start_time_mfcc))

            preparation_phase(mfcc)
            training_phase('gmm')
            prediction_phase('gmm', mfcc)

            debug.log(("MFCC_COUNT:", FEATURES.N_MFCC, ":", "----------------------------------------------------------", util.get_duration(start_time_mfcc)))

        for mfcc in mfccs:
            FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
            start_time = datetime.now()
            debug.log(("MFCC_COUNT:", FEATURES.N_MFCC, " Version SVM :", start_time))

            preparation_phase(mfcc)
            training_phase('svm')
            prediction_phase('svm', mfcc)

            debug.log(("MFCC_COUNT:", FEATURES.N_MFCC, ":", "----------------------------------------------------------", util.get_duration(start_time)))


