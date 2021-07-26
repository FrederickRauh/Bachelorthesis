import logging

import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm import GMM
from backend.svm import SVM
from backend.trainer import Trainer

from utils.config import IDS, FEATURES, CONFIG, SYSTEM
from utils import util


def preparation_phase(mfcc):
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=IDS.SPEAKER_IDS, create_dataframe=False,
                                                       feature_type=SYSTEM.FEATURE_TYPE, mfcc_count=mfcc)


def training_phase(version):
    if version == 'gmm':
        gmm.train(speaker_ids=IDS.SPEAKER_IDS)
    if version == 'svm':
        svm.train(speaker_ids=IDS.SPEAKER_IDS)


def prediction_phase(version, mfcc):
    if version == 'gmm':
        gmm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, mfcc=mfcc)
    if version == 'svm':
        svm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, mfcc=mfcc)


if __name__ == '__main__':
    #############Config##############
    version = np.arange(0, 1, 1)
    mfccs = np.arange(20, 21, 1)
    trainer = Trainer()
    svm = SVM()
    gmm = GMM()

    # MODELCONFIG.overwrite_n_jobs(MODELCONFIG, -2)

    for v in version:
        logging.debug("starting version:", v, " ...")
        CONFIG.overwrite_version(CONFIG, v)

        for mfcc in mfccs:
            preparation_phase(mfcc)

            FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
            start_time_mfcc = datetime.now()
            logging.debug("MFCC_COUNT:", FEATURES.N_MFCC, " Version GMM :", start_time_mfcc)

            training_phase('gmm')
            prediction_phase('gmm', mfcc)

            logging.debug(
                      "MFCC_COUNT:", FEATURES.N_MFCC, ":", "----------------------------------------------------------",
                      util.get_duration(start_time_mfcc))

            # FEATURES.overwrite_n_mfcc(FEATURES, mfcc)
            start_time = datetime.now()
            logging.debug("MFCC_COUNT:", FEATURES.N_MFCC, " Version SVM :", start_time)

            training_phase('svm')
            prediction_phase('svm', mfcc)

            logging.debug(
                      "MFCC_COUNT:", FEATURES.N_MFCC, ":", "----------------------------------------------------------",
                      util.get_duration(start_time))
