import logging

import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm import GMM
from backend.gmm_ubm import GMMUBM
from backend.svm import SVM

from config import IDS, FEATURES, CONFIG, SYSTEM
from utils import util


def preparation_phase(mfcc):
    # frontend.frontend.feature_extraction_for_files([IDS.SPEAKER_IDS[0]], 'librosa', 20)

    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=IDS.SPEAKER_IDS, create_dataframe=True,
                                                       feature_type=SYSTEM.FEATURE_TYPE, mfcc_count=mfcc)


def training_phase(version):
    if version == 'gmm':
        gmm.train(speaker_ids=IDS.SPEAKER_IDS)
    if version == 'gmm-ubm':
        gmm_ubm.train(speaker_ids=IDS.SPEAKER_IDS)
    if version == 'svm':
        svm.train(speaker_ids=IDS.SPEAKER_IDS)


def prediction_phase(version, mfcc):
    if version == 'gmm':
        gmm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, mfcc=mfcc)
    if version == 'gmm-ubm':
        gmm_ubm.predict_speaker(speaker_id=IDS.SPEAKER_IDS[0], speaker_ids=IDS.SPEAKER_IDS)
        # gmm_ubm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, mfcc=mfcc)
    if version == 'svm':
        svm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, mfcc=mfcc)


if __name__ == '__main__':
    #############Config##############
    logging.basicConfig(level=0)
    # logger = logging.getLogger()
    # logger.disabled = True
    version = np.arange(0, 1, 1)
    mfccs = np.arange(20, 21, 1)
    gmm = GMM()
    gmm_ubm = GMMUBM()
    svm = SVM()

    # MODELCONFIG.overwrite_n_jobs(MODELCONFIG, -2)

    for v in version:
        logging.info(f"starting version: {v}  ...")
        CONFIG.overwrite_version(CONFIG, v)

        for mfcc in mfccs:
            preparation_phase(mfcc)

            FEATURES.overwrite_n_mfcc(FEATURES, mfcc)

            """
            GMM
            """
            start_time_gmm = datetime.now()
            logging.info(f"MFCC_COUNT: {FEATURES.N_MFCC} Version GMM : {start_time_gmm}")

            training_phase('gmm')
            prediction_phase('gmm', mfcc)

            logging.info(
                f"MFCC_COUNT:{FEATURES.N_MFCC}: ----------------------------------------------------------{util.get_duration(start_time_gmm)}")

            """
            GMM-UBM
            """
            start_time_gmm = datetime.now()
            logging.info(f"MFCC_COUNT: {FEATURES.N_MFCC} Version GMM-UBM : {start_time_gmm}")

            training_phase('gmm-ubm')
            prediction_phase('gmm-ubm', mfcc)

            logging.info(
                f"MFCC_COUNT:{FEATURES.N_MFCC}: ----------------------------------------------------------{util.get_duration(start_time_gmm)}")


            """
            SVM
            """
            # start_time_svm = datetime.now()
            # logging.info(f"MFCC_COUNT: {FEATURES.N_MFCC} Version SVM :{start_time_svm}")
            #
            # training_phase('svm')
            # prediction_phase('svm', mfcc)
            #
            # logging.info(
            #     f"MFCC_COUNT:{FEATURES.N_MFCC}: ----------------------------------------------------------{util.get_duration(start_time_svm)}")
