import logging

import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm_ubm import GMMUBM

from utils import directoryManager as dm, util

from configparser import ConfigParser


def preparation_phase():
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=dm.get_all_ids(), create_dataframe=True)


def training_phase(version):
    if version == 'gmm-ubm':
        gmm_ubm.train(speaker_ids=dm.get_all_ids())


def prediction_phase(version):
    if version == 'gmm-ubm':
        gmm_ubm.predict_speaker(speaker_id=dm.get_all_ids()[0], speaker_ids=dm.get_all_ids())


if __name__ == '__main__':
    #############Config##############
    file = 'config.ini'
    config = ConfigParser()
    config.read(file)

    logging.basicConfig(level=0)
    # logger = logging.getLogger()
    # logger.disabled = True
    version = np.arange(0, 1, 1)
    # version = [5, 8, 13, 16, 20]
    # mfccs = np.arange(20, 21, 1)
    mfcc = 20

    gmm_ubm = GMMUBM()

    # MODELCONFIG.overwrite_n_jobs(MODELCONFIG, -2)

    for v in version:
        logging.info(f"starting version: {v}  ...")
        # config.overwrite_version(v)

        # preparation_phase()

        """
        GMM
        """


        """
        GMM-UBM
        """
        start_time_gmm = datetime.now()
        logging.info(f"FEATURE_VERSION: {config['features']['FEATURE_TYPE']} Version GMM-UBM : {start_time_gmm}")

        training_phase('gmm-ubm')
        prediction_phase('gmm-ubm')

        logging.info(
            f"----------------------------------------------------------{util.get_duration(start_time_gmm)}")


        """
        SVM
        """

