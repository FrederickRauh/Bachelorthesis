import logging

import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm_ubm import GMMUBM

from utils import directoryManager as dm, util

from configparser import ConfigParser

gmm_ubm = GMMUBM()

if __name__ == '__main__':
    #############Config##############
    file = dm.get_project_path() + '\\' + 'config.ini'
    config = ConfigParser()
    config.read(file)

    feature_type = config.get('features', 'FEATURE_TYPE')

    logging.basicConfig(level=0)
    # logger = logging.getLogger()
    # logger.disabled = True
    version = np.arange(0, 1, 1)
    # version = [5, 8, 13, 16, 20]
    # mfccs = np.arange(20, 21, 1)
    mfcc = 20



    # MODELCONFIG.overwrite_n_jobs(MODELCONFIG, -2)

    for v in version:
        logging.info(f"starting version: {v}  ...")
        # config.overwrite_version(v)

        frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=dm.get_all_ids(), create_dataframe=True)

        """
        GMM
        """


        """
        GMM-UBM
        """
        start_time_gmm = datetime.now()
        logging.info(f"FEATURE_VERSION: {config['features']['FEATURE_TYPE']} Version GMM-UBM : {start_time_gmm}")

        gmm_ubm.train(speaker_ids=dm.get_all_ids())
        gmm_ubm.predict_speaker(speaker_id=dm.get_all_ids()[0], speaker_ids=dm.get_all_ids())

        logging.info(
            f"----------------------------------------------------------{util.get_duration(start_time_gmm)}")


        """
        SVM
        """

