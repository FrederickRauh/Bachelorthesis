import logging
from configparser import ConfigParser

import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm import GMM

from utils import directoryManager as dm, util

gmm = GMM()

if __name__ == '__main__':
    #############Config##############
    file = dm.get_project_path() + '\\' + 'config.ini'
    config = ConfigParser()
    config.read(file)

    feature_type = config.get('features', 'FEATURE_TYPE')

    logging.basicConfig(level=0)
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'LOG')

    start_time = datetime.now()
    logging.info(f"Version GMM :{start_time}")
    logging.info(f"FEATURE_VERSION: {feature_type}")
    # preparation phase
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=dm.get_all_ids(), create_dataframe=True)
    # training phase
    gmm.train(speaker_ids=dm.get_all_ids())
    # prediction phase
    gmm.predict_n_speakers(speaker_ids=dm.get_all_ids())

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")