import logging
import os
from configparser import ConfigParser
from datetime import datetime

import frontend.frontend
from backend.gmm_ubm import GMMUBM

from utils import directoryManager as dm, util


"""
GMM-UBM
"""
gmm_ubm = GMMUBM()
if __name__ == '__main__':
    #############Config##############
    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)

    feature_type = config.get('features', 'FEATURE_TYPE')

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-gmmubm.log', level=config.getint('system', 'LOGLEVEL'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'LOG')
    if config.getboolean('system', 'LOG'):
        print("container running. logs can be found in info-{model_type}.log")

    start_time = datetime.now()
    logging.info(f"Version GMM-UBM :{start_time}")
    logging.info(f"FEATURE_VERSION: {feature_type}")
    # preparation phase
    if config.getboolean('system', 'EXTRACT_FEATURES'):
        logging.info(f"extracting features...")
        frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=dm.get_all_ids(), create_dataframe=True)
    # training phase
    if config.getboolean('system', 'TRAIN_MODEL'):
        logging.info(f"train models...")
        gmm_ubm.train(speaker_ids=dm.get_all_ids())
    # prediction phase
    if config.getboolean('system', 'PREDICT_SPEAKER'):
        logging.info(f"predicting speaker...")
        gmm_ubm.predict_speaker(speaker_id=dm.get_all_ids()[0], speaker_ids=dm.get_all_ids())

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")


