import logging
import os
from configparser import ConfigParser
from datetime import datetime

import frontend.frontend
from backend.svm import SVM

from utils import directoryManager as dm, util

"""
SVM
"""
svm = SVM()
if __name__ == '__main__':
    #############Config##############
    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)

    feature_type = config.get('features', 'FEATURE_TYPE')

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-svm.log', level=config.getint('system', 'LOGLEVEL'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'LOG')

    start_time = datetime.now()
    logging.info(f"Version SVM :{start_time}")
    logging.info(f"FEATURE_VERSION: {feature_type}")
    # preparation phase
    if config.getboolean('system', 'EXTRACT_FEATURES'):
        logging.info(f"extracting features...")
        frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=dm.get_all_ids(), create_dataframe=True)
    # training phase
    if config.getboolean('system', 'TRAIN_MODEL'):
        logging.info(f"train models...")
        svm.train(speaker_ids=dm.get_all_ids())
    # prediction phase
    if config.getboolean('system', 'PREDICT_SPEAKER'):
        logging.info(f"predicting speaker...")
        svm.predict_n_speakers(speaker_ids=dm.get_all_ids())

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
