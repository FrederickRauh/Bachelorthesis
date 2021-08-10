import logging
from configparser import ConfigParser

from datetime import datetime

import frontend.frontend
from backend.svm import SVM

from utils import directoryManager as dm, util

svm = SVM()

if __name__ == '__main__':
    #############Config##############
    file = rf'{dm.get_project_path()}//config.ini'
    print(file)
    config = ConfigParser()
    config.read(file)

    feature_type = config.get('features', 'FEATURE_TYPE')

    logging.basicConfig(level=0)
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'LOG')

    start_time = datetime.now()
    logging.info(f"Version SVM :{start_time}")
    logging.info(f"FEATURE_VERSION: {feature_type}")
    # preparation phase
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=dm.get_all_ids(), create_dataframe=True)
    # training phase
    svm.train(speaker_ids=dm.get_all_ids())
    # prediction phase
    svm.predict_n_speakers(speaker_ids=dm.get_all_ids())

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
