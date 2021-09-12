import configparser
import json
import logging
from configparser import ConfigParser
from datetime import datetime

import frontend.frontend
from backend.gmm import GMM
from backend.gmm_ubm import GMMUBM
from backend.svm import SVM

from utils import directoryManager as dm, trainingTestingManager as tt, util

if __name__ == '__main__':
    #############Config##############
    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)

    logging.basicConfig(filename=rf'{dm.get_project_path()}/predictions.log', level=config.getint('system', 'loglevel'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'log')

    if config.getboolean('system', 'log'):
        print("container running. logs can be found in info.log")

    feature_type = config.get('features', 'feature_type')
    speaker_ids = dm.get_all_ids()

    test_files = []
    if config.getboolean('stage', 'predict_speaker'):
        start_time = datetime.now()
        test_files, _ = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
        logging.info(f"loaded {len(test_files)} testing files, time spent: {util.get_duration(start_time)}")

    try:
        ids = json.loads(config.get("system", "ids"))
        if not ids == []:
            speaker_ids = ids
            logging.info(f"ids to process: \n {speaker_ids}")
    except configparser.NoOptionError:
        logging.info(f"No ids specified, using all")

    lengths = [28, 60, 120, 180, 240, 300]
    for length in lengths:
        """
        GMM
        """
        gmm = GMM()
        if config.getboolean("classifier", "gmm"):

            # prediction phase
            if config.getboolean('stage', 'predict_speaker'):
                start_time = datetime.now()
                logging.info(f"GMM::::::::::::::::predicting speaker..., {start_time}")
                gmm.predict_n_speakers(speaker_ids=speaker_ids,
                                       test_files=test_files,
                                       extra=length,
                                       attack_type=length)
                logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

        """
        GMM-UBM
        """

        gmm_ubm = GMMUBM()
        if config.getboolean("classifier", "gmm_ubm"):

            # prediction phase
            if config.getboolean('stage', 'predict_speaker'):
                start_time = datetime.now()
                logging.info(f"GMM-UBM::::::::::::::::predicting speaker..., {start_time}")
                gmm_ubm.predict_n_speakers(speaker_ids=speaker_ids,
                                           test_files=test_files,
                                           extra=length,
                                           attack_type=length)
                logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

        """
        SVM
        """
        svm = SVM()
        if config.getboolean("classifier", "svm"):

            # prediction phase
            if config.getboolean('stage', 'predict_speaker'):
                start_time = datetime.now()
                logging.info(f"SVM::::::::::::::::predicting speaker..., {start_time}")
                svm.predict_n_speakers(speaker_ids=speaker_ids,
                                       test_files=test_files,
                                       extra=length,
                                       attack_type=length)
                logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
