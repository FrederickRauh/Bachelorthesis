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

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info.log', level=config.getint('system', 'loglevel'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'log')

    if config.getboolean('system', 'log'):
        print("container running. logs can be found in info-{model_type}.log")



    feature_type = config.get('features', 'feature_type')
    speaker_ids = dm.get_all_ids()
    # speaker_ids = speaker_ids[10:]
    # preparation phase
    if config.getboolean('system', 'extract_features'):
        logging.info(f"extracting features...")
        frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=speaker_ids, create_dataframe=True)

    test_files = []
    if config.getboolean('system', 'predict_speaker'):
        start_time = datetime.now()
        test_files, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
        logging.info(f"loaded {len(test_files)} testing files, time spent: {util.get_duration(start_time)}")

    """
    GMM
    """
    gmm = GMM()
    if config.getboolean("system", "gmm"):
        start_time = datetime.now()
        logging.info(f"Version GMM :{start_time}")
        logging.info(f"FEATURE_VERSION: {feature_type}")

        # training phase
        if config.getboolean('system', 'train_model'):
            logging.info(f"train models...")
            gmm.train(speaker_ids=speaker_ids)

        # prediction phase
        if config.getboolean('system', 'predict_speaker'):
            logging.info(f"predicting speaker...")
            gmm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)

        logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

    """
    GMM-UBM
    """
    gmm_ubm = GMMUBM()
    if config.getboolean("system", "gmm_ubm"):
        start_time = datetime.now()
        logging.info(f"Version GMM-UBM :{start_time}")
        logging.info(f"FEATURE_VERSION: {feature_type}")

        # training phase
        if config.getboolean('system', 'train_model'):
            logging.info(f"train models...")
            gmm_ubm.train(speaker_ids=speaker_ids)

        # prediction phase
        if config.getboolean('system', 'predict_speaker'):
            logging.info(f"predicting speaker...")
            gmm_ubm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)

        logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

    """
    SVM
    """
    svm = SVM()
    if config.getboolean("system", "svm"):
        start_time = datetime.now()
        logging.info(f"Version SVM :{start_time}")
        logging.info(f"FEATURE_VERSION: {feature_type}")

        # training phase
        if config.getboolean('system', 'train_model'):
            logging.info(f"train models...")
            svm.train(speaker_ids=speaker_ids)
        # prediction phase
        if config.getboolean('system', 'predict_speaker'):
            logging.info(f"predicting speaker...")
            svm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)

        logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")