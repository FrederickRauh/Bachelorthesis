import json
import logging
import time
from configparser import ConfigParser
from datetime import datetime

from backend.gmm import GMM
from backend.gmm_ubm import GMMUBM
from backend.svm import SVM
from frontend import frontend
from utils import directoryManager as dm, trainingTestingManager as tt, util

if __name__ == '__main__':
    #############Config##############
    attack_config_file = rf'{dm.get_project_path()}/attack-config.ini'
    attackConfig = ConfigParser()
    attackConfig.read(attack_config_file)

    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-attack.log', level=attackConfig.getint('system', 'LOGLEVEL'))
    logger = logging.getLogger()
    logger.disabled = not attackConfig.getboolean('system', 'LOG')

    new_file_ids = json.loads(attackConfig.get('system', 'NEW_FILE_IDS'))

    if attackConfig.getboolean('system', 'MAKE_NEW_FILES'):
        for id in new_file_ids:
            time.sleep(10)
            frontend.get_voice_input_stream(4, 16000, attackConfig.getint('system', 'NEW_FILE_COUNT'), id, 'replay')

    speaker_ids = dm.get_all_ids()
    test_files, _ = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)

    test_files, extra_data_object = tt.get_attack_files_and_extra_data(speaker_ids=speaker_ids)

    version = config.get('system', 'VERSION')

    config.set('system', 'VERSION', 'dirty')
    with open('config.ini', "w") as f:
        config.write(f)

    if attackConfig.getboolean('system', 'PREDICT_SPEAKER'):
        if attackConfig.getboolean("system", "GMM"):
            start_time = datetime.now()
            logging.info(f"predicting speaker, gmm model...")
            gmm = GMM()
            gmm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)
            logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
        if attackConfig.getboolean("system", "GMM_UBM"):
            start_time = datetime.now()
            logging.info(f"predicting speaker, gmm-ubm model...")
            gmm_ubm = GMMUBM()
            gmm_ubm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)
            logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
        if attackConfig.getboolean("system", "SVM"):
            start_time = datetime.now()
            logging.info(f"predicting speaker, svm model...")
            svm = SVM()
            svm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)
            logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

    time.sleep(10)
    config.set('system', 'VERSION', version)
    with open('config.ini', "w") as f:
        config.write(f)