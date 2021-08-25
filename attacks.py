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

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-attack.log', level=attackConfig.getint('system', 'loglevel'))
    logger = logging.getLogger()
    logger.disabled = not attackConfig.getboolean('system', 'log')

    if config.getboolean('system', 'log'):
        print("container running. logs can be found in info-{model_type}.log")

    if attackConfig.getboolean('system', 'make_new_files'):
        print("STARTING")
        new_file_ids = json.loads(attackConfig.get('system', 'new_file_ids'))
        for id in new_file_ids:
            print(f"switch")
            time.sleep(5)
            print(f"Starting for:{id} in 5 seconds")
            time.sleep(5)
            frontend.get_voice_input_stream(4, 16000, attackConfig.getint('system', 'new_file_count'), id, 'replay-iphone')

    speaker_ids = dm.get_all_ids()
    test_files, _ = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)

    test_files, extra_data_object = tt.get_attack_files_and_extra_data(speaker_ids=speaker_ids)

    prev_version = config.get('result', 'version')

    config.set("result", 'version', 'dirty')
    with open("config.ini", "w") as f:
        config.write(f)
    time.sleep(60)

    if attackConfig.getboolean('system', 'predict_speaker'):
        if attackConfig.getboolean("system", "gmm"):
            start_time = datetime.now()
            logging.info(f"Version GMM :{start_time}")
            logging.info(f"predicting speaker, gmm model...")
            gmm = GMM()
            gmm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)
            logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
        if attackConfig.getboolean("system", "gmm_ubm"):
            start_time = datetime.now()
            logging.info(f"Version GMM-UBM :{start_time}")
            logging.info(f"predicting speaker, gmm-ubm model...")
            gmm_ubm = GMMUBM()
            gmm_ubm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)
            logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
        if attackConfig.getboolean("system", "svm"):
            start_time = datetime.now()
            logging.info(f"Version SVM :{start_time}")
            logging.info(f"predicting speaker, svm model...")
            svm = SVM()
            svm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files, extra_data_object=extra_data_object)
            logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

    time.sleep(10)
    if prev_version == 'dirty':
        prev_version = 'clean'
    config.set('result', 'version', prev_version)
    with open('config.ini', "w") as f:
        config.write(f)