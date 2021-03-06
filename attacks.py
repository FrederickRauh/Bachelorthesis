import configparser
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
    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)

    prev_version = config.get('result', 'version')

    #############Config##############
    attack_config_file = rf'{dm.get_project_path()}/attack-config.ini'
    attackConfig = ConfigParser()
    attackConfig.read(attack_config_file)

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-attack.log', level=attackConfig.getint('system', 'loglevel'))
    logger = logging.getLogger()
    logger.disabled = not attackConfig.getboolean('system', 'log')

    if config.getboolean('system', 'log'):
        print("container running. logs can be found in info-attack.log")

    if attackConfig.getboolean('stage', 'make_new_files'):
        print("STARTING")
        new_file_ids = json.loads(attackConfig.get('testing_attacking', 'new_file_ids'))
        for id in new_file_ids:
            print(f"switch")
            time.sleep(5)
            print(f"Starting for:{id} in 5 seconds")
            time.sleep(5)
            frontend.get_voice_input_stream(40, 16000, attackConfig.getint('testing_attacking', 'new_file_count'), id, 'replay-bluetooth-speaker')

    speaker_ids = dm.get_all_ids()
    try:
        ids = json.loads(attackConfig.get("system", "ids"))
        if not ids == []:
            speaker_ids = ids
            logging.info(f"ids to process: \n {speaker_ids}")
    except configparser.NoOptionError:
        logging.info(f"No ids specified, using all")
    test_files, _ = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)

    attack_type = attackConfig.get("testing_attacking", "attack_type")
    test_files, extra_data_object = tt.get_attack_files_and_extra_data(speaker_ids=speaker_ids, replay_type=attack_type)

    # lengths = [60, 120, 180, 240, 300]
    lengths = [28, 60, 120]
    for length in lengths:
        if attackConfig.getboolean('stage', 'predict_speaker'):
            if attackConfig.getboolean("classifier", "gmm"):
                start_time = datetime.now()
                logging.info(f"Version GMM :{start_time}")
                logging.info(f"predicting speaker, gmm model, with {length}...")
                gmm = GMM()
                gmm.predict_n_speakers(speaker_ids=speaker_ids,
                                       test_files=test_files,
                                       extra=length,
                                       attack_type=attack_type)
                logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
            if attackConfig.getboolean("classifier", "gmm_ubm"):
                start_time = datetime.now()
                logging.info(f"Version GMM-UBM :{start_time}")
                logging.info(f"predicting speaker, gmm-ubm model...")
                gmm_ubm = GMMUBM()
                gmm_ubm.predict_n_speakers(speaker_ids=speaker_ids,
                                           test_files=test_files,
                                           extra=length,
                                           attack_type=attack_type)
                logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
            if attackConfig.getboolean("classifier", "svm"):
                start_time = datetime.now()
                logging.info(f"Version SVM :{start_time}")
                logging.info(f"predicting speaker, svm model...")
                svm = SVM()
                svm.predict_n_speakers(speaker_ids=speaker_ids,
                                       test_files=test_files,
                                       extra=length,
                                       attack_type=attack_type)
                logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
