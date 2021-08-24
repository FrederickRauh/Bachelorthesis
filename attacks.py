import logging
from configparser import ConfigParser
from datetime import datetime

from backend.gmm import GMM
from backend.gmm_ubm import GMMUBM
from backend.svm import SVM
from frontend import frontend
from utils import directoryManager as dm, trainingTestingManager as tt, util

if __name__ == '__main__':
    #############Config##############
    file = rf'{dm.get_project_path()}/attack-config.ini'
    config = ConfigParser()
    config.read(file)

    logging.basicConfig(filename=rf'{dm.get_project_path()}/attack-info.log', level=config.getint('system', 'LOGLEVEL'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'LOG')

    speaker_id = 'id99990'

    if config.getboolean('system', 'MAKE_NEW_FILES'):
        frontend.get_voice_input_stream(4, 16000, config.getint('system', 'NEW_FILE_COUNT'), speaker_id + '-replay')

    speaker_ids = dm.get_all_ids()
    replay_speaker_ids = dm.get_all_replay_ids()
    test_files, _ = tt.get_test_files_and_extra_data(speaker_ids=replay_speaker_ids)


    if config.getboolean('system', 'PREDICT_SPEAKER'):
        if config.getboolean("system", "GMM"):
            logging.info(f"predicting speaker...")
            gmm = GMM()
            gmm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files)
        if config.getboolean("system", "GMM_UBM"):
            logging.info(f"predicting speaker...")
            gmm_ubm = GMMUBM()
            gmm_ubm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files)
        if config.getboolean("system", "SVM"):
            logging.info(f"predicting speaker...")
            svm = SVM()
            svm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files)