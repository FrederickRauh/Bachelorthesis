import configparser
import json
import logging
from configparser import ConfigParser
from datetime import datetime

import frontend.frontend
from backend.gmm import GMM

from utils import directoryManager as dm, trainingTestingManager as tt, util

"""
GMM
"""
gmm = GMM()
if __name__ == '__main__':
    #############Config##############
    file = rf'{dm.get_project_path()}/config.ini'
    config = ConfigParser()
    config.read(file)

    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-gmm.log', level=config.getint('system', 'loglevel'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'log')
    if config.getboolean('system', 'log'):
        print("container running. logs can be found in info-{model_type}.log")

    feature_type = config.get('features', 'feature_type')
    speaker_ids = dm.get_all_ids()

    # preparation phase
    if config.getboolean('stage', 'extract_features'):
        logging.info(f"extracting features...")
        frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=speaker_ids, create_dataframe=True)

    try:
        ids = json.loads(config.get("system", "ids"))
        if not ids == []:
            ids.reverse()
            speaker_ids = ids
            logging.info(f"ids to process: \n {speaker_ids}")
    except configparser.NoOptionError:
        logging.info(f"No ids specified, using all")
    try:
        finished_ids = json.loads(config.get("system", "finished_ids"))
        if not finished_ids == []:
            for id in finished_ids:
                if speaker_ids.__contains__(id):
                    speaker_ids.remove(id)
            speaker_ids = speaker_ids
    except configparser.NoOptionError:
        logging.info(f"No finished_ids specified, removing none")

    # training_phase
    start_time = datetime.now()
    logging.info(f"Starting version GMM :{start_time}")

    # training phase
    if config.getboolean('stage', 'train_model'):
        start_time = datetime.now()
        logging.info(f"started training models...")
        gmm.train(speaker_ids=speaker_ids)
        logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")

    # prediction phase
    test_files = []
    if config.getboolean('stage', 'predict_speaker'):
        start_time = datetime.now()
        test_files, extra_data_object = tt.get_test_files_and_extra_data(speaker_ids=speaker_ids)
        logging.info(f"loaded {len(test_files)} testing files, time spent: {util.get_duration(start_time)}")
        logging.info(f"predicting speaker...")
        gmm.predict_n_speakers(speaker_ids=speaker_ids, test_files=test_files,
                               extra_data_object=extra_data_object)
        logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
