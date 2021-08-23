import logging
from configparser import ConfigParser
from datetime import datetime

import frontend.frontend
from backend.svm import SVM

from utils import directoryManager as dm, modelManager as m, util

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
    speaker_ids = list(reversed(dm.get_all_ids()))
    # finished_ids = ['id10050',
    #                 'id10049', 'id10048', 'id10047', 'id10046', 'id10045', 'id10044', 'id10043', 'id10042', 'id10041', 'id10040',
    #                 'id10039', 'id10038', 'id10037', 'id10036', 'id10035', 'id10034', 'id10033', 'id10032', 'id10031', 'id10030',
    #                 'id10029', 'id10028', 'id10027', 'id10026', 'id10025', 'id10024', 'id10023', 'id10022', 'id10021']

    # for id in finished_ids:
    #     if speaker_ids.__contains__(id):
    #         speaker_ids.remove(id)
    # speaker_ids = speaker_ids


    logging.basicConfig(filename=rf'{dm.get_project_path()}/info-svm.log', level=config.getint('system', 'LOGLEVEL'))
    logger = logging.getLogger()
    logger.disabled = not config.getboolean('system', 'LOG')
    if config.getboolean('system', 'LOG'):
        print("container running. logs can be found in info-{model_type}.log")

    start_time = datetime.now()
    logging.info(f"Version SVM :{start_time}")
    logging.info(f"FEATURE_VERSION: {feature_type}")
    # preparation phase
    if config.getboolean('system', 'EXTRACT_FEATURES'):
        logging.info(f"extracting features...")
        frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=speaker_ids, create_dataframe=True)
    # training phase
    if config.getboolean('system', 'TRAIN_MODEL'):
        logging.info(f"retraining models...")
        retrain_ids = []
        t = "svm_" + config.getfloat("system", "FEATURE_THRESHOLD")
        for speaker_id in speaker_ids:
            params = m.get_model_best_estimator_(speaker_id, t)
            if params['C'] == 1.0:
                retrain_ids.append(speaker_id)
        print(retrain_ids)
        svm.train(speaker_ids=retrain_ids)
    # prediction phase
    if config.getboolean('system', 'PREDICT_SPEAKER'):
        logging.info(f"predicting speaker...")
        svm.predict_n_speakers(speaker_ids=speaker_ids)

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")