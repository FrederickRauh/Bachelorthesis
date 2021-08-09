import logging

from datetime import datetime

import frontend.frontend
from backend.svm import SVM

from utils import util

svm = SVM()

if __name__ == '__main__':
    while(True):
        print("HEY")


    #############Config##############
    logging.basicConfig(level=0)
    # logger = logging.getLogger()
    # logger.disabled = True

    start_time = datetime.now()
    logging.info(f"Version SVM :{start_time}")
    logging.info(f"feature version: {config.FEATURE_TYPE}, mfcc count: {config.N_MFCC}")
    # preparation phase
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=IDS.SPEAKER_IDS, create_dataframe=True,
                                                       feature_type=config.FEATURE_TYPE)
    # training phase
    svm.train(speaker_ids=IDS.SPEAKER_IDS, feature_type=config.FEATURE_TYPE)
    # prediction phase
    svm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, feature_type=config.FEATURE_TYPE)

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")
