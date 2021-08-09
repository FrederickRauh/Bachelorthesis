import logging

import numpy as np

from datetime import datetime

import frontend.frontend
from backend.gmm import GMM

from config import CONFIG as config, IDS
from utils import util

gmm = GMM()

if __name__ == '__main__':
    #############Config##############
    logging.basicConfig(level=0)
    # logger = logging.getLogger()
    # logger.disabled = True

    start_time = datetime.now()
    logging.info(f"FEATURE_VERSION: {config.FEATURE_TYPE} Version SVM :{start_time}")
    # preparation phase
    frontend.frontend.feature_extraction_for_n_speaker(speaker_ids=IDS.SPEAKER_IDS, create_dataframe=True,
                                                       feature_type=config.FEATURE_TYPE)
    # training phase
    gmm.train(speaker_ids=IDS.SPEAKER_IDS, feature_type=config.FEATURE_TYPE)
    # prediction phase
    gmm.predict_n_speakers(speaker_ids=IDS.SPEAKER_IDS, feature_type=config.FEATURE_TYPE)

    logging.info(f"----------------------------------------------------------{util.get_duration(start_time)}")