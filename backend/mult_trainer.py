import logging
import os

from backend.trainer import Trainer

from utils import util

from multiprocessing import Process, freeze_support

from utils import dataframeManager as dam
from utils import directoryManager as dm


class Mult_Trainer(object):

    def __init__(self):
        pass

    def train_multi(self):
        trainer = Trainer()

        process_count = os.cpu_count()
        if not process_count < len(speaker_ids):
            process_count = len(speaker_ids)
        speaker_ids_split = util.split_array_for_multiprocess(speaker_ids, process_count)
        logging.debug("splits: ", len(speaker_ids_split))
        processes = []
        for i in range(process_count):
            process = Process(target=trainer.train_multi_svm, args=[speaker_ids_split[i], dataframe])
            processes.append(process)
        for process in processes:
            logging.debug("starting process:", process.name)
            process.start()
        for process in processes:
            process.join()

if __name__ == '__main__':
    mult_trainer = Mult_Trainer()
    speaker_ids = dm.get_all_ids()
    dataframe_path = dm.get_data_path() + '\\' + 'psf-dataframe.json'
    dataframe = dam.load_dataframe_from_path(dataframe_path)
    mult_trainer.train_multi()