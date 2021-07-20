import numpy as np


class CONFIG:
    VERSION = 1

    def overwrite_version(self, x):
        self.VERSION = x


class DEBUG:
    LOGLEVEL = 1

    def overwrite_loglevel(self, x):
        self.LOGLEVEL = x


class FEATURES:
    # Parameters:
    # signal – the audio signal from which to compute features. Should be an N*1 array
    # samplerate – the samplerate of the signal we are working with.
    # winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    # winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    # numcep – the number of cepstrum to return, default 13
    # nfilt – the number of filters in the filterbank, default 26.
    # nfft – the FFT size. Default is 512.
    # lowfreq – lowest band edge of mel filters. In Hz, default is 0.
    # highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
    # preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    # ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    # appendEnergy – if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    # winfunc – the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    # Returns:
    # A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    SAMPLE_RATE = 16000
    N_MFCC = 24  # 40
    N_MELS = 26  # 40  # PREV: 26
    NFFT = 2048  # 2048  # PREV: 0.025 AND 2048(DUO TO ERROR MESSAGE)
    # HOP_LENGTH = 160  # PREV: 0.01
    FMIN = 0
    FMAX = None
    PREEMPH = 0.97  # 0.0  #0.97  # PREV: 0.07
    CEPLIFTER = 0  # PREV: 22
    APPENDENERGY = True  # PREV: TRUE
    WINLEN = 0.05  # 0.064  # PREV N_FFT / SR WIN LENGTH OF 16-64MS
    WINSTEP = 0.01  # PREV: 0.036, HOP_LENGTH / SR (DEFAULT 0.01 (10MS))
    WINFUNC = lambda x: np.hamming(x)

    def overwrite_n_mfcc(self, x):
        self.N_MFCC = x


from utils import directoryManager as dm


class IDS:
    # speaker_ids = dm.get_all_ids()

    # speaker_ids = [speaker_ids[0]]
    # speaker_ids = util.remove_finished_ids(speaker_ids, finished_ids)
    # speaker_ids = finished_ids

    # speaker_id = 'id00001'

    speaker_ids = dm.get_all_ids()
    finished_ids = ['id10001', 'id10002', 'id10003', 'id10004', 'id10005',
                    'id10006', 'id10007', 'id10008', 'id10009', 'id10010',
                    'id10011', 'id10012', 'id10013', 'id10014', 'id10015',
                    'id10016', 'id10017', 'id10018', 'id10019', 'id10020',
                    #                 ]
                    #                 ,
                    'id10021', 'id10022', 'id10023', 'id10024', 'id10025',
                    'id10026', 'id10027', 'id10028', 'id10029', 'id10030',
                    'id10031', 'id10032', 'id10033', 'id10034', 'id10035',
                    'id10036', 'id10037', 'id10038', 'id10039', 'id10040',
                    'id10041', 'id10042', 'id10043', 'id10044', 'id10045']

    def remove_finished_ids(self, ids=speaker_ids, finished_ids=finished_ids):
        for id in finished_ids:
            if ids.__contains__(id):
                ids.remove(id)
        self.speaker_ids = ids


from sklearn.model_selection import KFold


class MODELCONFIG:
    # --------------------- GridSearch ------------------------
    cv = KFold(n_splits=4)
    # helpful with large datasets to keep an overview
    # n_jobs = -1 use all cpus, -2 use all but one
    VERBOSE = 0
    N_JOBS = -2
    if dm.is_large_data_set():
        VERBOSE = 0
        N_JOBS = -1

    def overwrite_verbose(self, x):
        self.VERBOSE = x

    def overwrite_n_jobs(self, x):
        self.N_JOBS = x