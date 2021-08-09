import logging

import numpy as np

class CONFIG(object):
    ###############################
    #          CONFIG             #
    ###############################
    VERSION = 6
    LOCAL = False
    DATASET_PATH = 'E:' + '\\' + 'voxceleb' + '\\' + 'vox1_ba_wav'


    def overwrite_version(self, version):
        VERSION = version

    ###############################
    #          SYSTEM             #
    ###############################
    PROCESSES = 16
    LOGLEVEL = logging.INFO


    def overwrite_loglevel(self, loglevel):
        LOGLEVEL = loglevel

    ###############################
    #            GMM              #
    ###############################
    G_N_COMPONENTS = [16] # duo to https://appliedmachinelearning.blog/2017/11/14/spoken-speaker-identification-based-on-gaussian-mixture-models-python-implementation/ switched from 8 to 16
    G_MAX_ITER = [500]
    G_COVARIANCE_TYPE = ['diag']
    G_N_INIT = [3]


    def overwrite_g_n_components(self, g_n_components):
        G_N_COMPONENTS = g_n_components


    ###############################
    #        GMM-UBM              #
    ###############################
    UBM_N_COMPONENTS = [512]
    UBM_MAX_ITER = [500]
    UBM_COVARIANCE_TYPE = ['diag']
    UBM_N_INIT = [3]
    UBM_THRESHOLD = 0

    GMM_N_COMPONENTS = [16] # duo to https://appliedmachinelearning.blog/2017/11/14/spoken-speaker-identification-based-on-gaussian-mixture-models-python-implementation/ switched from 8 to 16
    GMM_MAX_ITER = [500]
    GMM_COVARIANCE_TYPE = ['diag']
    GMM_N_INIT = [3]
    THRESHOLD = 0

    ###############################
    #            SVM              #
    ###############################
    KERNELS = ['rbf']
    C = np.arange(0.1, 5.1, 0.1)
    C = [round(x, 2) for x in C]
    GAMMA = ['auto', 'scale']

    ###############################
    #          FEATURES           #
    ###############################
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

    APPENDENERGY = True  # PREV: TRUE
    CEPLIFTER = 0  # PREV: 22
    # which featureextractor to use librosa or psf
    # FEATURE_TYPE = 'psf'
    FEATURE_TYPE = 'librosa'
    FMAX = None
    FMIN = 0
    # HOP_LENGTH = 160  # PREV: 0.01
    NFFT = 2048  # 2048  # PREV: 0.025 AND 2048(DUO TO ERROR MESSAGE)
    N_MELS = 26  # 40  # PREV: 26
    N_MFCC = 20  # 40
    PREEMPH = 0.97  # 0.0  #0.97  # PREV: 0.07
    SAMPLE_RATE = 16000
    WINLEN = 0.025  # 0.064  # PREV N_FFT / SR WIN LENGTH OF 16-64MS
    WINSTEP = 0.01  # PREV: 0.036, HOP_LENGTH / SR (DEFAULT 0.01 (10MS))
    WINFUNC = lambda x: np.hamming(x)


    def overwrite_feature_type(self, feature_type):
        FEATURE_TYPE = feature_type


    def overwrite_n_mfcc(self, n_mfcc):
        N_MFCC = n_mfcc


    ###############################
    #       MODELCONFIG           #
    ###############################
    # --------------------- GridSearch ------------------------
    CV = 10
    REFIT = True
    VERBOSE = 2
    # n_jobs = -1 use all cpus, -2 use all but one
    N_JOBS = -2


    def overwrite_verbose(self, x):
        self.VERBOSE = x


    def overwrite_n_jobs(self, x):
        self.N_JOBS = x


from utils import directoryManager as dm


class IDS(object):
    ###############################
    #           IDS               #
    ###############################
    SPEAKER_IDS = dm.get_all_ids()
    FINISHED_IDS = ['id10001', 'id10002', 'id10003', 'id10004', 'id10005',
                        'id10006', 'id10007', 'id10008', 'id10009', 'id10010',
                        'id10011', 'id10012', 'id10013', 'id10014', 'id10015',
                        'id10016', 'id10017', 'id10018', 'id10019', 'id10020',

                        'id10021', 'id10022', 'id10023', 'id10024', 'id10025',
                        'id10026', 'id10027', 'id10028', 'id10029', 'id10030',
                        'id10031', 'id10032', 'id10033', 'id10034', 'id10035',
                        'id10036', 'id10037', 'id10038', 'id10039', 'id10040',
                        'id10041', 'id10042', 'id10043', 'id10044', 'id10045']


    def remove_finished_ids(self, ids=SPEAKER_IDS, finished_ids=FINISHED_IDS):
        for id in finished_ids:

            if ids.__contains__(id):
                ids.remove(id)
        self.speaker_ids = ids

