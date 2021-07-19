import numpy as np


class Features:
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

    N_MFCC = 23  # 40
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