Anaconda Environment: 
- conda create -n bachelorthesis python=3.9
- pip install tensorflow
- pip install scikit-learn
- pip install SpeechRecognition 
- pip install sounddevice
- pip install playsound
- pip install python_speech_features
- pip install matplotlib
- pip install pandas



[comment]: <> (- pip install pyannote.db.voxceleb)


To change configuration (mfccs used, or loglevel in the software ) change in  util/config.py:
- SYSTEM
    - PROCESSES: Amount of processes used in prediction and feature extraction (default 16)
    - LOGLEVEL: level depends on python logging.level
- CONFIG:
    - VERSION: used when making predictions, describes the folder where results are put down ./results/{dataset}/{model_version}/{version}/result.json
    - LOCAL: wether or not to use a local dataset. Audio file path: ./local/data/wav/ and  test audio file data in ./local/test/
    - DATASETPATH: path to dataset if not using a local dataset.
    
- FEATURES:
    - SAMPLKERATE 
    - N:_MFCC 
    - N_MELS
    - NFFT:
    - FMIN:
    - FMAX:
    - PREEMPH:
    - CEPLIFTER:
    - APPENDENERGY:
    - WINLEN:
    - WINSTEP:
    - WINFUNC:
    
- IDS:
    - SPEAKER_IDS: speaker ids that should be used in the training / prediction / feature extraction phase
    - FINISHED_IDS: in case you wish to break a phase down into multiple steps, add ids to finished_ids ([]) to skip these.
    
- MODELCONFIG:
    - VERBOSE: loglevel of gridsearch cv. 
    - N_JOBS: 
      - -1: use all available processes 
      - -2: use all but one
    