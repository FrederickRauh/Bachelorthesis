Anaconda Environment: 
- conda create -n bachelorthesis python=3.9
- pip install tensorflow
- pip install scikit-learn
- pip install SpeechRecognition 
- pip install sounddevice
- pip install playsound
- pip install python_speech_features
- pip install librosa  
- pip install matplotlib
- pip install pandas

a simpler way is to use the environment file with:
- conda env create -f environment.yml

To change configuration (mfccs used, or loglevel in the software ) change in  ./config.ini:
 
- system:
  -
      - EXTRACT_FEATURES (boolean) 
      - TRAIN_MODEL (boolean)
      - PREDICT_SPEAKER (boolean)
      - TRAINING_FILES = 10
      - FEATURE_THRESHOLD = 0.15
      - PROCESSES = 8
      - LOGLEVEL = 20
      - LOG = True
      - VERSION = local
      - ADD_EXTRA = False
      - DATASET_PATH = /dataset

- classifier
  - 
      - gmm (boolean) // use the gmm system
      - gmm-ubm (boolean) // use the gmm-ubm system
      - svm (boolean) // use the svm system
      - // possible to use all   

- stage
  - 
      - extract_features (boolean) // extract features and save in json files.
      - train_model (boolean) // train models
      - predict_speaker (boolean) // predict


- training_testing
  -
      - training_files (float) // if < 1 take the percentage of files (0.5 == 50 % of files), otherwise take the amount defined
      - testing_files  (float) // similar to training_files, except for testing

- gmm:
  -
      - G_THRESHOLD (float)
      - G_N_COMPONENTS (list of ints)   
      - G_MAX_ITER (list of ints)
      - G_COVARIANCE_TYPE (list of string)
      - G_N_INIT (list of ints)

- gmm-ubm:
  -
      - UBM_N_COMPONENTS (float)
      - UBM_MAX_ITER (list of ints)   
      - UBM_COVARIANCE_TYPE (list of ints)
      - UBM_N_INIT (list of string)
      - UBM_THRESHOLD (list of ints)
      - GMM_N_COMPONENTS (float)
      - GMM_MAX_ITER (list of ints)   
      - GMM_COVARIANCE_TYPE (list of ints)
      - GMM_N_INIT (list of string)
      - GMM_THRESHOLD (list of ints)

- svm:
  -
      - KERNELS (list of strings)
      - C.upper (float)   
      - C.lower (float)
      - GAMMA (list of string)


- FEATURES:
  - 
      - FEATURE_TYPE (psf or librosa) 
      - APPENDENERGY (boolean)    
      - CEPLIFTER (int)
      - FMAX (None)
      - FMIN:  
      - NFFT: 
      - N_MELS
      - N_MFCC 
      - PREEMPH
      - SAMPLKERATE
      - WINLEN:
      - WINSTEP:
      - WINFUNC:

- MODELCONFIG:
  - 
      - CV (int)
      - Refit (boolean)  
      - VERBOSE: loglevel of gridsearch cv. 
      - N_JOBS: 
        - -1: use all available processes 
        - -2: use all but one
    