**Setup**:
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

or to create a docker container using the docker-compose.

_**structure**_:
The data of the speakers and the models have to be in file named data, this can however be changed within the compose.yml.
Inside of the data folder should be a structure following (an example can be seen on the provided usbstick):
- wav
  - idOfSpeaker
    - folder_containing files
      - files (.wav)

      

_**Configuration**_
To change configuration (mfccs used, or loglevel in the software ) change in  ./config.ini:
 
- system:

  - dataset_path = path to the files
  - processes = Processes used(useful for multiprocessing)
  - loglevel (default 20)
  - log (default true)

- classifier
  - gmm (boolean) // use the gmm system
  - gmm-ubm (boolean) // use the gmm-ubm system
  - svm (boolean) // use the svm system
        - // possible to use all   

- stage
  - extract_features (boolean) // extract features and save in json files.
  - train_model (boolean) // train models
  - predict_speaker (boolean) // predict


- training_testing
  -
      - training_files (float) // indicate how long the trainings data should be ! gets overwritten by training_length
      - training_lengt: actuall does what training_files should do.
      - testing_files  (float) // indicate how many files should be selected per speaker for testing. The last files per speaker are selected, to avoid training files

- result:
  - 
      - version: name added before the result files
      - add_extra: add additional info to result.json, (testing file names and amount of files etc.)
      - create_single_results: create a result.json for each tested model.  

- gmm:
  -
      - THRESHOLD (float): percentage of vectors that have to be positive for a file to be accepted.
      - G_N_COMPONENTS (list of ints)   
      - G_MAX_ITER (list of ints)
      - G_COVARIANCE_TYPE (list of string)
      - G_N_INIT (list of ints)

- gmm-ubm:
  -
      - threshold = (default of 0.8 to avoid a natural offset) 
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
      - svm_threshold: percentage of vectors that have to be positive for a file to be accepted.
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
      - CV (int) number of crossvalidation splits
      - Refit (boolean)  
      - VERBOSE: loglevel of gridsearch cv. 
      - N_JOBS: 
        - -1: use all available processes 
        - -2: use all but one
    