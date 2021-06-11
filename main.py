from backend import trainer
from backend.trainer import Trainer
from backend import predictor
from frontend import frontend
from utils import directoryManager as dm
from utils import csvManager as cm
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


print("starting...")
speaker_id = 'id10002'

# timespan(sec), samplerate, number, speaker_id
# frontend.get_voice_input_stream(4, 44100, 1, speaker_id)
#
#
speaker_ids = dm.get_all_ids()
print("IDs : ", speaker_ids)

# frontend.process_features(speaker_id)
# #
# cm.create_overall_csv()

trainer = Trainer()
trainer.train_svm(speaker_id)

# predictor = Predictor()
# parent_path = dm.get_parent_path(speaker_id)
# file_path = parent_path + ''
# features = frontend.extract_mfcc_from_file()
# score = predictor.predict_svm(speaker_id, )


# print(feature_extraction)

