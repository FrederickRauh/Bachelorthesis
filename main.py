from backend import trainer
from backend.trainer import Trainer
from frontend import frontend
from utils import directoryManager as dm
from utils import csvManager as cm

print("starting...")
speaker_id = 'id00001'
# timespan(sec), samplerate, number, speaker_id
# frontend.get_voice_input_stream(4, 44100, 1, speaker_id)
#
#
frontend.process_features(speaker_id)
#
# cm.create_overall_csv()
#

trainer = Trainer()
trainer.train_svm(speaker_id)
# print(feature_extraction)

