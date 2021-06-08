from backend import trainer
from backend.trainer import Trainer
from frontend import frontend
from utils import directoryManager as dm
from utils import grabber as gb

print("starting...")
name = 'id00001'
# timespan(sec), samplerate, number, name
# frontend.get_voice_input_stream(4, 44100, 1, name)


# frontend.process_features(name)
# trainer.get_features_from_data(name)

gb.create_overall_csv()


trainer = Trainer()
trainer.train_svm(name)
# print(feature_extraction)

