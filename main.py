from frontend import frontend

print("starting...")
frontend.get_voice_input_stream(5, 44100, 1, 'frederick')
# frontend.getVoiceInput(5, 44100, 1, 'frederick')

#
feature_extraction = frontend.extractFeatures("frederick", 1)

print(feature_extraction)
