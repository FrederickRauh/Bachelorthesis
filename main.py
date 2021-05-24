from frontend import frontend

print("starting...")
frontend.getVoiceInput(5, 44100, 1, 'frederick')

print(frontend.extractFeatures("frederick", 1)[1:3,:])
