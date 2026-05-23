import pyttsx3

engine = pyttsx3.init()
engine.save_to_file('Hello, this is a test of your speaker. If you hear this, your setup works!', 'voice_test.wav')
engine.runAndWait()
