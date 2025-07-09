import pyttsx3
import numpy as np
import nltk
import time

class PyTTSX3Service:
    def __init__(self):
        self.engine = pyttsx3.init()

    def synthesize(self, text: str, **kwargs):
        self.engine.say(text)
        self.engine.runAndWait()
        return 0, np.array([]) # Return dummy values for compatibility

    def long_form_synthesize(self, text: str, **kwargs):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error during TTS synthesis: {e}")
            
        return 0, np.array([]) # Return dummy values for compatibility