import numpy as np
import librosa


def load_audio(path, sr=16000):
    
    audio, _ = librosa.load(path, sr=sr)
    return audio
