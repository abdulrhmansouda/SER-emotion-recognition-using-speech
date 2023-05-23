import soundfile
import librosa
import numpy as np
from scipy.stats import skew, kurtosis



# Spectral Subband Centroids
class Tonnetz:
    def __init__(self, sample_rate, signal, stft, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate
        self.stft = stft
    
    def extract(self):
        chroma = librosa.feature.chroma_stft(S=self.stft, sr=self.sample_rate).T
        # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(self.signal), sr=self.sample_rate).T
        tonnetz = librosa.feature.tonnetz(chroma=chroma)
        result = np.concatenate((
            tonnetz.mean(axis=0),
            tonnetz.min(axis=0),
            tonnetz.max(axis=0),
            tonnetz.var(axis=0),
            np.median(tonnetz, axis=0),
            np.ptp(tonnetz, axis=0),
            skew(tonnetz, axis=0, bias=True),
            kurtosis(tonnetz, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(tonnetz, [75, 50, 25],axis=0))
        return result