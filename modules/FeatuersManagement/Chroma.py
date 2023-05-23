# import soundfile
import librosa
from scipy.stats import skew, kurtosis
import numpy as np


# Spectral Subband Centroids
class Chroma:
    def __init__(self, sample_rate, signal, stft, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate
        self.stft = stft

    def extract(self):
        chroma = librosa.feature.chroma_stft(S=self.stft, sr=self.sample_rate).T
        
        result = np.concatenate((
            chroma.mean(axis=0),
            chroma.min(axis=0),
            chroma.max(axis=0),
            chroma.var(axis=0),
            np.median(chroma, axis=0),
            np.ptp(chroma, axis=0),
            skew(chroma, axis=0, bias=True),
            kurtosis(chroma, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(chroma, [75, 50, 25], axis=0))
        return result
