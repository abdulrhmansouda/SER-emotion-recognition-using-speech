import librosa
import numpy as np
from scipy.stats import skew, kurtosis


class Contrast:
    def __init__(self, sample_rate, signal, stft, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate
        self.stft = stft
    
    def extract(self):
        contrast = librosa.feature.spectral_contrast(S=self.stft, sr=self.sample_rate).T
        result = np.concatenate((
            contrast.mean(axis=0),
            contrast.min(axis=0),
            contrast.max(axis=0),
            contrast.var(axis=0),
            np.median(contrast, axis=0),
            np.ptp(contrast, axis=0),
            skew(contrast, axis=0, bias=True),
            kurtosis(contrast, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(contrast, [75, 50, 25],axis=0))
        return result