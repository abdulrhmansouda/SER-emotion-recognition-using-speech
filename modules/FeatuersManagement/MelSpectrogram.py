from scipy.stats import skew, kurtosis
import librosa
import numpy as np

class MelSpectrogram:
    def __init__(self, sample_rate, signal, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate

    def extract(self):
        mel_spectrogram = librosa.feature.melspectrogram(y=self.signal, sr=self.sample_rate).T

        result = np.concatenate((
            mel_spectrogram.mean(axis=0),
            mel_spectrogram.min(axis=0),
            mel_spectrogram.max(axis=0),
            mel_spectrogram.var(axis=0),
            np.median(mel_spectrogram, axis=0),
            np.ptp(mel_spectrogram, axis=0),
            skew(mel_spectrogram, axis=0, bias=True),
            kurtosis(mel_spectrogram, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(
            mel_spectrogram, [75, 50, 25], axis=0))
        return result