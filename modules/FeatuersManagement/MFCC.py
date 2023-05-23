from python_speech_features import mfcc
from scipy.stats import skew, kurtosis
import numpy as np
import librosa

# Mel Frequency Cepstral Coefficients


class MFCC:
    def __init__(self, sample_rate, signal, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate

    def extract(self):
        # (self.sig, self.rate) = librosa.load(self.file_name)
        mfcc_feat = np.array(mfcc(self.signal, self.sample_rate, numcep=40, nfilt=40, nfft=1024))

        result = np.concatenate((
            mfcc_feat.mean(axis=0),
            mfcc_feat.min(axis=0),
            mfcc_feat.max(axis=0),
            mfcc_feat.var(axis=0),
            np.median(mfcc_feat, axis=0),
            np.ptp(mfcc_feat, axis=0),
            skew(mfcc_feat, axis=0, bias=True),
            kurtosis(mfcc_feat, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(
            mfcc_feat, [75, 50, 25], axis=0))
        return result
