from python_speech_features import ssc
from scipy.stats import skew, kurtosis
import numpy as np
import librosa

class SSC:
    def __init__(self, sample_rate, signal, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate
    
    def extract(self):

        # (self.sig,self.rate) = librosa.load(self.file_name)
        ssc_var = ssc(self.signal,self.sample_rate,nfft=1024)
        result = np.concatenate((
            ssc_var.mean(axis=0),
            ssc_var.min(axis=0),
            ssc_var.max(axis=0),
            ssc_var.var(axis=0),
            np.median(ssc_var, axis=0),
            np.ptp(ssc_var, axis=0),
            skew(ssc_var, axis=0, bias=True),
            kurtosis(ssc_var, axis=0, bias=True),
        ))
        result = np.append(result, np.percentile(ssc_var, [75, 50, 25],axis=0))
        return result