from scipy.stats import skew, kurtosis
import librosa
import numpy as np

class F0:
    def __init__(self, sample_rate, signal, file_name='') -> None:
        self.file_name = file_name
        self.signal = signal
        self.sample_rate = sample_rate
    

    def extract(self):
        self.f0 = librosa.yin(self.signal,
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'))
        result = [
            self.f0.mean(axis=0),
            self.f0.min(axis=0),
            self.f0.max(axis=0),
            self.f0.var(axis=0),
            np.median(self.f0, axis=0),
            np.ptp(self.f0, axis=0),
            skew(self.f0, axis=0, bias=True),
            kurtosis(self.f0, axis=0, bias=True),
        ]
        result = np.append(result, np.percentile(self.f0, [75, 50, 25],axis=0))
        return result

    
    # def draw(self):
    #     import matplotlib.pyplot as plt
    #     times = librosa.times_like(self.f0)
    #     D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
    #     fig, ax = plt.subplots()
    #     img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    #     ax.set(title='pYIN fundamental frequency estimation')
    #     fig.colorbar(img, ax=ax, format="%+2.f dB")
    #     ax.plot(times, self.f0, label='f0', color='cyan', linewidth=3)
    #     ax.legend(loc='upper right')
