# from IPython.display import IFrame
# IFrame('https://tonejs.github.io/examples/envelope.html', width=700, height=350)

import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd

x, sr = librosa.load('dataset/custom_arabic/neutral\soumayaC_neutral.wav')
# print(x.shape)
# print(sr)

# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(x, sr=sr)
# plt.xlabel("Time (s)")
# plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
ipd.Audio('dataset/custom_arabic/neutral\soumayaC_neutral.wav') # load a local WAV file