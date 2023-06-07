import librosa
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import tqdm
import os
import sys
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
import modules.FeaturesManagement.helper as featureHelper

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
    
    
    def extract_feature(file_name):
        signal, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(signal))

        result = []

        if 'Contrast' in para.features:
            # Compute spectral contrast feature
            result.append(Contrast(file_name=file_name, signal=signal,sample_rate=sample_rate,stft=stft).extract())


        return np.concatenate(result)


    def extract_feature_emotion_X_y_array():
        feature_emotion_X_y_array_name = featureHelper.get_name_datasets_feature_emotions(feature='Contrast')

        if os.path.isfile(feature_emotion_X_y_array_name):
            # if file already exists, just load then
            if para.verbose:
                print("[+] Feature file already exists, loading...")
            feature_emotion_X_y_array = pd.read_csv(feature_emotion_X_y_array_name)
            feature_emotion_X_y_array = feature_emotion_X_y_array.iloc[:,1:].values

        else:
            path_emotion_array = pd.DataFrame({'path': [], 'emotion': []})
            for dataset_name in para.datasets:
                file_name = f"{dataset_name}.csv"
                path_emotion_array = pd.concat(
                    (path_emotion_array, pd.read_csv(file_name)), sort=False)

            feature_emotion_X_y_array = pd.DataFrame()
            pbar = tqdm.tqdm(total=path_emotion_array['path'].size)
            for path, emotion in zip(path_emotion_array['path'], path_emotion_array['emotion']):
                if not (emotion in para.emotions):
                    continue
                feature_emotion_X_y_array = feature_emotion_X_y_array.append(pd.Series(np.append(Contrast.extract_feature(path), para.emotions.index(emotion))), ignore_index=True)
                pbar.update(1)
            pbar.close()
            feature_emotion_X_y_array.to_csv(feature_emotion_X_y_array_name)
            feature_emotion_X_y_array = feature_emotion_X_y_array.values
        X = feature_emotion_X_y_array[:,:-1]

        return {
            "X": X,
            "y": np.concatenate(feature_emotion_X_y_array[:,-1:]),
        }
    
if __name__ == '__main__':
    Contrast.extract_feature_emotion_X_y_array()