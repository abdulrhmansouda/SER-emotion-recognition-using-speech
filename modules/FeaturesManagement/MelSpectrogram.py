from scipy.stats import skew, kurtosis
import librosa
import numpy as np
import pandas as pd
import tqdm
import os
import sys
sys.path.insert(0, os.getcwd())
import parameters as para
sys.path.insert(0, os.getcwd()+"\modules\FeaturesManagement")
import modules.FeaturesManagement.helper as featureHelper

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
    
    

    def extract_feature(file_name):
        # self.file_name = file_name
        signal, sample_rate = librosa.load(file_name)
        # stft = np.abs(librosa.stft(signal))
        # if 'Chroma' in para.features or 'Contrast' in para.features or 'Tonnetz' in para.features:
        #     self.stft = np.abs(librosa.stft(self.signal))

        result = []
        # if 'MFCC' in para.features:
        #     # Compute MFCC
        #     result.append(MFCC(file_name=file_name, signal=signal,sample_rate=sample_rate).extract())
        # if 'SSC' in para.features:
        #     # Compute SSC
        #     result.append(SSC(file_name=file_name, signal=signal,sample_rate=sample_rate).extract())
        # if 'Chroma' in para.features:
        #     # Compute chroma feature
        #     result.append(Chroma(file_name=file_name, signal=signal,stft=stft, sample_rate=sample_rate).extract())
        if 'MelSpectrogram' in para.features:
            # Compute MEL spectrogram feature
            result.append(MelSpectrogram(file_name=file_name, signal=signal,sample_rate=sample_rate).extract())
        # if 'Contrast' in para.features:
        #     # Compute spectral contrast feature
        #     result.append(Contrast(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate,stft=self.stft).extract())
        # if 'Tonnetz' in para.features:
        #     # Compute tonnetz feature
        #     result.append(Tonnetz(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate,stft=self.stft).extract())
        # if 'F0' in para.features:
        #     # Compute F0 feature
        #     result.append(F0(file_name=self.file_name, signal=self.signal,sample_rate=self.sample_rate).extract())

        return np.concatenate(result)


    def extract_feature_emotion_X_y_array(filter=True):
        feature_emotion_X_y_array_name = featureHelper.get_name_datasets_feature_emotions(feature='MelSpectrogram')
        # print(feature_emotion_X_y_array_name)
        # exit()
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
                feature_emotion_X_y_array = feature_emotion_X_y_array.append(pd.Series(np.append(MelSpectrogram.extract_feature(path), para.emotions.index(emotion))), ignore_index=True)
                pbar.update(1)
            pbar.close()
            feature_emotion_X_y_array.to_csv(feature_emotion_X_y_array_name)
            feature_emotion_X_y_array = feature_emotion_X_y_array.values
        X = feature_emotion_X_y_array[:,:-1]
        # if filter == True:
        #     sys.path.insert(0, os.getcwd()+"\modules\FeatureSelectionManagement")
        #     from  modules.FeatureSelectionManagement.CatBoostFeatureSelector import CatBoostFeatureSelector 
        #     X = CatBoostFeatureSelector.filter_features(X)
        return {
            "X": X,
            "y": np.concatenate(feature_emotion_X_y_array[:,-1:]),
        }
    
if __name__ == '__main__':
    MelSpectrogram.extract_feature_emotion_X_y_array()
    # extract_feature_emotion_X_y_array()