from Feature import Feature
import numpy as np
import pandas as pd
import tqdm
import os
import parameters as para
from MFCC import MFCC
from SSC import SSC
from Chroma import Chroma
from MelSpectrogram import MelSpectrogram
from Contrast import Contrast
from Tonnetz import Tonnetz
from F0 import F0


def get_name_datasets_features_emotions(prefix='', extension='.csv'):
    name = prefix
    for dataset in para.datasets:
        name = name + \
            f"_{dataset}"
    for feature in para.features:
        name = name + \
            f"_{feature}"
    for emotion in para.emotions:
        name = name + \
            f"_{emotion}"

    name = os.path.join(
        para.feature_emotion_X_Y_array_folder_path_name, f"{name}{extension}")
    return name


# def extract_feature_emotion_X_y_array():
#     feature_emotion_X_y_array_name = get_name_datasets_features_emotions()

#     if os.path.isfile(feature_emotion_X_y_array_name):
#         # if file already exists, just load then
#         if para.verbose:
#             print("[+] Feature file already exists, loading...")
#         feature_emotion_X_y_array = pd.read_csv(feature_emotion_X_y_array_name)
#         feature_emotion_X_y_array = feature_emotion_X_y_array.iloc[:,1:].values

#     else:
#         path_emotion_array = pd.DataFrame({'path': [], 'emotion': []})
#         for dataset_name in para.datasets:
#             file_name = f"{dataset_name}.csv"
#             path_emotion_array = pd.concat(
#                 (path_emotion_array, pd.read_csv(file_name)), sort=False)

#         feature_emotion_X_y_array = pd.DataFrame()
#         pbar = tqdm.tqdm(total=path_emotion_array['path'].size)
#         for path, emotion in zip(path_emotion_array['path'], path_emotion_array['emotion']):
#             if not (emotion in para.emotions):
#                 continue
#             feature_emotion_X_y_array = feature_emotion_X_y_array.append(pd.Series(np.append(Feature(path).extract_feature(), para.emotions.index(emotion))), ignore_index=True)
#             pbar.update(1)
#         pbar.close()
#         feature_emotion_X_y_array.to_csv(feature_emotion_X_y_array_name)
#         feature_emotion_X_y_array = feature_emotion_X_y_array.values

#     return {
#         "X": feature_emotion_X_y_array[:,:-1],
#         "y": np.concatenate(feature_emotion_X_y_array[:,-1:]),
#     }

def extract_feature_emotion_X_y_array():
    result = []

    if 'MFCC' in para.features:
        # Compute MFCC
        # X, y = MFCC.extract_feature_emotion_X_y_array()
        feature_emotion_X_Y_array = MFCC.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        # print(X)
        # exit()
        result.append(X)
    if 'SSC' in para.features:
        # Compute SSC
        # X, y = SSC.extract_feature_emotion_X_y_array()
        feature_emotion_X_Y_array = SSC.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
        # result.append(SSC(file_name=self.file_name, signal=self.signal,
        #               sample_rate=self.sample_rate).extract())
    if 'Chroma' in para.features:
        # Compute chroma feature
        # result.append(Chroma(file_name=self.file_name, signal=self.signal,
        #               stft=self.stft, sample_rate=self.sample_rate).extract())
        feature_emotion_X_Y_array = Chroma.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'MelSpectrogram' in para.features:
        # Compute MEL spectrogram feature
        # result.append(MelSpectrogram(file_name=self.file_name,
        #               signal=self.signal, sample_rate=self.sample_rate).extract())
        feature_emotion_X_Y_array = MelSpectrogram.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'Contrast' in para.features:
    #     # Compute spectral contrast feature
    #     result.append(Contrast(file_name=self.file_name, signal=self.signal,
    #                   sample_rate=self.sample_rate, stft=self.stft).extract())
        feature_emotion_X_Y_array = Contrast.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'Tonnetz' in para.features:
    #     # Compute tonnetz feature
    #     result.append(Tonnetz(file_name=self.file_name, signal=self.signal,
    #                   sample_rate=self.sample_rate, stft=self.stft).extract())
        feature_emotion_X_Y_array = Tonnetz.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    if 'F0' in para.features:
    #     # Compute F0 feature
    #     result.append(F0(file_name=self.file_name, signal=self.signal,
    #                   sample_rate=self.sample_rate).extract())
        feature_emotion_X_Y_array = F0.extract_feature_emotion_X_y_array()
        X = feature_emotion_X_Y_array['X']
        y = feature_emotion_X_Y_array['y']
        result.append(X)
    # print(result)
    result = np.concatenate(result,axis=1)
    print(result.shape)
    # exit()
    return {
        "X": result,
        "y": y,
    }


if __name__ == '__main__':

    extract_feature_emotion_X_y_array()
