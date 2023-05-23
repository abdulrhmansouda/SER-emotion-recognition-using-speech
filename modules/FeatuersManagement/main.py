from Feature import Feature
import numpy as np
import pandas as pd
import tqdm
import os
import parameters as para

def get_name_datasets_features_emotions(prefix='',extension='.csv'):
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

def get_feature_emotion_X_Y_array_for(train_or_test):
    feature_X_array_name = get_name_datasets_features_emotions(prefix=f"{train_or_test}_X_")
    emotion_Y_array_name = get_name_datasets_features_emotions(prefix=f"{train_or_test}_Y_")

    if os.path.isfile(feature_X_array_name):
        # if file already exists, just load then
        if para.verbose:
            print("[+] Feature file already exists, loading...")
        X_features = pd.read_csv(feature_X_array_name)
        Y_emotions = pd.read_csv(emotion_Y_array_name)
        
    else:
        path_emotion_speeches = pd.DataFrame({'path': [], 'emotion': []})
        for dataset_name in para.datasets:
            file_name = f"{train_or_test}_{dataset_name}.csv"
            path_emotion_speeches = pd.concat(
                (path_emotion_speeches, pd.read_csv(file_name)), sort=False)

        X_features = pd.DataFrame()
        Y_emotions = pd.DataFrame()
        pbar = tqdm.tqdm(total=path_emotion_speeches['path'].size)
        for path, emotion in zip(path_emotion_speeches['path'], path_emotion_speeches['emotion']):
            if  not (emotion in para.emotions):
                continue 
            # print(Feature(path).extract_feature())
            # exit()
            X_features = X_features.append(Feature(path).extract_feature(),ignore_index=True)
            Y_emotions = Y_emotions.append(para.one_hot_encode[emotion],ignore_index=True)
            pbar.update(1)
        pbar.close()
        X_features = X_features.loc[:, ~(X_features.isna().any() | np.isinf(X_features).any() )]
        X_features.to_csv(feature_X_array_name)
        Y_emotions.to_csv(emotion_Y_array_name)

    return {
        "X": X_features,
        "Y": Y_emotions,
    }

# def get_feature_emotion_X_Y_array_for_train_and_test():
#     train_feature_X_array_name = get_name_datasets_features_emotions(prefix=f"train_X_")
#     train_emotion_Y_array_name = get_name_datasets_features_emotions(prefix=f"train_Y_")
#     test_feature_X_array_name = get_name_datasets_features_emotions(prefix=f"test_X_")
#     test_emotion_Y_array_name = get_name_datasets_features_emotions(prefix=f"test_Y_")

#     if os.path.isfile(train_feature_X_array_name):
#         # if file already exists, just load then
#         if para.verbose:
#             print("[+] Feature file already exists, loading...")
#         train_X_features = pd.read_csv(train_feature_X_array_name)
#         train_Y_emotions = pd.read_csv(train_emotion_Y_array_name)
#         test_X_features = pd.read_csv(test_feature_X_array_name)
#         test_Y_emotions = pd.read_csv(test_emotion_Y_array_name)
        
#     else:
#         train_path_emotion_speeches = pd.DataFrame({'path': [], 'emotion': []})
#         test_path_emotion_speeches = pd.DataFrame({'path': [], 'emotion': []})
#         for dataset_name in para.datasets:
#             train_file_name = f"train_{dataset_name}.csv"
#             test_file_name = f"test_{dataset_name}.csv"

#             train_path_emotion_speeches = pd.concat((train_path_emotion_speeches, pd.read_csv(train_file_name)), sort=False)
#             test_path_emotion_speeches = pd.concat((test_path_emotion_speeches, pd.read_csv(test_file_name)), sort=False)

#         X_features = pd.DataFrame()
#         Y_emotions = pd.DataFrame()
#         pbar = tqdm.tqdm(total=path_emotion_speeches['path'].size)
#         for path, emotion in zip(path_emotion_speeches['path'], path_emotion_speeches['emotion']):
#             # print(Feature(path).extract_feature())
#             # exit()
#             X_features = X_features.append(Feature(path).extract_feature(),ignore_index=True)
#             Y_emotions = Y_emotions.append(para.one_hot_encode[emotion],ignore_index=True)
#             pbar.update(1)
#         pbar.close()
#         X_features = X_features.loc[:, ~(X_features.isna().any() | np.isinf(X_features).any() )]
#         X_features.to_csv(feature_X_array_name)
#         Y_emotions.to_csv(emotion_Y_array_name)

#     return {
#         "X": X_features,
#         "Y": Y_emotions,
#     }


if __name__ == '__main__':

    get_feature_emotion_X_Y_array_for('train')
    get_feature_emotion_X_Y_array_for('test')

# print(feature.extract_feature('dataset/emo-db/neutral\\14b01Na.wav').shape)
# np.save('dataset/abd',feature.extract_feature('dataset/custom_arabic/neutral\soumayaC_neutral.wav'))
