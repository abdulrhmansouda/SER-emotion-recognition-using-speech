# # from Feature import Feature
# import numpy as np
# import pandas as pd
# import tqdm
# import os
# import sys
# sys.path.insert(0, os.getcwd())
# import parameters as para

# def get_name_datasets_features_emotions(prefix='',extension='.csv'):
#     name = prefix
#     for dataset in para.datasets:
#         name = name + \
#             f"_{dataset}"
#     for feature in para.features:
#         name = name + \
#             f"_{feature}"
#     for emotion in para.emotions:
#         name = name + \
#             f"_{emotion}"

#     name = os.path.join(
#         para.feature_emotion_X_Y_array_folder_path_name, f"{name}{extension}")
#     return name

# def get_feature_emotion_X_Y_array_for(train_or_test):
#     feature_X_array_name = get_name_datasets_features_emotions(prefix=f"{train_or_test}_X_")
#     emotion_Y_array_name = get_name_datasets_features_emotions(prefix=f"{train_or_test}_Y_")

#     if os.path.isfile(feature_X_array_name):
#         # if file already exists, just load then
#         if para.verbose:
#             print("[+] Feature file already exists, loading...")
#         X_features = pd.read_csv(feature_X_array_name)
#         Y_emotions = pd.read_csv(emotion_Y_array_name)
        
#     else:
#         raise 'error'

#     return {
#         "X": X_features,
#         "Y": Y_emotions,
#     }
