import pandas as pd
emotions = [
    'angry',
    'calm',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'ps',
    'sad',

    # 'boredom'
]
one_hot_encode = pd.get_dummies(emotions)

features = [
    'MFCC',
    # 'SSC' ,
    # 'Chroma',
    # 'MelSpectrogram',
    # 'Contrast ',
    # 'Tonnetz',
    # 'F0',
]

datasets = [
    # 'custom_arabic',
    # 'emo-db',
    'tess_ravdess',
]

train_size_ratio = 0.7

verbose = 1

# emotions_Y_decode = {
#     'neutral': [1, 0, 0, 0, 0, 0, 0, 0, 0],
#     'happy': [0, 1, 0, 0, 0, 0, 0, 0, 0],
#     'sad': [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     'calm': [0, 0, 0, 1, 0, 0, 0, 0, 0],
#     'angry': [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     'fear': [0, 0, 0, 0, 0, 1, 0, 0, 0],
#     'disgust': [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     'ps': [0, 0, 0, 0, 0, 0, 0, 1, 0],
#     'boredom': [0, 0, 0, 0, 0, 0, 0, 0, 1],
# }
# emotions_Y_decode = {
#     'neutral': 1,
#     'happy': 2,
#     'sad': 3,
#     'calm': 4,
#     'angry': 5,
#     'fear': 6,
#     'disgust': 8,
#     'ps': 8,
#     'boredom': 9,
# }

feature_emotion_X_Y_array_folder_path_name = 'memory'
