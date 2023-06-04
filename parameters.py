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
    # 'MFCC',
    # 'SSC' ,
    # 'Chroma', 
    'MelSpectrogram',
    # 'Contrast',
    # 'Tonnetz',
    # 'F0',
]

datasets = [
    # 'custom_arabic',
    # 'emo-db',
    'tess_ravdess',
]

test_size = 0.2

verbose = 1

feature_emotion_X_Y_array_folder_path_name = 'memory'
