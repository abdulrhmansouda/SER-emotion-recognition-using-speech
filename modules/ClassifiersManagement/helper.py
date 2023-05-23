import parameters as para
import os


def get_special_name(folder_name=para.feature_emotion_X_Y_array_folder_path_name, prefix='', extension=''):
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

    name = os.path.join(folder_name, f"{name}{extension}")
    return name
