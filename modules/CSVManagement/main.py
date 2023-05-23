# import os
import pandas as pd
import glob


def write_csv(emotions, dataset_name, train_size_ratio, verbose=0):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for emotion in emotions:
        target = {"path": [], "emotion": []}
        for file in glob.glob(f"dataset/{dataset_name}/{emotion}/*.wav"):
            target['emotion'].append(emotion)
            target['path'].append(file)

        n_samples = len(target['path'])
        if verbose:
            print(f"[{dataset_name}] Total files to write:({emotion}) ", n_samples)

        # dividing training/testing sets
        test_size = int((1-train_size_ratio) * n_samples)
        train_size = n_samples - test_size

        if verbose:
            print(f"[{dataset_name}] Training samples:", train_size)
            print(f"[{dataset_name}] Testing samples:", test_size)
        X_train.extend(target['path'][:train_size])
        y_train.extend(target['emotion'][:train_size])
        X_test.extend(target['path'][train_size:])
        y_test.extend(target['emotion'][train_size:])

    train_file_name = f"train_{dataset_name}.csv"
    test_file_name = f"test_{dataset_name}.csv"
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_file_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_file_name)


# write_csv()
