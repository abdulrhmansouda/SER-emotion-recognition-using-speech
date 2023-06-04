import pandas as pd
import glob
import sys
import os
sys.path.insert(0, os.getcwd())
import parameters as para


def write_csv(emotions, dataset_name, verbose=0):
    X = []
    y = []
    # X_test = []
    # y_test = []
    for emotion in emotions:
        target = {"path": [], "emotion": []}
        for file in glob.glob(f"dataset/{dataset_name}/{emotion}/*.wav"):
            target['emotion'].append(emotion)
            target['path'].append(file)

        n_samples = len(target['path'])
        if verbose:
            print(f"[{dataset_name}] Total files to write:({emotion}) ", n_samples)

        # dividing training/testing sets
        # test_size = int((1-train_size_ratio) * n_samples)
        # train_size = n_samples - test_size

        # if verbose:
        #     print(f"[{dataset_name}] Number of samples:", n_samples)
        #     # print(f"[{dataset_name}] Testing samples:", test_size)
        X.extend(target['path'])
        y.extend(target['emotion'])
        # X_test.extend(target['path'][train_size:])
        # y_test.extend(target['emotion'][train_size:])

    file_name = f"{dataset_name}.csv"
    # test_file_name = f"test_{dataset_name}.csv"
    pd.DataFrame({"path": X, "emotion": y}).to_csv(file_name)
    # pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_file_name)

if __name__ ==  '__main__':
    for dataset in para.datasets:
        write_csv(emotions=para.emotions, dataset_name=dataset, verbose=para.verbose)
