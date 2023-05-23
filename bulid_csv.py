from modules.CSVManagement.main import write_csv
import parameters as para

for dataset in para.datasets:
    write_csv(emotions=para.emotions, dataset_name=dataset,
              train_size_ratio=para.train_size_ratio, verbose=para.verbose)
