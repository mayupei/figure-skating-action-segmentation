import os

import numpy as np
from sklearn.model_selection import KFold

from file_paths import DATA_PATH, RAW_DATA_PATH
from utils.config import get_config

def main():
    config = get_config()
    k_fold = config["data_prep"]["k_fold"]
    seed = config["data_prep"]["seed"]
    
    files = os.listdir(os.path.join(RAW_DATA_PATH, "labels"))
    files = np.array([file.split(".")[0] for file in files])

    kf = KFold(n_splits = k_fold, shuffle = True, random_state = seed)

    for i, (train_index, test_index) in enumerate(kf.split(files)):
        train = files[train_index]
        test = files[test_index]

        np.save(os.path.join(DATA_PATH, f"train{i}.npy"), train)
        np.save(os.path.join(DATA_PATH, f"test{i}.npy"), test)


if __name__ == "__main__":
    main()
