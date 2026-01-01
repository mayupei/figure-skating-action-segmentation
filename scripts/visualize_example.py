import argparse
import os

import numpy as np
from file_paths import OUTPUT_PATH, RAW_DATA_PATH
from utils.config import get_config
from utils.visualization import create_video_with_predictions

CONFIG = get_config()
LABEL_DICT = {0: "None", 1: "Jump", 2: "Spin"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run skeleton animation and save it to a file."
    )
    parser.add_argument("--input", type=str, required=True, help="file name")
    parser.add_argument(
        "--split", type=int, required=True, help="integer starting from 0"
    )
    return parser.parse_args()


def create_video_and_save(file_name, split):
    preds = np.load(os.path.join(OUTPUT_PATH, "stage2", f"{file_name}_{split}.npy"))
    labels = np.load(
        os.path.join(OUTPUT_PATH, "ground_truth", f"{file_name}_{split}.npy")
    )
    skeletons = np.load(os.path.join(RAW_DATA_PATH, "features", f"{file_name}.npy"))

    window_length = CONFIG["model_stage1"]["window_length"]
    window_interval = CONFIG["model_stage1"]["window_interval"]

    ### expand preditions and labels
    preds_full = np.zeros(len(skeletons))
    for i, pred in enumerate(preds):
        preds_full[window_length - 1 + window_interval * i] = pred
    for i in range(window_length - 1, len(skeletons), window_interval):
        if preds_full[i] != 0:
            preds_full[i : i + window_interval] = preds_full[i]

    labels_full = np.zeros(len(skeletons))
    for i, label in enumerate(labels):
        labels_full[window_length - 1 + window_interval * i] = label
    for i in range(window_length - 1, len(skeletons), window_interval):
        if labels_full[i] != 0:
            labels_full[i : i + window_interval] = labels_full[i]

    preds_full = [LABEL_DICT[i] for i in preds_full]
    labels_full = [LABEL_DICT[i] for i in labels_full]

    ### rotate skeletons
    skeletons = skeletons[:, :, [2, 0, 1]]
    skeletons[:, :, 2] *= -1

    ### create animation and save
    create_video_with_predictions(
        skeletons, preds_full, labels_full, f"{file_name}_{split}"
    )


if __name__ == "__main__":
    args = parse_args()

    os.makedirs("../examples", exist_ok=True)
    create_video_and_save(args.input, args.split)
