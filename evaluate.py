import json
import os
import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras

from file_paths import BASE_PATH, DATA_PATH, OUTPUT_PATH, EVAL_PATH
from utils.config import get_config
from utils.data_prep import stacking_and_split, stage2_data_by_file
from utils.metrics import OverlapF1
from utils.models import masked_accuracy, masked_categorical_crossentropy


def parse_args():
    parser = argparse.ArgumentParser(description="Train the models on the test set of a split.")
    parser.add_argument('--split', type=int, required=True,
                        help='an integer starting from 0')
    return parser.parse_args()


def remove_padding(X, y, pred, mask_value):
    pred = [pred[i] for i in range(len(pred)) if y[i] != mask_value]
    y = [i for i in y if i != mask_value]
    X = [i for i in X if i != mask_value]

    return X, y, pred


def remove_padding_all(X, y, pred, mask_value):
    X_new = []
    y_new = []
    pred_new = []

    N = len(X)
    for i in range(N):
        X_i, y_i, pred_i = remove_padding(X[i], y[i], pred[i], mask_value)
        X_new.append(X_i)
        y_new.append(y_i)
        pred_new.append(pred_i)

    return X_new, y_new, pred_new


def evaluate_one_fold(split):
    config = get_config()

    # stage 1
    files = np.load(os.path.join(DATA_PATH, f"test{split}.npy"))
    X, y, file_len = stacking_and_split(
        files,
        config["model_stage1"]["window_length"],
        config["model_stage1"]["window_interval"],
    )

    model1 = keras.models.load_model(
        os.path.join(BASE_PATH, "models", f"lstm_stage1_{split}.keras")
    )
    pred = model1.predict(X).argmax(axis=-1)

    ### stage 2
    (
        y2,
        X2,
        _,
    ) = stage2_data_by_file(
        y,
        pred,
        file_len,
    )

    model2 = keras.models.load_model(
        os.path.join(BASE_PATH, "models", f"cnn_stage2_{split}.keras"),
        custom_objects={
            "masked_categorical_crossentropy": masked_categorical_crossentropy,
            "masked_accuracy": masked_accuracy,
        },
    )
    pred2 = model2.predict(X2).argmax(axis=-1)

    # remove padding
    X2, y2, pred2 = remove_padding_all(X2, y2, pred2, config["data_prep"]["class_num"])

    # save predictions
    for i, file in enumerate(files):
        np.save(os.path.join(OUTPUT_PATH, "stage1", f"{file}_{split}.npy"), X2[i])
        np.save(os.path.join(OUTPUT_PATH, "ground_truth", f"{file}_{split}.npy"), y2[i])
        np.save(os.path.join(OUTPUT_PATH, "stage2", f"{file}_{split}.npy"), pred2[i])

    # evaluation
    accuracy1 = np.mean([accuracy_score(X2[i], y2[i]) for i in range(len(files))])
    accuracy2 = np.mean([accuracy_score(pred2[i], y2[i]) for i in range(len(files))])
    overlap_f1_1 = np.mean(
        [
            OverlapF1(y2[i], X2[i]).compute_overlap_f1(
                config["data_prep"]["class_num"],
                config["evaluation"]["bg_class"],
                config["evaluation"]["overlap"],
            )
            for i in range(len(files))
        ]
    )
    overlap_f1_2 = np.mean(
        [
            OverlapF1(y2[i], pred2[i]).compute_overlap_f1(
                config["data_prep"]["class_num"],
                config["evaluation"]["bg_class"],
                config["evaluation"]["overlap"],
            )
            for i in range(len(files))
        ]
    )

    eval_dict = {
        "stage1": {"accuracy": accuracy1, "overlap f1": overlap_f1_1},
        "stage2": {"accuracy": accuracy2, "overlap f1": overlap_f1_2},
    }
    with open(os.path.join(EVAL_PATH, f"performance_{split}.json"), "w") as f:
        json.dump(eval_dict, f, indent=4)
        
    return eval_dict


def main():
    config = get_config()
    
    ### evaluate the performance for each fold
    fold_num = config["data_prep"]["k_fold"]
    eval_dicts = []
    for i in range(fold_num):
        eval_dicts.append(evaluate_one_fold(i))
        
    ### take the average across folds
    ave_dict = {}
    for stage in ["stage1", "stage2"]:
        ave_dict[stage] = {}
        for metric in ["accuracy", "overlap f1"]:
            metric_mean = np.mean([d[stage][metric] for d in eval_dicts])
            ave_dict[stage][metric] = metric_mean
            
    with open(os.path.join(EVAL_PATH, "performance_overall.json"), "w") as f:
        json.dump(ave_dict, f, indent=4)
        
if __name__ == "__main__":
    main()
