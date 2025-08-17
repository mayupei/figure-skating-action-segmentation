import os
import argparse

import numpy as np
from tensorflow.keras.callbacks import CSVLogger

from file_paths import BASE_PATH, DATA_PATH
from utils.config import get_config
from utils.data_prep import stacking_and_split, stage2_data_by_file
from utils.models import build_lstm_classifier, cnn_stage2


def parse_args():
    parser = argparse.ArgumentParser(description="Train the models on the training set of a split.")
    parser.add_argument('--split', type=int, required=True,
                        help='an integer starting from 0')
    return parser.parse_args()

def main(split):
    config = get_config()

    # stage 1 - lstm
    files = np.load(os.path.join(DATA_PATH, f"train{split}.npy"))
    X, y, file_len = stacking_and_split(
        files,
        config["model_stage1"]["window_length"],
        config["model_stage1"]["window_interval"],
    )

    model = build_lstm_classifier(
        seq_len=config["model_stage1"]["window_length"],
        feature_dim=51,
        hidden_size=config["model_stage1"]["hidden_size"],
        num_classes=config["data_prep"]["class_num"],
    )
    
    stage1_log = CSVLogger(f"logs/stage1_training_{split}.csv", append=False)
    model.fit(
        X,
        y,
        epochs=config["model_stage1"]["epochs"],
        batch_size=config["model_stage1"]["batch_size"],
        validation_split=config["model_stage1"]["validation_split"],
        callbacks=[stage1_log]
    )
    model.save(os.path.join(BASE_PATH, "models", f"lstm_stage1_{split}.keras"))

    ### stage 2
    pred = [np.argmax(i) for i in model.predict(X)]
    (
        y2,
        X2,
        y2_onehot,
    ) = stage2_data_by_file(
        y,
        pred,
        file_len,
    )

    model2 = cnn_stage2(
        num_classes=config["data_prep"]["class_num"],
        embedding_dim=config["model_stage2"]["embedding_dim"],
        sequence_length=config["model_stage2"]["max_sequence_len_stage2"],
    )
    
    stage2_log = CSVLogger(f"logs/stage2_training_{split}.csv", append=False)
    model2.fit(
        X2,
        y2_onehot,
        batch_size=config["model_stage2"]["batch_size"],
        epochs=config["model_stage2"]["epochs"],
        validation_split=config["model_stage2"]["validation_split"],
        callbacks=[stage2_log]
    )
    model2.save(os.path.join(BASE_PATH, "models", f"cnn_stage2_{split}.keras"))


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs("logs", exist_ok=True)
    main(args.split)
