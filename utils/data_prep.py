import os

import numpy as np
from scipy.spatial.distance import euclidean
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from file_paths import RAW_DATA_PATH
from utils.config import get_config

SELECTED_BONES = [
    (5, 7),  # Left Shoulder → Left Elbow
    (7, 9),  # Left Elbow → Left Wrist
    (6, 8),  # Right Shoulder → Right Elbow
    (8, 10),  # Right Elbow → Right Wrist
    (13, 15),  # Left Knee → Left Ankle
    (14, 16),  # Right Knee → Right Ankle
]
LABEL_DICT = {"NONE": 0, "Jump": 1, "Spin": 2, "Sequence": 0}

CONFIG = get_config()
CLASS_NUM = CONFIG["data_prep"]["class_num"]
MAX_LEN_STAGE2 = CONFIG["model_stage2"]["max_sequence_len_stage2"]

class SkeletonNormalization:
    def __init__(self, frames, bones):
        """
        frames: numpy array of shape (T, J, 3)
        """
        self.frames = frames
        self.frame_length = frames.shape[0]
        self.bones = bones

    @staticmethod
    def compute_distance(frame, bones):
        dis = 0
        for b in bones:
            dis += euclidean(frame[b[0]], frame[b[1]])

        return dis

    @staticmethod
    def center_location_one_frame(frame):
        frame_mean = np.mean(frame, axis=0)

        return frame - frame_mean

    def scale_by_y_distance(self):
        y_diffs = [self.compute_distance(f, self.bones) for f in self.frames]
        self.frames = np.stack(
            [self.frames[i] / y_diffs[i] for i in range(self.frame_length)]
        )

    def center_location(self):
        self.frames = np.stack([self.center_location_one_frame(f) for f in self.frames])

    def normalize(self):
        self.scale_by_y_distance()
        self.center_location()

        return self.frames


def sliding_window(skeleton_seq, label_seq, window_length, window_interval):
    if len(skeleton_seq) != len(label_seq):
        raise ValueError("skeleton and label must be of equal length.")

    features = []
    labels = []
    for i in range(0, len(skeleton_seq), window_interval):
        if i + window_length <= len(skeleton_seq):
            features.append(skeleton_seq[i : i + window_length])
            labels.append(label_seq[i + window_length - 1])

    if len(features) != len(labels):
        raise ValueError("different lengths for feature slides and label slides.")

    return np.stack(features), np.stack(labels)


def video_to_windows(file_name, window_length, window_interval):
    with open(
        os.path.join(RAW_DATA_PATH, "labels", f"{file_name}.txt"),
        "r",
    ) as f:
        labels = f.read()
    labels = labels.splitlines()

    skeletons = np.load(os.path.join(RAW_DATA_PATH, "features", f"{file_name}.npy"))
    skeletons = SkeletonNormalization(skeletons, SELECTED_BONES).normalize()
    skeletons, labels = sliding_window(
        skeletons, labels, window_length, window_interval
    )
    skeletons = skeletons.reshape(-1, window_length, 17 * 3)
    labels = np.stack([LABEL_DICT[i] for i in labels])

    return skeletons, labels


def stacking_and_split(files, window_length, window_interval):
    X = []
    y = []
    file_len = []
    for file in files:
        skeletons, labels = video_to_windows(file, window_length, window_interval)
        X.append(skeletons)
        y.append(labels)
        file_len.append(labels.shape[0])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    file_len = np.cumsum(file_len)

    return X, y, file_len


def stage2_data_by_file(
    y,
    pred,
    file_len,
):
    y_split = [y[: file_len[0]]]
    pred_split = [pred[: file_len[0]]]
    prev_idx = 0
    for i in range(1, len(file_len)):
        y_split.append(y[file_len[prev_idx] : file_len[i]])
        pred_split.append(pred[file_len[prev_idx] : file_len[i]])
        prev_idx = i

    ### pad sequences
    y_split = pad_sequences(
        y_split, padding = "post", maxlen = MAX_LEN_STAGE2, truncating = "post", value = CLASS_NUM
    )
    pred_split = pad_sequences(
        pred_split, padding = "post", maxlen = MAX_LEN_STAGE2, truncating = "post", value= CLASS_NUM
    )
    y_onehot = to_categorical(y_split, num_classes = CLASS_NUM + 1)
    y_onehot = y_onehot[:, :, :CLASS_NUM]

    return (
        y_split,
        pred_split,
        y_onehot,
    )
