import os

import numpy as np

# note: codes adapted from https://github.com/colincsl/TemporalConvolutionalNetworks


class OverlapF1:
    def __init__(self, y, p):
        self.y = y
        self.p = p

    @staticmethod
    def segment_labels(Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
        return Yi_split

    @staticmethod
    def segment_intervals(Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        return intervals

    def compute_overlap_f1(self, n_classes, bg_class, overlap):
        true_intervals = np.array(self.segment_intervals(self.y))
        true_labels = self.segment_labels(self.y)
        pred_intervals = np.array(self.segment_intervals(self.p))
        pred_labels = self.segment_labels(self.p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels != bg_class]
            true_labels = true_labels[true_labels != bg_class]
            pred_intervals = pred_intervals[pred_labels != bg_class]
            pred_labels = pred_labels[pred_labels != bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, float)
        FP = np.zeros(n_classes, float)
        true_used = np.zeros(n_true, float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(
                pred_intervals[j, 1], true_intervals[:, 1]
            ) - np.maximum(pred_intervals[j, 0], true_intervals[:, 0])
            union = np.maximum(pred_intervals[j, 1], true_intervals[:, 1]) - np.minimum(
                pred_intervals[j, 0], true_intervals[:, 0]
            )
            IoU = (intersection / union) * (pred_labels[j] == true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1

        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1
