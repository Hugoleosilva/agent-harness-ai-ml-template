"""Evaluation layer. Report metrics suited to the task, not accuracy alone."""
import numpy as np


def f1_binary(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
