import numpy as np
from .utils import EPS, get_class_labels, get_support


def accuracy(Y_true, Y_pred):
    y_true, y_pred = get_class_labels(Y_true, Y_pred)
    return np.mean(y_true == y_pred)


def precision_weighted(Y_true, Y_pred, num_classes):
    y_true, y_pred = get_class_labels(Y_true, Y_pred)
    supports = get_support(y_true, num_classes)

    weighted_sum = 0.0
    total_support = sum(supports)

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))

        precision_c = tp / (tp + fp + EPS)
        weighted_sum += precision_c * supports[c]

    return weighted_sum / (total_support + EPS)


def recall_weighted(Y_true, Y_pred, num_classes):
    y_true, y_pred = get_class_labels(Y_true, Y_pred)
    supports = get_support(y_true, num_classes)

    weighted_sum = 0.0
    total_support = sum(supports)

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))

        recall_c = tp / (tp + fn + EPS)
        weighted_sum += recall_c * supports[c]

    return weighted_sum / (total_support + EPS)


def top_k_categorical_accuracy(Y_true, Y_pred, k=5):
    y_true_labels = np.argmax(Y_true, axis=1)
    top_k_preds = np.argsort(Y_pred, axis=1)[:, -k:]

    hits = [
        y_true_labels[i] in top_k_preds[i]
        for i in range(len(y_true_labels))
    ]

    return np.mean(hits)


def f1_score_weighted(Y_true, Y_pred, num_classes):
    y_true, y_pred = get_class_labels(Y_true, Y_pred)
    supports = get_support(y_true, num_classes)

    weighted_sum = 0.0
    total_support = sum(supports)

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision_c = tp / (tp + fp + EPS)
        recall_c = tp / (tp + fn + EPS)

        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + EPS)
        weighted_sum += f1_c * supports[c]

    return weighted_sum / (total_support + EPS)
