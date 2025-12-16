import numpy as np

EPS = 1e-12

def get_class_labels(Y_true, Y_pred):
    y_true_labels = np.argmax(Y_true, axis=1)
    y_pred_labels = np.argmax(Y_pred, axis=1)
    return y_true_labels, y_pred_labels


def get_support(y_true_labels, num_classes):
    return [np.sum(y_true_labels == c) for c in range(num_classes)]
