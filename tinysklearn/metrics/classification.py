import numpy as np


def confusion_matrix(y_true, y_pred):
    tp = tn = fp = fn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    return [[tn, fp],
            [fn, tp]]


def accuracy_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)


def recall_score(y_true, y_pred):
    tp = fn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def precision_score(y_true, y_pred):
    tp = fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1

    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def focal




