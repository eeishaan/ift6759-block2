'''
Module that contains scoring functions
'''

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, f1
