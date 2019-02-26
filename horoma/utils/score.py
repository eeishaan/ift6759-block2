'''
Module that contains scoring functions
'''

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    ari = adjusted_rand_score(y_true, y_pred)
    return accuracy, f1, ari
