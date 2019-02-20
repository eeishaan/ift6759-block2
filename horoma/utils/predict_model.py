import numpy as np
from sklearn.metrics import accuracy_score


def predict(model, test_data, label_definition_list=None):
    """Method used to predict the labels of test_data using a model.

    :param model: The clustering model to use (required).
    :param test_data: The data to predict labels for (required).
    :param label_definition_list: The list of label definitions (default None).
    :return: cluster label if label_definition_list is not provided and
    detailed labels (specie, density, height) otherwise.
    """

    y_pred = []
    pred_s = []
    pred_d = []
    pred_h = []

    if model:

        # First we apply PCA
        transformed_test_data = model.pca.transform(test_data)

        # We predict the cluster index of each example
        cluster_indices = model.clustering.predict(transformed_test_data)

        if label_definition_list is not None:  # use label definitions if provided

            for cluster_index in cluster_indices:
                predicted_label = model.cluster_labels[cluster_index]
                triplet = label_definition_list[predicted_label - 1]
                pred_s.append(triplet["specie"])
                pred_d.append(triplet["density"])
                pred_h.append(triplet["height"])

            y_pred = np.column_stack((pred_s, pred_d, pred_h))

        else:

            for cluster_index in cluster_indices:
                predicted_label = model.cluster_labels[cluster_index]
                y_pred.append(predicted_label)

            y_pred = np.array(y_pred)

    return y_pred


def get_accuracy(true_labels, predicted_labels):
    return accuracy_score(true_labels, predicted_labels)
