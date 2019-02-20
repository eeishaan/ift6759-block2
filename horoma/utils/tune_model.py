import argparse
import os

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid, train_test_split
from torch import save

import horoma.utils.build_features as build_features
import horoma.utils.load_dataset as load_dataset
import horoma.utils.predict_model as predict_model
import horoma.utils.train_model as train_model


class ModelTuner:
    """ModelTuner is a helper class used to tune the model by finding
    the best values for the hyper-parameters based on some metrics.

    The parameters of the initialization methods are:
        pca_components: The number of components to keep when doing PCA .
        n_clusters: The number of clusters to pick.
        clustering_obj: The fitted scikit-learn clustering object.
        pca_obj: The fitted scikit-learn pca object.
        predicted_clusters: predicted cluster indices.
    """

    def __init__(self, n_components=None, n_clusters=None):
        self.pca_components = n_components
        self.n_clusters = n_clusters
        self.clustering_obj = None
        self.pca_obj = None
        self.predicted_clusters = None

    def fit(self, data):
        transformed_data, self.pca_obj = build_features.fit_apply_pca(
            data, self.pca_components)
        self.clustering_obj = train_model.train_clustering(
            transformed_data, self.n_clusters, clustering_type)
        self.predicted_clusters = self.clustering_obj.predict(transformed_data)

        return

    def score(self, true_labels):
        score_results = {"n_components": self.pca_components,
                         "n_clusters": self.n_clusters}

        y_pred = []

        # To have an unbiased estimation we split the predicted clusters into train/test sets.
        predicted_clusters_train, predicted_clusters_test, true_labels_train, true_labels_test = train_test_split(
            self.predicted_clusters, true_labels, test_size=0.6, random_state=2019, stratify=true_labels)

        # We use the train split to label the clusters.
        cluster_labels = train_model.get_cluster_labels(predicted_clusters_train, true_labels_train,
                                                        self.n_clusters)

        # The test split will be used to calculate the accuracy.
        for cluster_index in predicted_clusters_test:
            predicted_label = cluster_labels[cluster_index]
            y_pred.append(predicted_label)

        accuracy = predict_model.get_accuracy(
            np.array(y_pred), true_labels_test)

        score_results['accuracy'] = accuracy

        score_results['ARI'] = adjusted_rand_score(
            valid_labels, self.predicted_clusters)

        return score_results


def tune_model(params, save_path):
    np.random.seed(2019)

    # The list where we will save the search results
    grid_search_results = []

    for param_combination in list(ParameterGrid(params)):
        m = ModelTuner(
            param_combination['n_components'], param_combination['n_clusters'])
        m.fit(valid_data)
        grid_search_results.append(m.score(valid_labels))

    print(grid_search_results)

    print("The grid search results were saved in file {}".format(save_path))

    with open(save_path, 'wb') as f:
        save(grid_search_results, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The script used to find the best hyper-parameters of the model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use_pca", type=str, choices=['true', 'false'],
                        help='Specifies if we will use PCA before clustering the data.', default='true')

    parser.add_argument("--clustering_type", type=str, choices=['Kmeans', 'GMM'],
                        help="Type of models to be used for clustering",
                        default='Kmeans')

    parser.add_argument("--n_components_list", nargs='+', type=int,
                        help="list of values for number of components to use in PCA.",
                        metavar='', default=[5, 10, 20])
    parser.add_argument("--n_clusters_list", nargs='+', type=int,
                        help="list of values for number of clusters to for the model.",
                        metavar='', default=[10, 20, 30])

    parser.add_argument("--params_save_path", type=str, help="The full path to the file where the parameters "
                                                             "will be saved", metavar='')

    args = parser.parse_args()

    clustering_type = args.clustering_type
    params_save_path = args.params_save_path

    if params_save_path is None or not os.path.exists(params_save_path):
        # Retrieve the absolute path to the project root
        project_root = os.path.abspath(os.path.join(os.path.join(
            os.path.realpath(__file__), os.pardir), os.pardir))
        params_save_path = os.path.join(
            project_root, 'models/saved/best_params.sav')

    n_components_params = args.n_components_list
    n_clusters_params = args.n_clusters_list

    params_dict = {'n_components': n_components_params,
                   'n_clusters': n_clusters_params}

    # Doing a grid search for hyper-parameters takes too long on the training dataset.
    # We will use the validation dataset to do the grid search instead.
    validation_dataset = load_dataset.HoromaDataset(
        load_dataset.DataSplit.VALIDATION)
    valid_data = validation_dataset.to_2d_array()
    valid_labels = validation_dataset.get_labels()

    tune_model(params_dict, params_save_path)
