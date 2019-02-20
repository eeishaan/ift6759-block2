import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def train_clustering(data_2d_array, n_clusters, clustering_type):
    clustering_obj = None

    if clustering_type == 'Kmeans':
        clustering_obj = KMeans(n_clusters, random_state=0, n_jobs=-1).fit(data_2d_array)

    elif clustering_type == 'GMM':
        clustering_obj = GaussianMixture(n_clusters, random_state=0).fit(data_2d_array)

    return clustering_obj


def get_cluster_labels(labeled_data_clusters, labels, n_clusters):
    cluster_labels = []

    for i in range(n_clusters):
        filtered_labels = labels[np.where(labeled_data_clusters == i)]

        if len(filtered_labels) > 0:
            counts = np.bincount(filtered_labels)
            cluster_labels.append(np.argmax(counts))
        else:
            cluster_labels.append(-1)  # No validation point found on this cluster. We can't label it.

    return np.array(cluster_labels)
