from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import load, save

from horoma.constants import PCA_MODEL_DEFAULT_PATH, TSNE_MODEL_DEFAULT_PATH


def fit_apply_pca(data_2d_array, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data_2d_array), pca


def fit_pca(data_2d_array, n_components, save_model=False, save_full_path=PCA_MODEL_DEFAULT_PATH):
    pca = PCA(n_components=n_components)
    pca = pca.fit(data_2d_array)

    if save_model:
        with open(save_full_path, 'wb') as f:
            save(pca, f)

    return pca


def apply_pca(data_2d_array, pca_model):
    if pca_model:
        return pca_model.transform(data_2d_array)


def load_pca(pca_model_savefile):
    with open(pca_model_savefile, 'rb') as f:
        loaded_pca_model = load(f)

    return loaded_pca_model


def fit_apply_tsne(data_2d_array, n_components):
    tsne = TSNE(n_components=n_components, n_iter=300)
    return tsne.fit_transform(data_2d_array)


def fit_tsne(data_2d_array, n_components, save_model=False, save_full_path=TSNE_MODEL_DEFAULT_PATH):
    tsne = TSNE(n_components=n_components, n_iter=300)
    tsne = tsne.fit(data_2d_array)

    if save_model:
        with open(save_full_path, 'wb') as f:
            save(tsne, f)

    return tsne


def apply_tsne(data_2d_array, tsne_model):
    if tsne_model:
        return tsne_model.transform(data_2d_array)


def load_tsne(tsne_model_savefile):
    with open(tsne_model_savefile, 'rb') as f:
        loaded_tsne_model = load(f)

    return loaded_tsne_model
