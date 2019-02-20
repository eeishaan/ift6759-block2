import matplotlib.pyplot as plt
import numpy as np


def visualize_example(example, label='No label'):
    image = []

    if isinstance(example, np.ndarray):
        image = example.reshape(32, 32, 4)
        image = image[:, :, :3].astype(int)

    plt.title(label)

    plt.imshow(image)
    plt.show()


def grid_pixel_patches(data, grid_dim, save_path=None):
    """Shows a grid of grid_dim*grid_dim pixel patches"""

    fig = plt.figure(figsize=(6, 6))
    fig.suptitle("Pixel Patches examples", fontsize=16)

    for i in range(grid_dim * grid_dim):
        fig.add_subplot(grid_dim, grid_dim, i + 1)
        im, _ = data[np.random.randint(data.data.shape[0])]
        plt.imshow(im[:, :, :3])
        plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path + "Pixel Patches examples")
    plt.show()


def embedding_plot(data, label, title, save_path=None):
    """Plots the transformed data after PCA or t-sne"""

    plt.scatter(data[:, 0], data[:, 1],
                c=label, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Paired', 9))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title(title)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path + title)
