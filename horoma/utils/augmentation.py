import numpy as np
import torchvision.transforms

def get_augmented_dataset(data):
    """
        Returns a bigger dataset with all images rotated of 90, 180 and 270 degrees.
    """
    nb_images = data.shape[0]
    augmented_dataset = np.zeros((nb_images * 4, data.shape[1], data.shape[2], data.shape[3]), dtype=uint8)
    
    augmented_dataset[0:nb_images] = data
    # first rotation
    augmented_dataset[nb_images+1, nb_images*2] = np.rot90(data, k=1, axes=(1, 2))
    # second rotation
    augmented_dataset[nb_images*2+1, nb_images*3] = np.rot90(data, k=2, axes=(1, 2))
    # third rotation
    augmented_dataset[nb_images*3+1, nb_images*4] = np.rot90(data, k=3, axes=(1, 2))

    return augmented_dataset