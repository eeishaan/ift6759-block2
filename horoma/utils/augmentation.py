import torchvision.transforms

def get_augmented_dataset(data):
    print(data.shape)
    augmented_dataset = data
    return augmented_dataset