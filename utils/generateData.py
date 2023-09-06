import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from torch import tensor, Tensor
import torchvision
from tqdm import tqdm
import os
from torchvision import transforms as tortra

def prepare_data():
    # Define the transform function
    
    transform = tortra.Compose([
        tortra.ToTensor(),
        tortra.Normalize((0.1307,), (0.3081,))
        ])

    # Load the train MNIST dataset
    train_mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform,
                                                     download=False)
    n_train_samples = len(train_mnist_dataset)
    # Load the test MNIST dataset
    test_mnist_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform,
                                                    download=False)

    if not os.path.exists("transformed_dataset.pt"):
        random_pairs = np.random.randint(n_train_samples, size=[n_train_samples, 2])
        random_pairs = [(row[0], row[1]) for row in random_pairs]

        # Transform the data
        transformed_dataset = [
            create_negative_image(train_mnist_dataset[pair[0]][0].squeeze(), train_mnist_dataset[pair[1]][0].squeeze())
            for pair in tqdm(random_pairs)]

        # Save the transformed images to a folder
        torch.save(transformed_dataset, './data/transformed_dataset.pt')


def create_mask(shape, iterations: int = 10):
    """
    Create a binary mask as described in (Hinton, 2022): start with a random binary image and then repeatedly blur
    the image with a filter, horizontally and vertically.

    Parameters
    ----------
    shape : tuple
        The shape of the output mask (height, width).
    iterations : int
        The number of times to blur the image.

    Returns
    -------
    numpy.ndarray
        A binary mask with the specified shape, containing fairly large regions of ones and zeros.
    """

    blur_filter_1 = np.array(((0, 0, 0), (0.25, 0.5, 0.25), (0, 0, 0)))
    blur_filter_2 = blur_filter_1.T

    # Create a random binary image
    image = np.random.randint(0, 2, size=shape)

    # Blur the image with the specified filter
    for i in range(iterations):
        image = np.abs(convolve2d(image, blur_filter_1, mode='same') / blur_filter_1.sum())
        image = np.abs(convolve2d(image, blur_filter_2, mode='same') / blur_filter_2.sum())

    # Binarize the blurred image, i.e. threshold it at 0.5
    mask = np.round(image).astype(np.uint8)

    return tensor(mask)


def create_negative_image(image_1: Tensor, image_2: Tensor):
    """
    Create a negative image by combining two images with a binary mask.

    Parameters:
    image_1 (Tensor): The first image to be combined.
    image_2 (Tensor): The second image to be combined.

    Returns:
    Tensor: The negative image created by combining the two input images.

    Raises:
    AssertionError: If the shapes of `image_1` and `image_2` are not the same.

    Examples:
    >>> image_1 = np.random.randint(0, 2, size=(5, 5))
    >>> image_2 = np.random.randint(0, 2, size=(5, 5))
    >>> create_negative_image(image_1, image_2)
    array([[0 0 0 0 1]
           [1 1 0 1 1]
           [0 0 0 1 1]
           [0 1 1 1 0]
           [1 1 0 0 1]])
    """

    assert image_1.shape == image_2.shape, "Incompatible images and mask shapes."

    mask = create_mask((image_1.shape[0], image_1.shape[1]))

    image_1 = torch.mul(image_1, mask)
    image_2 = torch.mul(image_2, 1 - mask)

    return torch.add(image_1, image_2)


def create_negative_batch(images: Tensor):
    neg_imgs = []
    batch_size = images.shape[0]
    for _ in range(batch_size):
        idx1, idx2 = np.random.randint(batch_size, size=2)
        neg_imgs.append(create_negative_image(images[idx1].squeeze(), images[idx2].squeeze()))
    return torch.unsqueeze(torch.stack(neg_imgs), dim=1)