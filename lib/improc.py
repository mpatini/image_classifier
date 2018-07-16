# Imports
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

__all__ = [group_transform, process_image, imshow]


def group_transform():
    """
    Defines image transforms for training and testing.
    Output: tuple -> train_transforms, test_transforms
    """
    train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomGrayscale(.1),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return train_transforms, test_transforms

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
        Input: path to individual image.
        Output: Pil Image
    '''
    image = Image.open(image_path)
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = test_transforms(image)
    return image

def imshow(image, ax=None, title=None):
    """
    Imshow for Tensor.
    Input: image processed by process_image(),
    an optional pyplot ax and title
    Output: image display to user
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
            
    ax.imshow(image)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax

