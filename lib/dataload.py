# Imports
import json
import torch
from torchvision import datasets, transforms, models

#__all__ = [load_dir, dataloader, write_labels]


def load_dir(path):
    """
    Loads the image to be used in training, validating, and testing.
    Input: path as a String, to the parent folder
    Output: three directories, that are subfolders of the parent
    """
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    return train_dir, valid_dir, test_dir

def dataloader(dir, transforms):
    """
    Loads image data from directories
    Input: Three dirs train, valid, and test as a tuple, and a tuple -> 
    (train, test) transforms
    Output: Three dataloaders, trainloader, validloader testloader
    Example of usage:
        image, label = next(iter(train_loader))
    """

    train_dir, valid_dir, test_dir = dir

    # Load the data
    train_data = datasets.ImageFolder(train_dir, transform=transforms[0])
    valid_data = datasets.ImageFolder(valid_dir, transform=transforms[1])
    test_data = datasets.ImageFolder(test_dir, transform=transforms[1])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    return train_data, train_loader, valid_loader, test_loader

def write_labels(path):
    """
    Loads categorical labels for image data from json file
    Input: path to json file
    Output: dictionary
    """
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name