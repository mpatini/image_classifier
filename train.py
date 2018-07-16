
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import helper
from collections import OrderedDict
from PIL import Image

from lib.dataload import load_dir, dataloader, write_labels
from lib.improc import group_transform
from lib.deeplearn import init_model, classifier, train_deep, validation, test_network
from lib.checkpoint import save_checkpoint, load_checkpoint
from lib.args import get_train_args


def main():
    # Get command line arguments
    in_args = get_train_args()

    # Define transforms
    trans = group_transform()

    # Create dataloaders
    dirs = load_dir(in_args.dir)
    train_loader, valid_loader, test_loader = dataloader(dirs, trans)
    
    # Import label mapping
    labels = write_labels(in_args.labels)

    # Initialize model
    

    # Define global criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.lr)







# Call to main function to run the program
if __name__ == "__main__":
    main()