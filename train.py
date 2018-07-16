
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
    train_data, train_loader, valid_loader, test_loader = dataloader(dirs, trans)
    
    # Import label mapping
    labels = write_labels(in_args.labels)

    # Initialize model
    if in_args.checkpoint:
        model = load_checkpoint(in_args.checkpoint, in_args.arch, in_args.device)
    else:
        model = init_model(in_args.arch, train_data, in_args.units)

    # Define global criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.lr)

    # Train model
    train_deep(model, train_loader, valid_loader, criterion,
               in_args.lr, in_args.epochs, in_args.device)

    # Save model as checkpoint
    save_checkpoint(model, in_args.newcheckpoint)
    

# Call to main function to run the program
if __name__ == "__main__":
    main()