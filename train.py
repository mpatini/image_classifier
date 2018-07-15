# Imports
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

# Load data
data_dir = 'flower'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define Image Transforms
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

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# Test of above
image, label = next(iter(train_loader))

# Label mapping
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Choose deep learning model
model = models.densenet121(pretrained=True)

# Freeze parameters so no backprop
for param in model.parameters():
    param.requires_grad = False

# Replace deep learning model's classifier
classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(1024, 500)),
                           ('relu1', nn.ReLU()),
                           ('fc2', nn.Linear(500, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))
model.classifier = classifier

# Define deep learning model training
def train_deep(model, train_loader, valid_loader, criterion, 
               optimizer, device='cuda', break_condition=True):
    # Train deep learning network
    epochs = 3
    print_every = 40
    steps = 0
    
    # Change devide
    if device == 'cuda':
        model.to('cuda')
        
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (images, labels) in enumerate(train_loader):
            steps += 1
            
            if device == 'cuda':
                images, labels = images.to('cuda'), labels.to('cuda')
                
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            if break_condition:
                if ii==3:
                    break
        
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valid_loader, criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                running_loss = 0
                
            model.train()

# Implement a function for the validation pass
def validation(model, dataloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    # Change devide
    if device == 'cuda':
        model.to('cuda')
    for images, labels in dataloader:
        
        if device == 'cuda':
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# Define Criteria (TODO move to train file)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train Model (TODO move to train file)
train_deep(model, train_loader, criterion, optimizer)

def test_network(model, testloader, device='cpu'):
    model.eval()

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    if device == 'cuda':
        images, labels = images.to('cuda'), labels.to('cuda')
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(images)

    ps = torch.exp(output) 
    equality = (labels.data == ps.max(dim=1)[1])
    accuracy = equality.type(torch.FloatTensor).mean()
    return accuracy

# Test Network (TODO move to train file)
test_network(model, test_loader)











if __name__ == '__main__':
    pass

