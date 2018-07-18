import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

__all__ = ["init_model", "classifier", "train_deep",
           "validation", "test_network"]

def init_model(arch, train_data, hidden_units, output_units):
    """
    Initializes a pytorch pretrained deep learning model
    Input: 1 of 2 architectures as a String, tuple ->
    (int, int) number of desired units for one
    hidden layer and the output layer, train_data from which
    to init the models label to idx dict
    Output: model
    """
    # Load model and checkpoint
    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    model.class_to_idx = train_data.class_to_idx
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Freeze parameters so no backprop
    for param in model.parameters():
        param.requires_grad = False

    # Replace deep learning model's classifier
    model.classifier = classifier(arch, hidden_units, output_units)

    return model

# TODO define classifier input for two arch's
def classifier(arch, hidden_units, output_units):
    """
    Defines a classifier for a pretrained model
    Input: 1 of 2 architectures as a String, tuple ->
    (int, int) number of desired units for one
    hidden layer and the output layer
    Output: classifier as an ordered dict
    """
    if arch == 'densenet':
        input_size = 1024
    elif arch == 'alexnet':
        input_size = 9216
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_size, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('fc2', nn.Linear(hidden_units, output_units)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    return classifier

def train_deep(model, train_loader, valid_loader, criterion,
               lr=0.001, epochs=3, device='cpu'):
    """
    Define deep learning model training
    Input: initialized deep learning model,
    train and valid loader as iterators,
    criterion for calculating loss,
    learning rate as float, number of training epochs,
    device 'cpu' or 'cuda' as Strings
    Output: Prints regularly epoch, loss, test loss, and
    test accuracy. Trains model in place
    """

    # Train deep learning network
    print_every = 40
    steps = 0

    # Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    # Change devide
    if device == 'cuda':
        model.to('cuda')
    else:
        model.to('cpu')
        
    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            steps += 1
            
            if device == 'cuda':
                images, labels = images.to('cuda'), labels.to('cuda')
                
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
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

def validation(model, valid_loader, criterion, device='cpu'):
    """
    Validated model during training
    Intput: initialized model, dataloader for validation data,
    criterion for loss calculation, device 'cpu' or 'cuda' as Strings
    Output: test_loss and accuracy of model on validation data 
    """
    test_loss = 0
    accuracy = 0
    # Change devide
    if device == 'cuda':
        model.to('cuda')
    else:
        model.to('cpu')
    for images, labels in valid_loader:
        
        if device == 'cuda':
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def test_network(model, test_loader, criterion, device='cpu'):
    """
    Evaluate model on data unseen during training
    Intput: initialized model, dataloader for testing data,
    criterion for loss calculation, device 'cpu' or 'cuda' as Strings
    Output: prints and returns test_loss and accuracy of model on test data 
    """
    model.eval()

    test_loss = 0
    accuracy = 0
    # Change device
    if device == 'cuda':
        model.to('cuda')
    else:
        model.to('cpu')
    for images, labels in test_loader:
        
        if device == 'cuda':
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    test_loss /= len(test_loader)
    accuracy /= len(test_loader)

    print("Test Loss: {:.3f}.. ".format(test_loss),
          "Test Accuracy: {:.3f}".format(accuracy))
    
    return test_loss, accuracy

