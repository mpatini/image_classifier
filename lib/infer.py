
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import lib.improc as ip

__all__ = ["predict", "plot_probs", "output"]


def predict(image_path, model, label_map, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained
    deep learning model.
    Input: path to one image, fully loaded deep learning model,
    number of final results to give
    Output: topk probabilities the image is a certain category,
    and the matching category labels sorted from most to least likely
    """
    # Load Image and set module to eval
    model.eval()
    image = ip.process_image(image_path)
    image = ip.np.array(image)
    image = image.transpose((0, 2, 1))
    image = torch.FloatTensor(image)
    image.unsqueeze_(0)

    # Change device
    if device == 'cuda':
        model.to('cuda')
    else:
        model.to('cpu')
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(image)
    probs_inc, classes = torch.topk(output, 5)
    probs_inc, classes = np.array(probs_inc)[0], np.array(classes)[0]
    
    # Turn probs and classes into python lists of type prob and label
    labels = []
    for idx in classes:
        new_idx = str(model.idx_to_class[idx])
        labels.insert(0, label_map[new_idx])
    probs = []
    for prob in probs_inc:
        probs.insert(0, np.exp(prob))
    data = dict(zip(probs, labels))
    probs_sorted = sorted(data.keys(), reverse=True)
    labels_sorted = []
    for prob in probs_sorted:
        labels_sorted.append(data[prob])
    return probs_sorted, labels_sorted

def plot_probs(probs, labels, image): 
    """
    Displays image and topk probabilities as a horizontal barchart
    Input: probs, labels from predict(), Pil image
    Output: displays image and probs, no return
    """
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5, 9))
    # image on top
    ax1.set_title(labels[0])
    ip.imshow(image, ax=ax1, title=labels[0])
    # plot on bottom
    ax2.barh(y=labels, width=probs)
    plt.yticks(rotation = 25)
    fig.tight_layout(pad=2)
    plt.show()

def output(image_path, label_map, model, device):
    """
    Takes processed image, forms prediction, and displays final output
    Input: image_path and a deep learning model
    Output: display image and topk probs as a barchart,
    no return
    """
    image = ip.process_image(image_path)
    probs, labels = predict(image_path, model, label_map, device)
    plot_probs(probs, labels, image)