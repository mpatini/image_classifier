# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Train.py
Command line application that trains a new neural network on a dataset and saves the model as a checkpoint.

### Usage
Basic usage: python train.py data_directory  
Prints out training loss, validation loss, and validation accuracy as the network trains  
- Options:  
    - Use previous checkpoint for continued training: python train.py dir --checkpoint my_recent_checkpoint.pth (no default)
    - Provide optional learning rate: python train.py dir --lr 0.01 (default: 0.001)
    - Provide optional number of epochs: python train.py dir --epochs 5 (default: 3)
    - Provide optional number hidden and output units as tuple: python train.py dir --units (1000, 12) (default: (500, 102))
    - Use GPU for inference: python train.py dir --device cuda (default: cpu)
    - Choose between densenet or alexnet for modeling: python predict.py input checkpoint --arch alexnet (default: densenet)
    - Provide path/name.pth of new checkpoint: python train.py dir --newcheckpoint my_new_checkpoint.pth (default: checkpoint.pth)


## Predict.py
Command line application that uses a trained network to predict the class for an input image.  

### Usage
Basic usage: python predict.py /path/to/image checkpoint  
* Options:  
    - Return top KK most likely classes: python predict.py input checkpoint --topk 3 (default: 5)  
    - Use a mapping of categories to real names: python predict.py input checkpoint --labels categories_to_names.json  (default: cat_to_names.json)
    - Use GPU for inference: python predict.py input checkpoint --device cuda (default: cpu
    - Choose between densenet or alexnet for modeling: python predict.py input checkpoint --arch alexnet (default: densenet)
