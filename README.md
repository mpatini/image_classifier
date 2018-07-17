### AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## Train.py
Command line application that trains a new neural network on a dataset and saves the model as a checkpoint.

# Usage
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu


## Predict.py
Command line application that uses a trained network to predict the class for an input image.

# Usage
Basic usage: python predict.py /path/to/image checkpoint
Options:
    Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    Use GPU for inference: python predict.py input checkpoint --gpu
