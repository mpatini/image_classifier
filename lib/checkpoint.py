import torch
from torchvision import datasets, transforms, models

__all__ = [save_checkpoint, load_checkpoint]


def save_checkpoint(model, file_name):
    """
    Saves trained model as checkpoint
    Input: model, and filename (w/o .pth extension) to save it as
    Output: saves model to file, no return
    """
    file_name += '.pth'
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'idx_to_class': model.idx_to_class}

    torch.save(checkpoint, file_name)

def load_checkpoint(filepath, arch, device='cpu'):
    """
    Takes an initialized model and inloads previous checkpoint
    Input: model, filepath to checkpoint saved data, device
    cpu or gpu
    Output: returns presaved model
    """
    # Load model and checkpoint
    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    if device == 'cuda':
        model.to('cuda')
    checkpoint = torch.load(filepath)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Update model using classifier and state_dict
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']

    return model