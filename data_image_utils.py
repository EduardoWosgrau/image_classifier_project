# PROGRAMMER: Eduardo Wosgrau dos Santos
# DATE CREATED: 10/25/2020
# REVISED DATE: 10/28/2020
# PURPOSE: 
#
##

# Imports python modules
import torch
from torchvision import datasets, transforms, models
from torch import optim
from PIL import Image
import json

def get_label_mapping(label_map_path = 'cat_to_name.json'):
    """
    """
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    return label_map

def load_data(data_dir = 'flowers'):
    """
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    return train_data, trainloader, validloader, testloader

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL    image for a PyTorch model,      returns an Numpy array
    '''
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    image = transform(image).float()
    return image
