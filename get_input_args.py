# PROGRAMMER: Eduardo Wosgrau dos Santos
# DATE CREATED: 10/25/2020
# REVISED DATE: 10/28/2020
# PURPOSE: 
#
##

# Imports python modules
import argparse

def get_input_args_train():
    """
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str, default = 'flowers/', help = 'Data Directory')
    parser.add_argument('--save_dir', type = str, default = '', help = 'Directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Number of hidden units')
    parser.add_argument('--epochs', type = int, default = 2, help = 'Number of epochs')
    parser.add_argument('--gpu', action='store_true', help = 'GPU Mode')
    
    return parser.parse_args()


def get_input_args_predict():
    """
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, default = '', help = 'Path to Image')
    parser.add_argument('--input', type = str,  default = '', help = 'Directory of checkpoint for restoration')
    parser.add_argument('--top_k', type = int, default = 3, help = 'Top k most likely classes')
    parser.add_argument('--category_names', type = str, default = '', help = 'Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help = 'GPU Mode')
    
    return parser.parse_args()