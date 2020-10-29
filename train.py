# PROGRAMMER: Eduardo Wosgrau dos Santos
# DATE CREATED: 10/25/2020
# REVISED DATE: 10/28/2020
# PURPOSE: 
#   Train a new network on a data set with train.py
#   - Basic usage: python train.py data_directory
#   - Prints out training loss, validation loss, and validation accuracy as the network trains
#   - Options:
#       - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#       - Choose architecture: python train.py data_dir --arch "vgg13"
#       - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#       - Use GPU for training: python train.py data_dir --gpu
# 
#   Example call:
#    python train.py data_directory
##

# Imports python modules
from time import time, sleep

# Imports functions created
from get_input_args import get_input_args_train
import model_utils
import data_image_utils

# Main program
def main():
    start_time = time()
    
    # Handle Arguments
    in_arg = get_input_args_train()
    print(in_arg)
    # Load Data
    train_data, trainloader, validloader, testloader = data_image_utils.load_data(in_arg.data_dir)
    # Build model
    input_size = 25088
    output_size = 256
    model = model_utils.build_model(in_arg.arch, input_size, output_size, in_arg.hidden_units, in_arg.gpu)
    # Train model
    model_utils.train_model(trainloader, validloader, model, in_arg.learning_rate, in_arg.epochs, in_arg.gpu)
    # Save checkpoint
    model_utils.save_checkpoint(model, train_data, input_size, output_size, in_arg.hidden_units, 
                                in_arg.arch, in_arg.epochs, in_arg.learning_rate, in_arg.save_dir)
    
    tot_time = time()- start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
