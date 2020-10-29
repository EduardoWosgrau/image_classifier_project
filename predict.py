# PROGRAMMER: Eduardo Wosgrau dos Santos
# DATE CREATED: 10/25/2020
# REVISED DATE: 10/28/2020
# PURPOSE: 
#   Predict flower name from an image with predict.py along with the probability of that name.
#   That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
#   - Basic usage: python predict.py /path/to/image checkpoint
#   - Options:
#      - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
#       - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#       - Use GPU for inference: python predict.py input checkpoint --gpu
# 
#   Example call:
#    python predict.py /path/to/image checkpoint
##

# Imports python modules
from time import time, sleep
from torch.autograd import Variable

# Imports functions created
from get_input_args import get_input_args_predict
import model_utils
import data_image_utils

# Main program
def main():
    start_time = time()
    
    # Handle Arguments
    in_arg = get_input_args_predict()
    print(in_arg)
    
    # Load checkpoint and rebuild network
    model = model_utils.load_checkpoint(in_arg.input, in_arg.gpu)
    # Process image
    image = data_image_utils.process_image(in_arg.image_path)
    # Label mapping
    cat_to_name = data_image_utils.get_label_mapping(in_arg.category_names)
    # Predict
    probs, classes = model_utils.predict(Variable(image).unsqueeze(0), model, in_arg.top_k)
    model_utils.print_prediction(classes, probs, model.class_to_idx, cat_to_name)
    
    tot_time = time()- start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

# Call to main function to run the program
if __name__ == "__main__":
    main()
