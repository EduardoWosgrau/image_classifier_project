# PROGRAMMER: Eduardo Wosgrau dos Santos
# DATE CREATED: 10/25/2020
# REVISED DATE: 10/28/2020
# PURPOSE: 
#
##

# Imports python modules
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
    
def build_model(arch = 'vgg16', input_size = 25088, output_size = 256, hidden_units = 512, is_gpu = False):
    """
    """
    if (arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif (arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
    else:
        print('Not implemented Architecture')
        return false
    
    device = torch.device("cuda" if is_gpu else "cpu")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    p_dropout = 0.5
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=p_dropout),
                               nn.Linear(hidden_units, output_size),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.to(device);
    return model

def train_model(trainloader, validloader, model, learning_rate = 0.001, epochs = 2, is_gpu = False):
    """
    """
    steps = 0
    running_loss = 0
    print_every = 10
    
    device = torch.device("cuda" if is_gpu else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
        model.train()
    print("End of trainning.")
    return
    
def test_model(model, testloader, is_gpu = False):
    """
    """
    device = torch.device("cuda" if is_gpu else "cpu")
    criterion = nn.NLLLoss()
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
    return

def save_checkpoint(model, train_data, input_size = 25088, output_size = 256, hidden_units = 512, 
                    arch = 'vgg16', epochs = 2, learning_rate = 0.001, save_dir = ''):
    """
    """
    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'input_size': input_size,
        'output_size': output_size,
        'hidden_units': hidden_units,
        'arch': arch,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'map': model.class_to_idx
    }, save_dir + 'checkpoint.pth')
    return True

def load_checkpoint(filepath = '', is_gpu = False):
    """
    """
    device = torch.device("cuda" if is_gpu else "cpu")

    checkpoint = torch.load(filepath + 'checkpoint.pth')
    
    if (checkpoint['arch'] == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif (checkpoint['arch'] == 'vgg13'):
        model = models.vgg13(pretrained=True)
    else:
        print('Not implemented Architecture')
        return false

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['map']

    model.to(device);
    return model
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    output = model.forward(image_path)
    output = torch.exp(output)
    top_p, top_class = output.topk(topk, dim=1)
    return top_p, top_class

def print_prediction(classes, probs, class_to_idx, cat_to_name):
    """
    """
    # Index to Class
    idx_to_class = dict()
    for classe, idx in class_to_idx.items():
        idx_to_class[idx] = classe
    
    prediction_class_ps = []
    labels = classes.data.numpy()[0].astype(str)
    for i in range(len(probs[0])):
        prediction_class_ps.append({
            'class_name': cat_to_name[idx_to_class[labels[i].astype(int)]],
            'probability': probs[0][i].item()
        })
    print('Prediction:')
    print(prediction_class_ps)
    return